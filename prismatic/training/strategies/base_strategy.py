"""
base_strategy.py

Shared optimizer, checkpoint, and fixed VLA training-loop logic for the FSDP strategy.
"""

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import VLAMetrics
from prismatic.training.robotwin_performance import (
    CudaPhaseTimer,
    OptimizerStepTimingWindow,
    estimate_robotwin_training_flops,
    estimated_mfu,
    should_sample_performance,
)
from prismatic.util import check_bloat16_supported
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


_VLA_FORWARD_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pair_pixel_values",
    "world_input_ids",
    "world_attention_mask",
    "current_frame_pairs",
    "future_frames",
    "actions",
    "proprio",
    "dataset_names",
)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _record_batch_stream(batch, stream) -> None:
    if isinstance(batch, torch.Tensor):
        batch.record_stream(stream)
    elif isinstance(batch, dict):
        for value in batch.values():
            _record_batch_stream(value, stream)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            _record_batch_stream(value, stream)


class CUDAPrefetcher:
    """Transfer the next pinned-memory batch on a dedicated CUDA stream."""

    def __init__(self, loader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield move_batch_to_device(batch, self.device)
            return

        iterator = iter(self.loader)
        prefetch_stream = torch.cuda.Stream(device=self.device)
        next_batch = None

        def preload() -> bool:
            nonlocal next_batch
            try:
                host_batch = next(iterator)
            except StopIteration:
                next_batch = None
                return False
            with torch.cuda.stream(prefetch_stream):
                next_batch = move_batch_to_device(host_batch, self.device)
            return True

        preload()
        while next_batch is not None:
            consumer_stream = torch.cuda.current_stream(self.device)
            consumer_stream.wait_stream(prefetch_stream)
            batch = next_batch
            _record_batch_stream(batch, consumer_stream)
            preload()
            yield batch


def build_vla_forward_kwargs(batch: dict) -> dict:
    return {key: batch[key] for key in _VLA_FORWARD_KEYS if key in batch and batch[key] is not None}


def get_cosine_schedule_with_warmup_and_group_min_lrs(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Cosine schedule with per-parameter-group LR floors."""
    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group["lr"])
        min_lr = group.get("min_lr", 0.0)

        if base_lr <= 0:
            raise ValueError(f"`base_lr` must be positive, got {base_lr}.")
        if min_lr < 0:
            raise ValueError(f"`min_lr` must be non-negative, got {min_lr}.")
        if min_lr > base_lr:
            raise ValueError(f"`min_lr` ({min_lr}) must not exceed `base_lr` ({base_lr}).")

        floor_ratio = min_lr / base_lr

        def lr_lambda(current_step: int, floor_ratio: float = floor_ratio) -> float:
            if num_training_steps <= 0:
                return floor_ratio

            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cosine

        lr_lambdas.append(lr_lambda)

    return LambdaLR(optimizer, lr_lambdas)


def parse_lr_milestones(spec: str) -> List[Tuple[int, float]]:
    """Parse `"20000:5e-6,50000:1e-6"` into ascending (step, lr) anchors."""
    milestones = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step_text, _, lr_text = chunk.partition(":")
        if not lr_text:
            raise ValueError(f"lr milestone must look like `step:lr`, got {chunk!r}")
        milestones.append((int(step_text), float(lr_text)))
    if not milestones:
        raise ValueError("lr milestones must contain at least one `step:lr` entry")
    steps = [step for step, _ in milestones]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise ValueError(f"lr milestones must have strictly increasing steps, got {steps}")
    if any(step <= 0 or lr <= 0 for step, lr in milestones):
        raise ValueError("lr milestone steps and learning rates must be positive")
    return milestones


def get_piecewise_cosine_schedule(
    optimizer,
    num_warmup_steps: int,
    milestones: List[Tuple[int, float]],
    start_step: int = 0,
):
    """Cosine-decay between successive (step, lr) anchors, then hold the final lr.

    One cosine cannot pass through two prescribed interior points, so a schedule
    like "5e-5 -> 5e-6 by step 20k -> 1e-6 by step 50k" needs one cosine segment
    per interval. Milestone steps are absolute global training steps; `start_step`
    shifts the scheduler's own counter onto that axis so a resumed run stays on
    the same curve.
    """
    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group["lr"])
        if base_lr <= 0:
            raise ValueError(f"`base_lr` must be positive, got {base_lr}.")
        # Keep the whole curve and shift the counter onto the absolute step axis.
        # Re-anchoring at `start_step` was the bug: the previous code anchored there
        # at `base_lr`, so a run resumed at step 7000 of a 5e-5 -> 5e-6@4k -> 5e-7@10k
        # schedule came back at 5e-5 instead of the 2.75e-6 it had reached -- 18x too
        # high, its action loss went 0.0050 -> 0.0083, and there was no room left to
        # anneal before max_steps. Re-anchoring at the *right* value still bends the
        # curve, because a fresh cosine segment starts flat while the original is
        # mid-descent; keeping the anchors makes a resumed run trace it exactly.
        anchors = [(0, base_lr), *milestones]
        if not [s for s, _ in milestones if s > start_step]:
            raise ValueError(f"no lr milestone lies beyond start_step={start_step}; got {milestones}")

        def lr_lambda(current_step: int, anchors: List = anchors, base_lr: float = base_lr) -> float:
            step = current_step + start_step
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            for (s0, lr0), (s1, lr1) in zip(anchors, anchors[1:]):
                if step <= s1:
                    progress = min(max(float(step - s0) / max(1, s1 - s0), 0.0), 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return (lr1 + (lr0 - lr1) * cosine) / base_lr
            return anchors[-1][1] / base_lr

        lr_lambdas.append(lr_lambda)

    return LambdaLR(optimizer, lr_lambdas)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        max_steps: int,
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        min_learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        warmup_ratio: float,
        resume_step: int = 0,
        lr_decay_end_step: Optional[int] = None,
        lr_milestones: Optional[str] = None,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.max_steps = max_steps
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.min_learning_rate = learning_rate, min_learning_rate
        self.weight_decay, self.max_grad_norm = weight_decay, max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.resume_step = resume_step
        if not 0 <= self.resume_step < self.max_steps:
            raise ValueError(
                f"resume_step must satisfy 0 <= resume_step < max_steps; got {self.resume_step} and {self.max_steps}."
            )
        self.lr_decay_end_step = self.max_steps if lr_decay_end_step is None else lr_decay_end_step
        self.lr_milestones = parse_lr_milestones(lr_milestones) if lr_milestones else None
        if not self.resume_step < self.lr_decay_end_step <= self.max_steps:
            raise ValueError(
                "lr_decay_end_step must satisfy resume_step < lr_decay_end_step <= max_steps; "
                f"got {self.resume_step}, {self.lr_decay_end_step}, and {self.max_steps}."
            )

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn
        self.cpu_memory_log_interval = 10

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Optional recovery-only guard. Normal training leaves this disabled and
        # therefore pays no additional collective-communication cost.
        self.action_loss_guard = None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @staticmethod
    def _read_cgroup_memory_bytes() -> tuple[Optional[int], Optional[int]]:
        """Return the current and maximum memory of the active cgroup when available."""
        candidates = (
            (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
            (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
        )
        for current_path, limit_path in candidates:
            if not current_path.exists():
                continue
            try:
                current_bytes = int(current_path.read_text().strip())
                raw_limit = limit_path.read_text().strip() if limit_path.exists() else "max"
                limit_bytes = None if raw_limit == "max" else int(raw_limit)
                return current_bytes, limit_bytes
            except (OSError, ValueError):
                continue
        return None, None

    def collect_cpu_memory_metrics(self) -> dict[str, float]:
        """Collect process RSS across all ranks and node/cgroup memory on rank zero."""
        try:
            local_rss_bytes = float(psutil.Process().memory_info().rss)
        except psutil.Error:
            local_rss_bytes = 0.0

        rss_sum_bytes = local_rss_bytes
        rss_max_bytes = local_rss_bytes
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            reduce_device = torch.device("cuda", self.device_id) if torch.cuda.is_available() else torch.device("cpu")
            rss = torch.tensor([local_rss_bytes], dtype=torch.float64, device=reduce_device)
            rss_sum = rss.clone()
            rss_max = rss.clone()
            dist.all_reduce(rss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(rss_max, op=dist.ReduceOp.MAX)
            rss_sum_bytes = float(rss_sum.item())
            rss_max_bytes = float(rss_max.item())
            world_size = dist.get_world_size()

        if not overwatch.is_rank_zero():
            return {}

        gib = float(1024 ** 3)
        metrics = {
            "System/CPU RSS Max Rank (GiB)": rss_max_bytes / gib,
            "System/CPU RSS Mean Rank (GiB)": rss_sum_bytes / world_size / gib,
            "System/CPU RSS Sum Ranks (GiB)": rss_sum_bytes / gib,
        }
        try:
            virtual_memory = psutil.virtual_memory()
            metrics["System/Host Memory Used (GiB)"] = virtual_memory.used / gib
            metrics["System/Host Memory Available (GiB)"] = virtual_memory.available / gib
            metrics["System/Host Memory Percent"] = float(virtual_memory.percent)
        except psutil.Error:
            pass

        cgroup_current, cgroup_limit = self._read_cgroup_memory_bytes()
        if cgroup_current is not None:
            metrics["System/Cgroup Memory Current (GiB)"] = cgroup_current / gib
        if cgroup_limit is not None and cgroup_limit > 0:
            metrics["System/Cgroup Memory Limit (GiB)"] = cgroup_limit / gib
            metrics["System/Cgroup Memory Percent"] = 100.0 * cgroup_current / cgroup_limit
        return metrics

    def _reduce_max_performance(self, values: dict[str, float], device: torch.device) -> dict[str, float]:
        """Max-reduce sampled timings and memory so the slowest rank is reported."""
        if not (dist.is_available() and dist.is_initialized()):
            return values
        keys = tuple(values)
        tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return {key: float(value) for key, value in zip(keys, tensor.tolist(), strict=True)}

    # Learned position-like embeddings and zero-init gates carry identity, not
    # scale, so decaying them shrinks the very signal they exist to provide. The
    # horizon embedding is the clearest case: it is the only thing distinguishing
    # the five future horizons from one another, and under decay it went from
    # std 0.020 at init to 0.0197 after 5000 steps -- it never grew.
    UNDECAYED_PARAMETER_NAMES = (
        "horizon_embeddings",
        "view_embeddings",
        "view_gate",
        "position_embedding",
    )

    def build_optimizer_groups(self, named_parameters):
        decay, no_decay = [], []
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            leaf = name.rsplit(".", 1)[-1]
            skip_decay = (
                param.ndim <= 1
                or name.endswith(".bias")
                or leaf in self.UNDECAYED_PARAMETER_NAMES
            )
            (no_decay if skip_decay else decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.weight_decay, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "no-decay"},
        ]
        return [group for group in groups if group["params"]]

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> torch.Tensor: ...

    @staticmethod
    def _distributed_any(flag: torch.Tensor) -> bool:
        """Return the same boolean on every rank using a single scalar reduction."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _raise_if_nonfinite_losses(self, output: dict, device: torch.device, global_step: int) -> None:
        """Coordinate loss-finiteness checks so all ranks fail together."""
        checked = {
            key: output[key]
            for key in ("loss", "loss_action", "loss_world")
            if key in output and isinstance(output[key], torch.Tensor)
        }
        local_bad = any(not torch.isfinite(value.detach()).all() for value in checked.values())
        bad_flag = torch.tensor(int(local_bad), dtype=torch.int32, device=device)
        if self._distributed_any(bad_flag):
            local_names = [name for name, value in checked.items() if not torch.isfinite(value.detach()).all()]
            raise FloatingPointError(
                f"Non-finite training loss before backward at global step {global_step + 1}; "
                f"local_nonfinite={local_names}"
            )

    def _cuda_mem_snapshot(self, label: str) -> dict:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        return {
            "label": label,
            "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }

    def _log_cuda_mem_snapshot(self, label: str, step: int) -> None:
        if not overwatch.is_rank_zero() or not getattr(self, "debug_memory_stats", False):
            return
        snap = self._cuda_mem_snapshot(label)
        if not snap:
            return
        overwatch.info(
            f"[Mem][step={step:06d}] {label}: "
            f"alloc={snap['allocated_gb']:.2f}G reserved={snap['reserved_gb']:.2f}G max={snap['max_allocated_gb']:.2f}G"
        )

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        co_train_streams: Optional[List[Tuple[IterableDataset, object]]] = None,
        co_train_every: int = 0,
        co_train_ratio: Optional[float] = None,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`.

        `co_train_streams` holds one (dataset, collator) pair per extra robot, and
        `co_train_ratio` is the fraction of micro-batches drawn from them (falling
        back to `1 / co_train_every`). A ratio is used rather than a period because
        the deployment robot's corpus is the small one here: holding it to a minority
        of the batches is the point, and "every Nth batch" cannot express a share
        above one half. Streams cycle so each robot gets an equal part of that
        fraction, and everything accumulates into the same optimizer step, so one
        update carries gradients from several embodiments. Each stream carries its own
        collator because the collator pads actions and proprio to a per-robot width.
        Leaving both off keeps the original single-stream loop untouched.
        """
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"

        def build_dataloader(dataset: IterableDataset, collate_fn=None) -> DataLoader:
            workers = getattr(dataset, "dataloader_num_workers", 0)
            kwargs = {
                "dataset": dataset,
                "batch_size": self.per_device_batch_size,
                "sampler": None,
                "collate_fn": collator if collate_fn is None else collate_fn,
                "num_workers": workers,
                "pin_memory": getattr(dataset, "dataloader_pin_memory", False),
            }
            if workers > 0:
                kwargs.update(
                    prefetch_factor=getattr(dataset, "dataloader_prefetch_factor", 2),
                    persistent_workers=True,
                )
            return DataLoader(**kwargs)

        dataloader_num_workers = getattr(vla_dataset, "dataloader_num_workers", 0)
        dataloader_kwargs = {
            "dataset": vla_dataset,
            "batch_size": self.per_device_batch_size,
            "sampler": None,
            "collate_fn": collator,
            "num_workers": dataloader_num_workers,
            "worker_init_fn": self.worker_init_fn,
            "pin_memory": getattr(vla_dataset, "dataloader_pin_memory", False),
        }
        if dataloader_num_workers > 0:
            dataloader_kwargs.update(
                prefetch_factor=getattr(vla_dataset, "dataloader_prefetch_factor", 2),
                persistent_workers=True,
            )
        dataloader = DataLoader(**dataloader_kwargs)
        overwatch.info(
            "VLA DataLoader: num_workers=%d prefetch_factor=%s persistent_workers=%s pin_memory=%s",
            dataloader_num_workers,
            dataloader_kwargs.get("prefetch_factor"),
            dataloader_kwargs.get("persistent_workers", False),
            dataloader_kwargs["pin_memory"],
        )

        action_loss_sum = None
        action_loss_count = 0

        def process_batch(
            batch,
            *,
            grad_step_ready: bool,
            accum_divisor: int,
            epoch_value: int,
            phase_timer: CudaPhaseTimer,
            step_start: float,
            device: torch.device,
        ) -> bool:
            nonlocal action_loss_sum, action_loss_count
            should_log_memory = (
                getattr(self, "debug_memory_stats", False)
                and getattr(self, "debug_memory_stats_interval", 0) > 0
                and (metrics.global_step % self.debug_memory_stats_interval) == 0
            )
            if getattr(self, "debug_batch_shapes", False) and not getattr(self, "_printed_batch_shapes", False):
                if overwatch.is_rank_zero():
                    shape_lines = []
                    for key in (
                        "pixel_values",
                        "pair_pixel_values",
                        "current_frame_pairs",
                        "future_frames",
                        "actions",
                        "proprio",
                        "input_ids",
                        "attention_mask",
                    ):
                        value = batch.get(key)
                        if isinstance(value, torch.Tensor):
                            shape_lines.append(f"{key}={tuple(value.shape)} dtype={value.dtype}")
                        else:
                            shape_lines.append(f"{key}={type(value).__name__}")
                    dataset_names = batch.get("dataset_names")
                    if dataset_names is not None:
                        shape_lines.append(f"dataset_names[0]={dataset_names[0]!r}")
                    overwatch.info("First training batch shapes: %s", " | ".join(shape_lines))
                self._printed_batch_shapes = True

            phase_timer.start("forward")
            with torch.autocast(
                "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
            ):
                output = self.vlm(**build_vla_forward_kwargs(batch))
                loss = output["loss"]
            phase_timer.stop("forward")

            if self.action_loss_guard is not None:
                self._raise_if_nonfinite_losses(output, device, metrics.global_step)
                if "loss_action" not in output:
                    raise RuntimeError("Action-loss guard is enabled, but the model did not return loss_action")
                action_value = output["loss_action"].detach().float().mean()
                action_loss_sum = action_value if action_loss_sum is None else action_loss_sum + action_value
                action_loss_count += 1

            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_forward", metrics.global_step)
                if isinstance(output, dict) and "memory_stats" in output:
                    for entry in output["memory_stats"]:
                        overwatch.info(
                            f"[Mem][step={metrics.global_step:06d}] {entry['label']}: "
                            f"alloc={entry['allocated_gb']:.2f}G reserved={entry['reserved_gb']:.2f}G "
                            f"max={entry['max_allocated_gb']:.2f}G"
                        )

            metrics.commit(loss=loss)
            phase_timer.start("backward")
            (loss / accum_divisor).backward()
            phase_timer.stop("backward")
            if "loss_action" in output:
                metrics.commit(loss_action=output["loss_action"])
            if "loss_visual_token_cosine" in output:
                metrics.commit(loss_visual_token_cosine=output["loss_visual_token_cosine"])
            if "loss_world" in output:
                metrics.commit(loss_world=output["loss_world"])
            if "loss_latent_ar" in output:
                metrics.commit(loss_latent_ar=output["loss_latent_ar"])
            # The two numbers that say whether a falling latent loss is real.
            # `target_motion_rms` is how much the encoder still separates a frame
            # one second ahead from the frame now; `residual_cosine` is how well
            # the head predicts the direction of that change, and being a cosine
            # it cannot be improved by shrinking the target.
            # Driven by DIAGNOSTIC_KEYS rather than a list repeated here: two
            # hardcoded tuples had already drifted from it, which is why
            # `loss_temporal_hinge` was computed every step and never logged.
            for diagnostic in metrics.DIAGNOSTIC_KEYS:
                if output.get(diagnostic) is not None:
                    metrics.commit(**{diagnostic: output[diagnostic]})
            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_backward", metrics.global_step)

            metrics.commit(update_step_time=grad_step_ready)

            if not grad_step_ready:
                return False

            if self.action_loss_guard is not None:
                if action_loss_sum is None or action_loss_count == 0:
                    raise RuntimeError("Action-loss guard reached an optimizer boundary without observations")
                action_loss_mean = action_loss_sum / action_loss_count
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(action_loss_mean, op=dist.ReduceOp.SUM)
                    action_loss_mean /= dist.get_world_size()
                observed_action_loss = float(action_loss_mean.item())
                action_loss_sum = None
                action_loss_count = 0
                if self.action_loss_guard.observe(observed_action_loss):
                    raise RuntimeError(
                        "Action-loss stability guard tripped before optimizer update at "
                        f"global step {metrics.global_step + 1}: observed={observed_action_loss:.6f}, "
                        f"moving_average={self.action_loss_guard.moving_average}, "
                        f"threshold={self.action_loss_guard.threshold:.6f}"
                    )

            phase_timer.start("grad_clip")
            grad_norm = self.clip_grad_norm()
            phase_timer.stop("grad_clip")
            grad_norm_tensor = torch.as_tensor(grad_norm, device=device).detach()
            if self.action_loss_guard is not None:
                bad_grad = torch.tensor(
                    int(not torch.isfinite(grad_norm_tensor).all()), dtype=torch.int32, device=device
                )
                if self._distributed_any(bad_grad):
                    raise FloatingPointError(
                        f"Non-finite gradient norm before optimizer update at global step {metrics.global_step + 1}"
                    )
            metrics.commit(grad_norm=grad_norm_tensor)
            phase_timer.start("optimizer")
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            # Momentum-update the world-model target encoder, if the online encoder
            # is being fine-tuned. A no-op when the encoder stays frozen.
            update_target = getattr(self.vlm, "update_vision_target_encoder", None)
            if update_target is not None:
                update_target()
            phase_timer.stop("optimizer")
            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_optimizer_step", metrics.global_step + 1)

            metrics.commit(global_step=metrics.global_step + 1, epoch=epoch_value, lr=self.lr_scheduler.get_last_lr()[0])
            should_log_cpu_memory = self.cpu_memory_log_interval > 0 and (
                metrics.global_step == 1 or metrics.global_step % self.cpu_memory_log_interval == 0
            )
            if should_log_cpu_memory:
                cpu_memory_metrics = self.collect_cpu_memory_metrics()
                if overwatch.is_rank_zero():
                    metrics.set_system_metrics(**cpu_memory_metrics)
            if phase_timer.enabled:
                sampled = phase_timer.finish()
                sampled["step"] = time.perf_counter() - step_start
                sampled["allocated_gib"] = torch.cuda.memory_allocated(device) / (1024**3)
                sampled["reserved_gib"] = torch.cuda.memory_reserved(device) / (1024**3)
                sampled["peak_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                sampled["allocator_retries"] = float(
                    torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
                )
                sampled = self._reduce_max_performance(sampled, device)
                performance_metrics = {
                    "Performance/Dataloader Wait (s)": sampled["dataloader"],
                    "Performance/H2D Wait (s)": sampled["h2d"],
                    "Performance/Forward (s)": sampled["forward"],
                    "Performance/Backward (s)": sampled["backward"],
                    "Performance/Grad Clip (s)": sampled["grad_clip"],
                    "Performance/Optimizer (s)": sampled["optimizer"],
                    "Performance/Step Time (s)": sampled["step"],
                    "Performance/Samples/s": self.global_batch_size / sampled["step"],
                    "Performance/Allocated Memory (GiB)": sampled["allocated_gib"],
                    "Performance/Reserved Memory (GiB)": sampled["reserved_gib"],
                    "Performance/Peak Memory (GiB)": sampled["peak_gib"],
                    "Performance/Allocator Retries": sampled["allocator_retries"],
                }
                flop_config = getattr(self, "performance_flop_config", None)
                if flop_config is not None:
                    flop_config = dict(flop_config)
                    # Joint Qwen length = padded text/action placeholders plus
                    # action/current vision tokens and five future tubelets.
                    flop_config["qwen_sequence_length"] = int(batch["input_ids"].shape[1]) + 12_096
                    global_flops = estimate_robotwin_training_flops(flop_config, self.global_batch_size)
                    performance_metrics["Performance/EstimatedMFU"] = estimated_mfu(
                        global_flops,
                        sampled["step"],
                        overwatch.world_size(),
                    )
                metrics.set_performance_metrics(**performance_metrics)
            status = metrics.push()

            step_save_due = (
                save_interval is not None
                and save_interval > 0
                and (metrics.global_step % save_interval) == 0
            )
            if (terminate := metrics.global_step >= self.max_steps) or step_save_due:
                # Keep every rank at the same checkpoint boundary before FSDP starts
                # reconstructing state, then wait for rank zero's atomic commit.
                dist.barrier(device_ids=[torch.cuda.current_device()])
                self.save_checkpoint(
                    metrics.run_dir, metrics.global_step, epoch_value, loss.item(), only_trainable=False
                )
                dist.barrier(device_ids=[torch.cuda.current_device()])
                if terminate:
                    return True

            progress.update()
            progress.set_description(status)
            return False

        status = metrics.get_status()
        with tqdm(
            total=self.max_steps,
            initial=metrics.global_step,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()
            self.optimizer.zero_grad()

            metrics.commit(global_step=self.resume_step)

            global_dataset_length = getattr(vla_dataset, "global_dataset_length", len(vla_dataset))
            epoch_denominator = max(1, math.ceil(global_dataset_length / self.global_batch_size))
            device = torch.device("cuda", self.device_id)
            batch_iterator = iter(CUDAPrefetcher(dataloader, device))
            # A deterministic fractional schedule: every rank derives the same
            # sequence from the micro-batch index, so no rank can disagree about
            # which stream this step draws from and stall the collective.
            effective_ratio = (
                float(co_train_ratio)
                if co_train_ratio is not None
                else (1.0 / co_train_every if co_train_every > 0 else 0.0)
            )
            if not 0.0 <= effective_ratio < 1.0:
                raise ValueError(f"co-training ratio must be in [0, 1), got {effective_ratio}")

            co_train_iterators = []
            if co_train_streams and effective_ratio > 0:
                co_train_iterators = [
                    iter(CUDAPrefetcher(build_dataloader(dataset, stream_collator), device))
                    for dataset, stream_collator in co_train_streams
                ]
                overwatch.info(
                    "Co-training: %.0f%% of micro-batches across %d extra stream(s); "
                    "the deployment robot keeps %.0f%%",
                    100 * effective_ratio,
                    len(co_train_iterators),
                    100 * (1 - effective_ratio),
                )
            train_idx = 0

            def make_phase_timer() -> CudaPhaseTimer:
                return CudaPhaseTimer(
                    enabled=should_sample_performance(
                        metrics.global_step + 1,
                        int(getattr(self, "performance_timing_interval", 0)),
                    ),
                    phases=("dataloader", "h2d", "forward", "backward", "grad_clip", "optimizer"),
                    reset_peak_memory=(lambda: torch.cuda.reset_peak_memory_stats(device)),
                )

            stream_batch_counts: dict = {}
            timing_window = OptimizerStepTimingWindow(make_phase_timer)
            while True:
                phase_timer, step_start = timing_window.acquire()
                data_wait_start = time.perf_counter()
                phase_timer.start("h2d")
                drawn_before = int(train_idx * effective_ratio)
                use_co_train = bool(co_train_iterators) and (
                    int((train_idx + 1) * effective_ratio) > drawn_before
                )
                source = batch_iterator
                stream_index = None
                if use_co_train:
                    # Round-robin, so a large corpus does not crowd out a small one.
                    stream_index = drawn_before % len(co_train_iterators)
                    source = co_train_iterators[stream_index]
                try:
                    batch = next(source)
                except StopIteration:
                    break
                # Report the mix once per stream: a misrouted or starved co-training
                # stream is otherwise invisible until the run is over.
                stream_batch_counts[stream_index] = stream_batch_counts.get(stream_index, 0) + 1
                if stream_batch_counts[stream_index] == 1:
                    names = batch.get("dataset_names")
                    overwatch.info(
                        "First micro-batch from %s stream: `%s`",
                        "deployment" if stream_index is None else f"co-training #{stream_index}",
                        names[0] if names else "<unnamed>",
                    )
                phase_timer.stop("h2d")
                phase_timer.add_cpu_time("dataloader", time.perf_counter() - data_wait_start)
                grad_step_ready = ((train_idx + 1) % self.grad_accumulation_steps) == 0
                epoch_value = (metrics.global_step + 1) // epoch_denominator
                terminate = process_batch(
                    batch,
                    grad_step_ready=grad_step_ready,
                    accum_divisor=self.grad_accumulation_steps,
                    epoch_value=epoch_value,
                    phase_timer=phase_timer,
                    step_start=step_start,
                    device=device,
                )
                if grad_step_ready:
                    timing_window.reset()
                if terminate:
                    return
                train_idx += 1

        raise RuntimeError("VLA dataloader ended before max_steps was reached.")
