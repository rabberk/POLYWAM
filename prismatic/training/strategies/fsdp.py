"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

import math
import os
import random
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.checkpoint_migration import (
    NEW_MODULE_PREFIXES,
    migrate_state_dict,
    seed_target_encoder,
)
from prismatic.training.recovery import (
    reset_optimizer_schedule_for_recovery,
    validate_resume_metadata,
)
from prismatic.training.strategies.base_strategy import (
    TrainingStrategy,
    get_cosine_schedule_with_warmup_and_group_min_lrs,
    get_piecewise_cosine_schedule,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def atomic_save_checkpoint(payload: dict, checkpoint_path: Path) -> None:
    """Atomically publish one checkpoint and repoint ``latest`` without copying it."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    latest_path = checkpoint_path.parent / "latest-checkpoint.pt"
    temporary_link = checkpoint_path.parent / ".latest-checkpoint.pt.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(checkpoint_path.name)
    os.replace(temporary_link, latest_path)


def validate_attention_checkpoint_policy(attention_implementation: str | None, enabled: bool) -> None:
    """Reject checkpoint recomputation that bypasses the compiled FA4 kernel."""
    if attention_implementation == "b300_fa4" and enabled:
        raise RuntimeError(
            "B300 FA4 is incompatible with FSDP activation checkpointing: "
            "checkpoint recomputation disables Dynamo and causes dense attention fallback"
        )


class FSDPStrategy(TrainingStrategy):
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
        save_optimizer_state: bool = True,
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            resume_step=resume_step,
            lr_decay_end_step=lr_decay_end_step,
            lr_milestones=lr_milestones,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # FSDP-Specific Parameters
        #   =>> Use HYBRID_SHARD only for multi-node; single-node falls back to standard FULL_SHARD/SHARD_GRAD_OP
        local_world_size = torch.cuda.device_count()
        is_single_node = dist.is_initialized() and dist.get_world_size() <= local_world_size

        default_strategy = "FULL_SHARD" if is_single_node else "HYBRID_SHARD"
        # FULL_SHARD all-gathers every layer's parameters on both the forward and
        # the backward. That is the right trade when the parameters dominate
        # memory, but a LoRA run here trains 17.6M of a 1.5B model on 80GB cards,
        # so the gathers buy headroom that is not needed and cost a communication
        # gap in every layer. NO_SHARD replicates instead: the only collective
        # left is the gradient all-reduce over what is actually trainable.
        requested = os.getenv("VLA_FSDP_SHARDING_STRATEGY", default_strategy).upper()
        try:
            self.fsdp_sharding_strategy = ShardingStrategy[requested]
        except KeyError as error:
            valid = ", ".join(strategy.name for strategy in ShardingStrategy)
            raise ValueError(f"VLA_FSDP_SHARDING_STRATEGY={requested!r} is not one of: {valid}") from error
        overwatch.info(f"FSDP sharding strategy: {self.fsdp_sharding_strategy.name}")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.fsdp_optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.start_step = 0
        self.start_epoch = 0
        self.checkpoint_metadata = None
        self.required_checkpoint_metadata = None
        self.save_optimizer_state = bool(save_optimizer_state)

    @staticmethod
    def _capture_rng_state() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"])

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"

        gathered_rng_states = None
        if self.save_optimizer_state:
            local_rng_state = self._capture_rng_state()
            gathered_rng_states = [None] * dist.get_world_size() if overwatch.is_rank_zero() else None
            dist.gather_object(local_rng_state, gathered_rng_states, dst=0)

        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(
            self.vlm,
            self.fsdp_state_dict_type,
            self.fsdp_save_policy,
            self.fsdp_optim_save_policy,
        ):
            full_vlm_state_dict = self.vlm.state_dict()
            # Model-only recovery is the reliable default for the RoboTwin AR launcher:
            # its resume path intentionally rebuilds the optimizer/scheduler. Avoiding
            # this collective removes the largest checkpoint-boundary all-reduce.
            full_optimizer_state = (
                FSDP.optim_state_dict(self.vlm, self.optimizer) if self.save_optimizer_state else None
            )
            model_state_dicts = {
                mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
            }

            # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            for key, param in full_vlm_state_dict.items():
                for mkey in model_state_dicts:
                    if key.startswith(mprefix := f"{mkey}."):
                        model_state_dicts[mkey][key.removeprefix(mprefix)] = param

            # Save on rank zero *only*
            save_status = [None]
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / "checkpoints"
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
                else:
                    checkpoint_path = (
                        checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                    )

                payload = {
                        "model": model_state_dicts,
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "metadata": self.checkpoint_metadata,
                }
                if self.save_optimizer_state:
                    payload.update(
                        optimizer=full_optimizer_state,
                        scheduler=self.lr_scheduler.state_dict(),
                        rng_states=gathered_rng_states,
                    )
                try:
                    atomic_save_checkpoint(payload, checkpoint_path)
                    overwatch.info(
                        "Checkpoint committed atomically: %s (optimizer_state=%s)",
                        checkpoint_path,
                        self.save_optimizer_state,
                    )
                except Exception as exc:  # synchronize rank-zero I/O failures to every rank
                    save_status[0] = f"{type(exc).__name__}: {exc}"

            dist.broadcast_object_list(save_status, src=0, device=torch.device("cuda", torch.cuda.current_device()))
            if save_status[0] is not None:
                raise RuntimeError(f"rank-zero checkpoint write failed: {save_status[0]}")

    def load_training_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        reset_scheduler_on_resume: bool = False,
        allow_episode_selection_override: bool = False,
        allow_global_batch_size_override: bool = False,
    ) -> None:
        """Restore post-training state after ``run_setup`` has wrapped the model."""
        assert isinstance(self.vlm, FSDP), "load_training_checkpoint must run after run_setup"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        required = {"model", "optimizer", "scheduler", "step", "epoch"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(
                f"{checkpoint_path} is not a resumable RoboTwin checkpoint; missing {missing}. "
                "Use it only as an initialization checkpoint."
            )
        if self.required_checkpoint_metadata is not None:
            actual_metadata = checkpoint.get("metadata")
            validate_resume_metadata(
                self.required_checkpoint_metadata,
                actual_metadata,
                allow_schedule_overrides=reset_scheduler_on_resume,
                allow_episode_selection_override=allow_episode_selection_override,
                allow_global_batch_size_override=allow_global_batch_size_override,
            )
        rng_states = checkpoint.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != dist.get_world_size():
            raise ValueError("checkpoint does not contain one RNG state per current distributed rank")

        nested_model = checkpoint["model"]
        flat_model = OrderedDict(
            (f"{module_name}.{name}", value)
            for module_name, component in nested_model.items()
            for name, value in component.items()
        )
        # A checkpoint written before the target encoder was persisted has no entry
        # for it; it is seeded from the online encoder below rather than failing.
        allowed_missing_prefixes = ("vision_backbone.", "projector.", "vision_target_encoder.")
        if getattr(self, "migrate_checkpoint", False):
            flat_model, renamed = migrate_state_dict(flat_model)
            # Seeding needs the target's key names, which only a first load reports.
            with FSDP.state_dict_type(
                self.vlm, self.fsdp_state_dict_type, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            ):
                probe = self.vlm.load_state_dict(flat_model, strict=False)
            seeded = seed_target_encoder(flat_model, probe.missing_keys)
            allowed_missing_prefixes = allowed_missing_prefixes + NEW_MODULE_PREFIXES
            overwatch.info(
                "Migrating checkpoint: %d prefix-encoder tensors renamed, %d EMA target tensors "
                "seeded from the online encoder; new cross-embodiment modules start fresh",
                len(renamed),
                seeded,
            )

        if any(key.startswith("vision_backbone.") for key in flat_model) and not any(
            key.startswith("vision_target_encoder.") for key in flat_model
        ):
            with FSDP.state_dict_type(
                self.vlm, self.fsdp_state_dict_type, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            ):
                probe = self.vlm.load_state_dict(flat_model, strict=False)
            seeded = seed_target_encoder(flat_model, probe.missing_keys)
            if seeded:
                overwatch.info(
                    "Checkpoint predates the persisted EMA target; seeded %d tensors from the "
                    "online encoder (its momentum lag rebuilds over the next ~%d steps)",
                    seeded,
                    int(1.0 / max(1e-6, 1.0 - getattr(self.vlm, "vision_target_momentum", 0.999))),
                )

        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, load_policy):
            incompatible = self.vlm.load_state_dict(flat_model, strict=False)
        allowed_missing_exact = {"latent_mean", "latent_std"}
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if key not in allowed_missing_exact and not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"checkpoint model mismatch: {incompatible}")

        if getattr(self, "migrate_checkpoint", False):
            # The optimizer state is keyed to the old parameter set, so reusing it
            # would pair moments with the wrong tensors. Start the optimizer fresh
            # and let the schedule warm back up rather than corrupt it silently.
            overwatch.info("Migrated checkpoint: starting from a fresh optimizer state")
            self.start_step = int(checkpoint["step"])
            self.start_epoch = int(checkpoint["epoch"])
            return

        optimizer_state = FSDP.optim_state_dict_to_load(
            self.vlm,
            self.optimizer,
            checkpoint["optimizer"],
        )
        self.optimizer.load_state_dict(optimizer_state)
        self.start_step = int(checkpoint["step"])
        self.start_epoch = int(checkpoint["epoch"])
        if reset_scheduler_on_resume:
            self.lr_scheduler = reset_optimizer_schedule_for_recovery(
                self.optimizer,
                start_lr=self.learning_rate,
                min_lr=self.min_learning_rate,
                recovery_steps=self.max_steps - self.start_step,
            )
        else:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self._restore_rng_state(rng_states[dist.get_rank()])
        overwatch.info(
            "Resumed RoboTwin model/optimizer/%s/RNG state from %s at step=%d epoch=%d",
            "new recovery scheduler" if reset_scheduler_on_resume else "checkpoint scheduler",
            checkpoint_path,
            self.start_step,
            self.start_epoch,
        )
        del checkpoint, flat_model, optimizer_state

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        llm = getattr(getattr(self.vlm, "llm_backbone", None), "llm", None)
        attention_implementation = getattr(getattr(llm, "config", None), "_attn_implementation", None)
        validate_attention_checkpoint_policy(attention_implementation, self.enable_gradient_checkpointing)

        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
            #   => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype
            )

            # Half precision here is a memory saving for a backbone that never produces
            # gradients. Once it is being fine-tuned its gradients would be bf16 among
            # every other module's fp32, and FSDP's gradient clipping rejects a mixed
            # set outright -- so a trainable backbone stays fp32 and takes its bf16
            # compute from the mixed-precision policy like everything else.
            if getattr(self.vlm, "vision_backbone_requires_grad", False):
                overwatch.info("Vision Backbone is trainable: keeping parameters in full precision")
            else:
                overwatch.info("Casting frozen Vision Backbone to half precision")
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)

        else:
            # If we're not using mixed precision, everything is in default full precision!
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        # The EMA target encoder is sharded like the online encoder it tracks. Both
        # must be laid out the same way: the momentum update pairs them parameter for
        # parameter, and under FSDP each rank holds only its own shard, so an
        # unsharded copy would present full-sized tensors against sharded ones.
        self.vlm = FSDP(
            self.vlm,
            auto_wrap_policy=vlm_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier(device_ids=[torch.cuda.current_device()])

        # Optimizer should only operate on parameters that are unfrozen/trainable.
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        num_training_steps = self.lr_decay_end_step - self.resume_step

        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        groups = self.build_optimizer_groups(self.vlm.named_parameters())
        optimizer_groups = [
            {
                "params": group["params"],
                "weight_decay": group["weight_decay"],
                "lr": group["base_lr"],
                "base_lr": group["base_lr"],
                "min_lr": group["min_lr"],
                "name": group["name"],
            }
            for group in groups
        ]

        self.optimizer = AdamW(optimizer_groups, lr=self.learning_rate)
        if self.lr_milestones is not None:
            # Piecewise cosine: one segment per (step, lr) anchor. Overrides the
            # single min_lr/decay_end_step cosine, which cannot hit two prescribed
            # interior points.
            self.lr_scheduler = get_piecewise_cosine_schedule(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                milestones=self.lr_milestones,
                start_step=self.resume_step,
            )
        elif any(group["min_lr"] > 0 for group in optimizer_groups):
            self.lr_scheduler = get_cosine_schedule_with_warmup_and_group_min_lrs(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps, num_training_steps
            )
        if num_warmup_steps > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> Min Cosine LR = {self.min_learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            "         |-> LR Scheduler Type = linear warmup + cosine decay\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Global Step = {self.max_steps}\n"
            f"         |-> Resume Step = {self.resume_step} (fresh optimizer; {self.max_steps - self.resume_step} updates remain)\n"
            f"         |-> LR Decay End Step = {self.lr_decay_end_step} ({num_training_steps} decay updates)\n"
        )
        for group in self.optimizer.param_groups:
            overwatch.info(
                "         |-> Optimizer Group `%s`: base_lr=%s min_lr=%s weight_decay=%s params=%d",
                group.get("name", "unnamed"),
                group.get("base_lr", group["lr"]),
                group.get("min_lr", 0.0),
                group["weight_decay"],
                len(group["params"]),
            )

    def clip_grad_norm(self) -> torch.Tensor:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        return self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)
