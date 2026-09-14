"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/path/to/local_resource"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 --module prismatic.training.train
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K --module prismatic.training.train
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import numpy as np
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_fsdp_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.robotwin_paper import RoboTwinPaperDataset, RoboTwinPaperDatasetConfig

from peft import LoraConfig, get_peft_model
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_dataset_statistics(dataset_statistics, run_dir: Path) -> None:
    path = run_dir / "dataset_statistics.json"
    path.write_text(json.dumps(_jsonable(dataset_statistics), indent=2))
    overwatch.info("Saved dataset statistics file at path %s", path)


def _normalize_lora_target_modules(target_modules):
    if isinstance(target_modules, tuple):
        return list(target_modules)
    if isinstance(target_modules, list):
        return target_modules
    if isinstance(target_modules, str):
        normalized = target_modules.strip()
        if normalized == "all-linear":
            return normalized
        if "," in normalized:
            return [module.strip() for module in normalized.split(",") if module.strip()]
        return normalized
    return target_modules


def apply_lora_to_vlm(vlm, vla_cfg: VLAConfig, unfreeze_llm: bool = False) -> None:
    if unfreeze_llm:
        # Full-parameter language training wants the bare backbone: wrapping it in
        # adapters first would leave the base weights frozen inside PEFT's wrapper.
        overwatch.info("Training Qwen with full parameters; skipping LoRA wrap.")
        return
    if hasattr(vlm.llm_backbone.llm, "peft_config"):
        overwatch.info("LLM already wrapped with LoRA; skipping re-wrap.")
        return

    lora_config = LoraConfig(
        r=vla_cfg.lora_rank,
        lora_alpha=vla_cfg.lora_alpha,
        target_modules=_normalize_lora_target_modules(vla_cfg.lora_target_modules),
        lora_dropout=vla_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )
    vlm.llm_backbone.llm = get_peft_model(vlm.llm_backbone.llm, lora_config)
    vlm.llm_backbone.llm.print_trainable_parameters()


def build_vla_from_base_vlm(
    base_vlm_id_or_path: Union[str, Path],
    cfg: "TrainConfig",
    hf_token: Optional[str],
):
    """
    Load the released base VLM checkpoint and attach the fixed JEPA-WAM heads.
    """
    from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm

    base_vlm_path = Path(base_vlm_id_or_path)
    if base_vlm_path.is_dir():
        run_dir = base_vlm_path
        checkpoint_dir = run_dir / "checkpoints"
        latest_checkpoint = checkpoint_dir / "latest-checkpoint.pt"
        if latest_checkpoint.exists():
            checkpoint_path = latest_checkpoint
        else:
            checkpoint_candidates = sorted(checkpoint_dir.glob("step-*.pt"))
            if not checkpoint_candidates:
                raise ValueError(f"Could not find a base VLM checkpoint under `{checkpoint_dir}`")
            checkpoint_path = checkpoint_candidates[-1]
    elif base_vlm_path.is_file():
        checkpoint_path = base_vlm_path
        run_dir = checkpoint_path.parent.parent
    else:
        raise ValueError(
            "JEPA-VLA training expects `vla.base_vlm` to point to either a base VLM run directory "
            "with `config.json` and `checkpoints/latest-checkpoint.pt`, or directly to a checkpoint `.pt` file."
        )

    with open(run_dir / "config.json", "r") as f:
        model_cfg = json.load(f)["model"]

    model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if "llm_backbone" not in model_state_dict or "projector" not in model_state_dict:
        raise ValueError(
            f"Base VLM checkpoint `{checkpoint_path}` must contain `llm_backbone` and `projector` weights."
        )

    vision_checkpoint_path = cfg.vla.vjepa_checkpoint_path or model_cfg.get("vision_checkpoint_path")
    llm_checkpoint_path = str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else model_cfg.get("llm_local_path")

    cfg_cross_embodiments = {
        name: spec for name, spec, _, _ in parse_co_train_specs(getattr(cfg, "co_train_specs", None))
    }

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        checkpoint_path=vision_checkpoint_path,
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=False,
        custom_hf_path=llm_checkpoint_path,
        use_flash_attention_2=cfg.vla.attention_backend == "flash_attention_2",
    )
    vlm = get_vlm(
        model_cfg["model_id"],
        model_cfg["arch_specifier"],
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        d_action=cfg.vla.d_action,
        d_proprio=cfg.vla.d_proprio,
        action_horizon=cfg.vla.action_horizon,
        fm_hidden_size=cfg.vla.fm_hidden_size,
        fm_num_layers=cfg.vla.fm_num_layers,
        fm_num_inference_timesteps=cfg.vla.fm_num_inference_timesteps,
        fm_num_timestep_buckets=cfg.vla.fm_num_timestep_buckets,
        fm_noise_beta_alpha=cfg.vla.fm_noise_beta_alpha,
        fm_noise_beta_beta=cfg.vla.fm_noise_beta_beta,
        fm_noise_s=cfg.vla.fm_noise_s,
        fm_num_target_vision_tokens=cfg.vla.fm_num_target_vision_tokens,
        fm_add_pos_embed=cfg.vla.fm_add_pos_embed,
        fm_max_seq_len=cfg.vla.fm_max_seq_len,
        fm_state_dropout=cfg.vla.fm_state_dropout,
        action_prediction_type=cfg.vla.action_prediction_type,
        flow_gr00t_placeholder_tokens=cfg.vla.flow_gr00t_placeholder_tokens,
        lambda_visual_token_cosine=cfg.vla.lambda_visual_token_cosine,
        enable_official_latent_head=cfg.vla.enable_official_latent_head,
        lambda_official_latent=cfg.vla.lambda_official_latent,
        enable_five_tubelet_ar=cfg.vla.enable_five_tubelet_ar,
        enable_fast_lewm=cfg.vla.enable_fast_lewm,
        enable_action_conditioned_dynamics=cfg.vla.enable_action_conditioned_dynamics,
        latent_patch_merge_size=cfg.vla.latent_patch_merge_size,
        lambda_latent_ar=cfg.vla.lambda_latent_ar,
        lambda_absolute_latent=cfg.vla.lambda_absolute_latent,
        lambda_horizon_consistency=cfg.vla.lambda_horizon_consistency,
        horizon_loss_weights=cfg.vla.horizon_loss_weights,
        action_horizon_weights=cfg.vla.action_horizon_weights,
        fast_lewm_num_prefixes=cfg.vla.fast_lewm_num_prefixes,
        fast_lewm_segment_targets=cfg.vla.fast_lewm_segment_targets,
        fast_lewm_prefix_dim=cfg.vla.fast_lewm_prefix_dim,
        fast_lewm_prefix_depth=cfg.vla.fast_lewm_prefix_depth,
        fast_lewm_prefix_heads=cfg.vla.fast_lewm_prefix_heads,
        fast_lewm_prefix_dropout=cfg.vla.fast_lewm_prefix_dropout,
        fast_lewm_action_conditioning=cfg.vla.fast_lewm_action_conditioning,
        fast_lewm_action_gradient_scale=cfg.vla.fast_lewm_action_gradient_scale,
        fast_lewm_query_cosine=cfg.vla.fast_lewm_query_cosine,
        fast_lewm_query_source=cfg.vla.fast_lewm_query_source,
        fast_lewm_head_type=cfg.vla.fast_lewm_head_type,
        fast_lewm_transformer_dim=cfg.vla.fast_lewm_transformer_dim,
        fast_lewm_transformer_depth=cfg.vla.fast_lewm_transformer_depth,
        fast_lewm_transformer_heads=cfg.vla.fast_lewm_transformer_heads,
        fast_lewm_transformer_mlp_dim=cfg.vla.fast_lewm_transformer_mlp_dim,
        fast_lewm_transformer_window_sizes=cfg.vla.fast_lewm_transformer_window_sizes,
        fast_lewm_transformer_dropout=cfg.vla.fast_lewm_transformer_dropout,
        # These reach the resume path through `load_vla`; without them here a
        # from-scratch run silently builds a different model than the same config
        # resumed -- no head gradient checkpointing (which OOM'd all sixteen
        # ranks), horizon embeddings back at their 0.02 default, and an
        # unnormalised target the encoder is free to shrink.
        fast_lewm_horizon_embedding_std=getattr(
            cfg.vla, "fast_lewm_horizon_embedding_std", 0.02
        ),
        fast_lewm_head_gradient_checkpointing=getattr(
            cfg.vla, "fast_lewm_head_gradient_checkpointing", False
        ),
        fast_lewm_normalize_target_scale=getattr(
            cfg.vla, "fast_lewm_normalize_target_scale", False
        ),
        fast_lewm_gate_bias_init=getattr(cfg.vla, "fast_lewm_gate_bias_init", 0.0),
        fast_lewm_pooled_latent_dim=getattr(cfg.vla, "fast_lewm_pooled_latent_dim", 256),
        fast_lewm_pooled_depth=getattr(cfg.vla, "fast_lewm_pooled_depth", 6),
        fast_lewm_pooled_hidden_dim=getattr(cfg.vla, "fast_lewm_pooled_hidden_dim", 2048),
        fast_lewm_pooled_fusion_dim=getattr(cfg.vla, "fast_lewm_pooled_fusion_dim", 768),
        fast_lewm_pooled_queries=getattr(cfg.vla, "fast_lewm_pooled_queries", 1),
        fast_lewm_per_sample_normalization=getattr(
            cfg.vla, "fast_lewm_per_sample_normalization", False
        ),
        fast_lewm_pooler_type=getattr(cfg.vla, "fast_lewm_pooler_type", "attention"),
        fast_lewm_pooler_depth=getattr(cfg.vla, "fast_lewm_pooler_depth", 2),
        fast_lewm_pooler_heads=getattr(cfg.vla, "fast_lewm_pooler_heads", 8),
        fast_lewm_pooler_mlp_dim=getattr(cfg.vla, "fast_lewm_pooler_mlp_dim", 2048),
        fast_lewm_directional_loss=getattr(cfg.vla, "fast_lewm_directional_loss", False),
        d_jepa=vision_backbone.embed_dim,
        cross_embodiments=cfg_cross_embodiments,
        unfreeze_vision_blocks=cfg.unfreeze_vision_blocks,
        unfreeze_projector=cfg.unfreeze_projector,
        vision_target_momentum=cfg.vision_target_momentum,
        use_ema_target=cfg.use_ema_target,
        unfreeze_llm=cfg.unfreeze_llm,
        lambda_latent_variance=cfg.lambda_latent_variance,
        lambda_latent_covariance=cfg.lambda_latent_covariance,
        latent_variance_floor=cfg.latent_variance_floor,
        lambda_latent_sigreg=cfg.lambda_latent_sigreg,
        sigreg_num_directions=cfg.sigreg_num_directions,
        lambda_pooled_latent_reg=cfg.lambda_pooled_latent_reg,
        lambda_temporal_hinge=cfg.lambda_temporal_hinge,
        lambda_query_diversity=cfg.lambda_query_diversity,
        lambda_pooled_variance=cfg.lambda_pooled_variance,
        lambda_pooled_covariance=cfg.lambda_pooled_covariance,
        lambda_action=cfg.lambda_action,
        fast_lewm_pooler_competitive=cfg.fast_lewm_pooler_competitive,
        pooled_variance_floor=cfg.pooled_variance_floor,
    )

    # The base run only provides pretrained projector + LLM weights.
    # Vision weights should come from the explicit V-JEPA checkpoint path above,
    # while VLA heads are newly initialized for Libero training.
    vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
    vlm.projector.load_state_dict(model_state_dict["projector"])
    # V-JEPA checkpoints can come in bf16; keep training initialization consistent
    # with the rest of the codepath by materializing the full train-time model in fp32.
    vlm = vlm.to(dtype=torch.float32)
    return vlm


def log_module_parameter_breakdown(vlm) -> None:
    def summarize_module(module) -> tuple[int, int]:
        total = sum(param.numel() for param in module.parameters())
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        return total, trainable

    module_specs = [
        ("vision_backbone", getattr(vlm, "vision_backbone", None)),
        ("projector", getattr(vlm, "projector", None)),
        ("llm_backbone", getattr(vlm, "llm_backbone", None)),
        ("action_head", getattr(vlm, "action_head", None)),
        ("visual_token_cosine_head", getattr(vlm, "visual_token_cosine_head", None)),
        ("official_latent_head", getattr(vlm, "official_latent_head", None)),
        ("latent_patch_codec", getattr(vlm, "latent_patch_codec", None)),
        ("fast_lewm_prefix_encoder", getattr(vlm, "fast_lewm_prefix_encoder", None)),
        ("fast_lewm_query_head", getattr(vlm, "fast_lewm_query_head", None)),
    ]

    lines = ["Module Parameter Breakdown:"]
    for name, module in module_specs:
        if module is None:
            continue
        total, trainable = summarize_module(module)
        lines.append(
            f"  - {name}: total={total / 10**6:.3f}M, trainable={trainable / 10**6:.3f}M"
        )

    llm_module = getattr(getattr(vlm, "llm_backbone", None), "llm", None)
    if llm_module is not None and hasattr(llm_module, "named_parameters"):
        lora_total = sum(param.numel() for name, param in llm_module.named_parameters() if "lora_" in name)
        lora_trainable = sum(
            param.numel() for name, param in llm_module.named_parameters() if "lora_" in name and param.requires_grad
        )
        if lora_total > 0:
            lines.append(
                f"  - llm_lora: total={lora_total / 10**6:.3f}M, trainable={lora_trainable / 10**6:.3f}M"
            )

    overwatch.info("\n".join(lines))


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(
            VLARegistry.JEPAVLA_QWEN25_VJEPA_224PX_0_5B_LIBERO_90.vla_id
        )
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Optional JEPA-WAM weights used to initialize a new run.
    initial_checkpoint: Optional[Path] = None
    # Resizing the world head means its checkpoint tensors no longer fit; every
    # other module still resumes.  Opt-in so a shape mismatch is never silent.
    reinitialize_world_head: bool = False
    # SIGReg (LeJEPA): drives the embedding distribution to an isotropic Gaussian,
    # which is a strictly stronger target than a variance/covariance pair -- set
    # this and zero the other two to swap regularizers.
    lambda_latent_sigreg: float = 0.0
    # Applied to the pooled latent stacked over current + future timesteps,
    # which is the only place a temporal collapse of the pooler is visible.
    lambda_pooled_latent_reg: float = 0.0
    # Zero while the latent is healthy, so its weight brakes without steering.
    lambda_temporal_hinge: float = 0.0
    lambda_query_diversity: float = 0.0
    lambda_pooled_variance: float = 0.0
    lambda_pooled_covariance: float = 0.0
    lambda_action: float = 1.0
    fast_lewm_pooler_competitive: bool = False
    pooled_variance_floor: float = 0.7
    sigreg_num_directions: int = 64
    resume_step: int = 0
    lr_decay_end_step: Optional[int] = None
    lr_milestones: Optional[str] = None

    # Cross-embodiment co-training. `co_train_specs` is a semicolon-separated list
    # of `<embodiment>:<data root>:<norm stats>` entries -- one per extra robot --
    # and every `co_train_every`-th micro-batch is drawn from them in turn. Empty
    # (the default) trains on the deployment robot alone.
    co_train_specs: Optional[str] = None
    co_train_every: int = 4
    # Fraction of micro-batches drawn from the co-training robots. Overrides
    # `co_train_every` when set; a ratio can express a co-training majority, which
    # matters because the deployment corpus is the small, easily-memorised one.
    co_train_ratio: Optional[float] = None
    co_train_max_episodes: Optional[int] = None
    cross_embodiments: Optional[dict] = None

    # Log a dataloader/forward/backward breakdown every N optimizer steps. With
    # several video streams feeding one model, whether a step is bound by the GPU
    # or by decoding is not something you can read off the step time alone.
    performance_timing_interval: int = 0

    # Fine-tune the visual frontend. 0 keeps it frozen (default); >0 unfreezes that
    # many trailing V-JEPA blocks and switches world-model targets to an EMA copy.
    unfreeze_vision_blocks: int = 0
    unfreeze_projector: bool = False
    unfreeze_llm: bool = False

    # Variance/covariance regularization on the encoder's latents. Only meaningful
    # with a trainable encoder; 0 leaves the loss exactly as it was.
    lambda_latent_variance: float = 0.0
    lambda_latent_covariance: float = 0.0
    latent_variance_floor: float = 1.0
    vision_target_momentum: float = 0.999
    use_ema_target: bool = True

    # Custom Local Paths (for JEPA-VLA and local model checkpoints)
    llm_checkpoint_path: Optional[Path] = None                      # Local path to LLM (e.g., Qwen2.5-0.5B)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logs and checkpoints
    run_id_note: Optional[str] = None                               # Optional suffix for the run ID
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    save_optimizer_state: bool = True                               # Exact resume; disable for robust model-only saves
    debug_batch_shapes: bool = True                                 # Print first training batch tensor shapes
    debug_memory_stats: bool = False                                # Print CUDA memory breakdown during training
    debug_memory_stats_interval: int = 0                            # Log memory every N optimizer steps; 0 disables
    cpu_memory_log_interval: int = 10                               # Log CPU/cgroup memory every N optimizer steps; 0 disables
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "swanlab")
    swanlab_project: str = "jepa-wam"
    swanlab_entity: Optional[str] = None
    use_swanlab: bool = False

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.min_learning_rate = self.vla.min_learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.warmup_ratio = self.vla.warmup_ratio
        if self.cpu_memory_log_interval < 0:
            raise ValueError("cpu_memory_log_interval must be non-negative.")
        if self.resume_step < 0 or self.resume_step >= self.max_steps:
            raise ValueError("resume_step must satisfy 0 <= resume_step < max_steps.")
        if self.resume_step and self.initial_checkpoint is None:
            raise ValueError("resume_step requires initial_checkpoint.")
        if self.lr_decay_end_step is not None and not self.resume_step < self.lr_decay_end_step <= self.max_steps:
            raise ValueError("lr_decay_end_step must satisfy resume_step < lr_decay_end_step <= max_steps.")

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on



def parse_co_train_specs(spec: Optional[str]):
    """`<embodiment>:<data root>:<norm stats>[;...]` -> [(name, dims, root, stats)].

    Each embodiment's action width, proprio width and chunk length are read from
    the dataset's own metadata rather than declared here, so a mirror recorded at a
    different frame rate cannot silently train the shared head on the wrong horizon.
    """
    if not spec or not spec.strip():
        return []

    from prismatic.vla.datasets.droid_worldmodel import droid_embodiment_spec
    from prismatic.vla.datasets.multitask_worldmodel import SPECS, embodiment_spec_for

    parsed = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        fields = entry.split(":")
        if len(fields) != 3:
            raise ValueError(f"co-train entry must be `<embodiment>:<root>:<stats>`, got `{entry}`")
        embodiment, root, stats = (field.strip() for field in fields)
        if embodiment in SPECS:
            name, dims = embodiment_spec_for(embodiment, root)
        elif embodiment == "droid_qpos8":
            name, dims = droid_embodiment_spec(root)
        else:
            known = sorted(list(SPECS) + ["droid_qpos8"])
            raise ValueError(f"unknown embodiment `{embodiment}`; known: {known}")
        parsed.append((name, dims, Path(root), Path(stats)))
    return parsed


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("JEPA-WAM Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    from datetime import datetime
    cfg.run_id += f"--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Start =>> Build Directories and Set Randomness
    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip() if cfg.hf_token.exists() else None
    else:
        hf_token = os.environ.get(cfg.hf_token)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    cross_embodiments = {
        name: spec for name, spec, _, _ in parse_co_train_specs(cfg.co_train_specs)
    }
    # Recorded so inference can rebuild the same robot set without re-deriving it.
    cfg.cross_embodiments = dict(cross_embodiments)
    for name, spec in cross_embodiments.items():
        overwatch.info("Cross-embodiment co-training: %s -> d_action=%d d_proprio=%d horizon=%d",
                       name, *spec)

    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.initial_checkpoint is not None:
        vlm = load_vla(
            cfg.initial_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            enable_official_latent_head=cfg.vla.enable_official_latent_head,
            lambda_official_latent=cfg.vla.lambda_official_latent,
            base_vlm=cfg.vla.base_vlm,
            llm_checkpoint_path=str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else None,
            vjepa_checkpoint_path=cfg.vla.vjepa_checkpoint_path,
            enable_five_tubelet_ar=cfg.vla.enable_five_tubelet_ar,
            enable_fast_lewm=cfg.vla.enable_fast_lewm,
            enable_action_conditioned_dynamics=cfg.vla.enable_action_conditioned_dynamics,
            latent_patch_merge_size=cfg.vla.latent_patch_merge_size,
            lambda_visual_token_cosine=cfg.vla.lambda_visual_token_cosine,
            lambda_latent_ar=cfg.vla.lambda_latent_ar,
            lambda_absolute_latent=cfg.vla.lambda_absolute_latent,
            lambda_horizon_consistency=cfg.vla.lambda_horizon_consistency,
            horizon_loss_weights=cfg.vla.horizon_loss_weights,
            action_horizon_weights=cfg.vla.action_horizon_weights,
            fast_lewm_num_prefixes=cfg.vla.fast_lewm_num_prefixes,
            fast_lewm_segment_targets=cfg.vla.fast_lewm_segment_targets,
            fast_lewm_prefix_dim=cfg.vla.fast_lewm_prefix_dim,
            fast_lewm_prefix_depth=cfg.vla.fast_lewm_prefix_depth,
            fast_lewm_prefix_heads=cfg.vla.fast_lewm_prefix_heads,
            fast_lewm_prefix_dropout=cfg.vla.fast_lewm_prefix_dropout,
            fast_lewm_action_conditioning=cfg.vla.fast_lewm_action_conditioning,
            fast_lewm_action_gradient_scale=cfg.vla.fast_lewm_action_gradient_scale,
            fast_lewm_query_cosine=cfg.vla.fast_lewm_query_cosine,
            fast_lewm_head_type=cfg.vla.fast_lewm_head_type,
            fast_lewm_transformer_dim=cfg.vla.fast_lewm_transformer_dim,
            fast_lewm_transformer_depth=cfg.vla.fast_lewm_transformer_depth,
            fast_lewm_transformer_heads=cfg.vla.fast_lewm_transformer_heads,
            fast_lewm_transformer_mlp_dim=cfg.vla.fast_lewm_transformer_mlp_dim,
            fast_lewm_transformer_window_sizes=cfg.vla.fast_lewm_transformer_window_sizes,
            fast_lewm_transformer_dropout=cfg.vla.fast_lewm_transformer_dropout,
            fast_lewm_horizon_embedding_std=getattr(
                cfg.vla, "fast_lewm_horizon_embedding_std", 0.02
            ),
            fast_lewm_head_gradient_checkpointing=getattr(
                cfg.vla, "fast_lewm_head_gradient_checkpointing", False
            ),
            fast_lewm_normalize_target_scale=getattr(
                cfg.vla, "fast_lewm_normalize_target_scale", False
            ),
            fast_lewm_gate_bias_init=getattr(cfg.vla, "fast_lewm_gate_bias_init", 0.0),
            fast_lewm_pooled_latent_dim=getattr(cfg.vla, "fast_lewm_pooled_latent_dim", 256),
            fast_lewm_pooled_depth=getattr(cfg.vla, "fast_lewm_pooled_depth", 6),
            fast_lewm_pooled_hidden_dim=getattr(cfg.vla, "fast_lewm_pooled_hidden_dim", 2048),
            fast_lewm_pooled_fusion_dim=getattr(cfg.vla, "fast_lewm_pooled_fusion_dim", 768),
            fast_lewm_pooled_queries=getattr(cfg.vla, "fast_lewm_pooled_queries", 1),
            fast_lewm_per_sample_normalization=getattr(
                cfg.vla, "fast_lewm_per_sample_normalization", False
            ),
            fast_lewm_pooler_type=getattr(cfg.vla, "fast_lewm_pooler_type", "attention"),
            fast_lewm_pooler_depth=getattr(cfg.vla, "fast_lewm_pooler_depth", 2),
            fast_lewm_pooler_heads=getattr(cfg.vla, "fast_lewm_pooler_heads", 8),
            fast_lewm_pooler_mlp_dim=getattr(cfg.vla, "fast_lewm_pooler_mlp_dim", 2048),
            fast_lewm_directional_loss=getattr(cfg.vla, "fast_lewm_directional_loss", False),
            reinitialize_world_head=cfg.reinitialize_world_head,
            lambda_latent_sigreg=cfg.lambda_latent_sigreg,
            sigreg_num_directions=cfg.sigreg_num_directions,
            lambda_pooled_latent_reg=cfg.lambda_pooled_latent_reg,
            lambda_temporal_hinge=cfg.lambda_temporal_hinge,
            lambda_query_diversity=cfg.lambda_query_diversity,
            lambda_pooled_variance=cfg.lambda_pooled_variance,
            lambda_pooled_covariance=cfg.lambda_pooled_covariance,
            lambda_action=cfg.lambda_action,
            fast_lewm_pooler_competitive=cfg.fast_lewm_pooler_competitive,
            pooled_variance_floor=cfg.pooled_variance_floor,
            action_prediction_type=cfg.vla.action_prediction_type,
            fast_lewm_query_source=cfg.vla.fast_lewm_query_source,
            cross_embodiments=cross_embodiments,
            unfreeze_vision_blocks=cfg.unfreeze_vision_blocks,
            unfreeze_projector=cfg.unfreeze_projector,
            vision_target_momentum=cfg.vision_target_momentum,
            use_ema_target=cfg.use_ema_target,
            unfreeze_llm=cfg.unfreeze_llm,
            lambda_latent_variance=cfg.lambda_latent_variance,
            lambda_latent_covariance=cfg.lambda_latent_covariance,
            latent_variance_floor=cfg.latent_variance_floor,
        )
        vlm = vlm.to(dtype=torch.float32)

    else:
        overwatch.info(f"Rebuilding base VLM `{cfg.vla.base_vlm}` with the fixed JEPA-WAM heads")
        vlm = build_vla_from_base_vlm(cfg.vla.base_vlm, cfg, hf_token)

    overwatch.info(
        "Applying Qwen LoRA (rank=%d, alpha=%d, targets=%s)",
        cfg.vla.lora_rank,
        cfg.vla.lora_alpha,
        cfg.vla.lora_target_modules,
    )
    apply_lora_to_vlm(vlm, cfg.vla, unfreeze_llm=cfg.unfreeze_llm)

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    overwatch.info("Freezing the base V-JEPA, projector, and Qwen weights")
    vlm.freeze_for_training()
    vlm.debug_memory_stats = cfg.debug_memory_stats

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    log_module_parameter_breakdown(vlm)

    # Get VLA Dataset & Collator
    if cfg.vla.data_mix in {
        "robotwin_paper20_clean",
        "robotwin_paper20_group1_clean",
        "robotwin_paper20_group2_clean",
    }:
        if not cfg.vla.normalization_stats_path:
            raise ValueError("RoboTwin training requires vla.normalization_stats_path")
        overwatch.info("Creating RoboTwin paper clean LeRobot dataset: %s", cfg.vla.data_mix)
        tokenizer = vlm.llm_backbone.get_tokenizer()
        vla_dataset = RoboTwinPaperDataset(
            RoboTwinPaperDatasetConfig(
                root=cfg.data_root_dir,
                stats_path=Path(cfg.vla.normalization_stats_path),
                seed=cfg.seed,
                num_workers=cfg.vla.dataloader_num_workers,
                prefetch_factor=cfg.vla.dataloader_prefetch_factor,
                # Selects the current-pair + five-future-frame sample layout.
                # Every world-model path needs it, not just five-tubelet AR --
                # without it the batch carries no `current_frame_pairs` and
                # `forward()` silently falls through to the plain VLA path.
                enable_five_tubelet_ar=(
                    cfg.vla.enable_five_tubelet_ar
                    or cfg.vla.enable_fast_lewm
                    or cfg.vla.enable_action_conditioned_dynamics
                ),
                enable_photometric_augmentation=cfg.vla.enable_photometric_augmentation,
                photometric_augmentation_probability=cfg.vla.photometric_augmentation_probability,
                photometric_augmentation_strength=cfg.vla.photometric_augmentation_strength,
                episode_selection=cfg.vla.data_mix,
            ),
            image_transform=vlm.vision_backbone.get_image_transform(),
            tokenizer=tokenizer,
            prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
            rank=overwatch.rank(),
            world_size=overwatch.world_size(),
        )
        collator = PaddedCollatorForActionPrediction(
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side="right",
            target_action_dim=cfg.vla.d_action,
            target_proprio_dim=cfg.vla.d_proprio,
        )
    else:
        overwatch.info(f"Creating LIBERO RLDS dataset: mixture={cfg.vla.data_mix}")
        vla_dataset, collator = get_vla_dataset_and_collator(
            cfg.data_root_dir,
            cfg.vla.data_mix,
            release_statistics=(vlm.norm_stats if cfg.vla.enable_official_latent_head else None),
            image_transform=vlm.vision_backbone.get_image_transform(),
            tokenizer=vlm.llm_backbone.get_tokenizer(),
            prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
            default_image_resolution=vlm.vision_backbone.default_image_resolution,
            shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
            visual_token_pair_offset=cfg.vla.visual_token_pair_offset,
            # Same layout the RoboTwin loader builds: the observed pair
            # `(anchor - 1, anchor)` followed by one future anchor per world-head
            # horizon, spread evenly over the action horizon, so horizon k lines
            # up with the actions the policy is being asked to predict.  The
            # count tracks `fast_lewm_num_prefixes` because the head's horizon
            # embeddings, the action blocks and the supervision grids must agree.
            world_frame_offsets=(
                (-1, 0) + tuple(
                    round(
                        cfg.vla.action_horizon
                        * (index + 1)
                        / cfg.vla.fast_lewm_num_prefixes
                    )
                    for index in range(cfg.vla.fast_lewm_num_prefixes)
                )
                if (
                    cfg.vla.enable_five_tubelet_ar
                    or cfg.vla.enable_fast_lewm
                    or cfg.vla.enable_action_conditioned_dynamics
                )
                else ()
            ),
            target_action_dim=cfg.vla.d_action,
            target_proprio_dim=cfg.vla.d_proprio,
        )

    global_dataset_length = getattr(vla_dataset, "global_dataset_length", len(vla_dataset))
    overwatch.info(
        "VLA dataset backend: class=%s global_examples=%d local_examples=%d",
        type(vla_dataset).__name__,
        global_dataset_length,
        len(vla_dataset),
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info("Initializing fixed FSDP full-shard strategy")
    train_strategy = get_fsdp_strategy(
        vlm=vlm,
        device_id=device_id,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        resume_step=cfg.resume_step,
        lr_decay_end_step=cfg.lr_decay_end_step,
        lr_milestones=cfg.lr_milestones,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        save_optimizer_state=cfg.save_optimizer_state,
    )
    train_strategy.cpu_memory_log_interval = cfg.cpu_memory_log_interval
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=global_dataset_length)

    # Create Metrics =>> Handles JSONL and optional SwanLab tracking.
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        swanlab_project=cfg.swanlab_project,
        swanlab_entity=cfg.swanlab_entity,
        use_swanlab=cfg.use_swanlab,
    )
    train_strategy.debug_batch_shapes = cfg.debug_batch_shapes
    train_strategy.debug_memory_stats = cfg.debug_memory_stats
    train_strategy.debug_memory_stats_interval = cfg.debug_memory_stats_interval
    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    co_train_streams = []
    for name, dims, root, stats in parse_co_train_specs(cfg.co_train_specs):
        # A co-training stream supplies one micro-batch in `co_train_every * n_streams`,
        # so giving it the deployment stream's worker count buys nothing and costs a
        # lot: persistent workers prefetch continuously, so every loader's workers are
        # resident at once, and three full-sized pools per rank oversubscribe the node
        # badly enough to stall a rank past the collective timeout.
        co_train_workers = max(1, cfg.vla.dataloader_num_workers // 4)
        shared = dict(
            seed=cfg.seed,
            num_workers=co_train_workers,
            prefetch_factor=cfg.vla.dataloader_prefetch_factor,
            enable_photometric_augmentation=cfg.vla.enable_photometric_augmentation,
            photometric_augmentation_probability=cfg.vla.photometric_augmentation_probability,
            photometric_augmentation_strength=cfg.vla.photometric_augmentation_strength,
        )
        builders = dict(
            image_transform=vlm.vision_backbone.get_image_transform(),
            tokenizer=vlm.llm_backbone.get_tokenizer(),
            prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
            rank=overwatch.rank(),
            world_size=overwatch.world_size(),
        )
        if name == "droid_qpos8":
            from prismatic.vla.datasets.droid_worldmodel import (
                DroidWorldModelDataset,
                DroidWorldModelDatasetConfig,
            )

            dataset = DroidWorldModelDataset(
                DroidWorldModelDatasetConfig(
                    root=root, stats_path=stats, max_episodes=cfg.co_train_max_episodes, **shared
                ),
                **builders,
            )
        else:
            from prismatic.vla.datasets.multitask_worldmodel import (
                MultiTaskWorldModelDataset,
                MultiTaskWorldModelDatasetConfig,
            )

            dataset = MultiTaskWorldModelDataset(
                MultiTaskWorldModelDatasetConfig(
                    root=root,
                    stats_path=stats,
                    spec_name=name,
                    max_episodes_per_task=cfg.co_train_max_episodes,
                    **shared,
                ),
                **builders,
            )
        # The collator pads actions/proprio to a fixed width, and that width is
        # per-robot -- reusing the deployment robot's collator would reject every
        # batch from a robot with a wider action space.
        stream_tokenizer = vlm.llm_backbone.get_tokenizer()
        stream_collator = PaddedCollatorForActionPrediction(
            model_max_length=stream_tokenizer.model_max_length,
            pad_token_id=stream_tokenizer.pad_token_id,
            padding_side="right",
            target_action_dim=dims[0],
            target_proprio_dim=dims[1],
        )
        overwatch.info(
            "Co-training stream ready: %s from %s (d_action=%d d_proprio=%d, %d workers)",
            name, root, dims[0], dims[1], co_train_workers,
        )
        co_train_streams.append((dataset, stream_collator))

    train_strategy.performance_timing_interval = cfg.performance_timing_interval

    train_strategy.run_vla_training(
        vla_dataset=vla_dataset,
        collator=collator,
        metrics=metrics,
        save_interval=cfg.save_interval,
        co_train_streams=co_train_streams,
        co_train_every=cfg.co_train_every if co_train_streams else 0,
        co_train_ratio=cfg.co_train_ratio if co_train_streams else None,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
