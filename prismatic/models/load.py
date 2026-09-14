"""Load local JEPA-WAM checkpoints produced by the public training recipe."""

import json
import os
from pathlib import Path
from typing import Optional, Union

import torch

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlas import OpenVLA
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.vla.embodiments import cross_embodiments_from_spec

overwatch = initialize_overwatch(__name__)


def _normalize_lora_target_modules(target_modules):
    if isinstance(target_modules, tuple):
        return list(target_modules)
    if isinstance(target_modules, str) and "," in target_modules:
        return [item.strip() for item in target_modules.split(",") if item.strip()]
    return target_modules


def _apply_lora_to_llm_backbone(llm_backbone, vla_cfg: dict, is_trainable: bool) -> None:
    from peft import LoraConfig, get_peft_model

    if hasattr(llm_backbone.llm, "peft_config"):
        return
    llm_backbone.llm = get_peft_model(
        llm_backbone.llm,
        LoraConfig(
            r=int(vla_cfg.get("lora_rank", 32)),
            lora_alpha=int(vla_cfg.get("lora_alpha", 64)),
            target_modules=_normalize_lora_target_modules(vla_cfg.get("lora_target_modules", "all-linear")),
            lora_dropout=float(vla_cfg.get("lora_dropout", 0.1)),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            inference_mode=not is_trainable,
        ),
    )


def _path_candidates(raw: Union[str, Path]) -> list[Path]:
    """Every place a recorded absolute path might actually live on this host.

    A checkpoint written on another machine records paths against a top-level
    mount (e.g. `/path/to/local_resource`) that only exists there as a symlink
    into a home directory. Retrying under each configured root keeps a missing
    symlink on this host from being a hard failure.
    """
    candidates = [Path(os.path.expanduser(str(raw)))]
    if candidates[0].is_absolute():
        for root in os.environ.get("JEPA_WAM_PATH_ROOTS", "").split(":"):
            if root:
                candidates.append(Path(root) / str(candidates[0]).lstrip("/"))
    return candidates


def _resolve_existing_path(raw: Optional[Union[str, Path]], *, what: str) -> Optional[str]:
    """Return the first candidate that exists, or the original path unchanged."""
    if not raw:
        return raw if raw is None else str(raw)
    candidates = _path_candidates(raw)
    for path in candidates:
        if path.exists():
            if path is not candidates[0]:
                overwatch.info(f"Resolved {what} under a fallback root: `{path}`")
            return str(path)
    # Fall through with the original so the downstream loader raises the error
    # naming the path the checkpoint actually recorded.
    return str(candidates[0])


def _resolve_base_config(base_vlm: Union[str, Path]) -> tuple[Path, dict]:
    candidates = _path_candidates(base_vlm)
    for path in candidates:
        run_dir = path if path.is_dir() else path.parent.parent
        config_path = run_dir / "config.json"
        if config_path.exists():
            if path is not candidates[0]:
                overwatch.info(f"Resolved base VLM under a fallback root: `{run_dir}`")
            with open(config_path, "r") as handle:
                return run_dir, json.load(handle)["model"]

    searched = ", ".join(
        f"`{(p if p.is_dir() else p.parent.parent) / 'config.json'}`" for p in candidates
    )
    raise ValueError(f"Base VLM config not found; searched: {searched}")


def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    load_for_training: bool = False,
    base_vlm: Optional[Union[str, Path]] = None,
    llm_checkpoint_path: Optional[str] = None,
    vjepa_checkpoint_path: Optional[str] = None,
    load_visual_token_cosine_head: bool = True,
    enable_official_latent_head: Optional[bool] = None,
    lambda_official_latent: Optional[float] = None,
    enable_five_tubelet_ar: Optional[bool] = None,
    enable_fast_lewm: Optional[bool] = None,
    enable_action_conditioned_dynamics: Optional[bool] = None,
    latent_patch_merge_size: Optional[int] = None,
    lambda_latent_ar: Optional[float] = None,
    lambda_visual_token_cosine: Optional[float] = None,
    lambda_absolute_latent: Optional[float] = None,
    lambda_horizon_consistency: Optional[float] = None,
    horizon_loss_weights: Optional[tuple[float, ...]] = None,
    action_horizon_weights: Optional[tuple[float, ...]] = None,
    fast_lewm_num_prefixes: Optional[int] = None,
    fast_lewm_segment_targets: Optional[bool] = None,
    fast_lewm_prefix_dim: Optional[int] = None,
    fast_lewm_prefix_depth: Optional[int] = None,
    fast_lewm_prefix_heads: Optional[int] = None,
    fast_lewm_prefix_dropout: Optional[float] = None,
    fast_lewm_action_conditioning: Optional[str] = None,
    fast_lewm_action_gradient_scale: Optional[float] = None,
    fast_lewm_query_cosine: Optional[bool] = None,
    fast_lewm_head_type: Optional[str] = None,
    fast_lewm_transformer_dim: Optional[int] = None,
    fast_lewm_transformer_depth: Optional[int] = None,
    fast_lewm_transformer_heads: Optional[int] = None,
    fast_lewm_transformer_mlp_dim: Optional[int] = None,
    fast_lewm_transformer_window_sizes: Optional[tuple[int, ...]] = None,
    fast_lewm_transformer_dropout: Optional[float] = None,
    fast_lewm_horizon_embedding_std: Optional[float] = None,
    fast_lewm_head_gradient_checkpointing: Optional[bool] = None,
    fast_lewm_normalize_target_scale: Optional[bool] = None,
    fast_lewm_gate_bias_init: Optional[float] = None,
    fast_lewm_pooled_latent_dim: Optional[int] = None,
    fast_lewm_pooled_depth: Optional[int] = None,
    fast_lewm_pooled_hidden_dim: Optional[int] = None,
    fast_lewm_pooled_fusion_dim: Optional[int] = None,
    fast_lewm_pooled_queries: Optional[int] = None,
    fast_lewm_per_sample_normalization: Optional[bool] = None,
    fast_lewm_pooler_type: Optional[str] = None,
    fast_lewm_pooler_depth: Optional[int] = None,
    fast_lewm_pooler_heads: Optional[int] = None,
    fast_lewm_pooler_mlp_dim: Optional[int] = None,
    fast_lewm_directional_loss: Optional[bool] = None,
    reinitialize_world_head: bool = False,
    lambda_latent_sigreg: Optional[float] = None,
    lambda_pooled_latent_reg: Optional[float] = None,
    lambda_temporal_hinge: Optional[float] = None,
    lambda_query_diversity: Optional[float] = None,
    lambda_pooled_variance: Optional[float] = None,
    lambda_pooled_covariance: Optional[float] = None,
    lambda_action: Optional[float] = None,
    fast_lewm_pooler_competitive: Optional[bool] = None,
    pooled_variance_floor: Optional[float] = None,
    sigreg_num_directions: Optional[int] = None,
    action_prediction_type: Optional[str] = None,
    fast_lewm_query_source: Optional[str] = None,
    cross_embodiments: Optional[dict] = None,
    unfreeze_vision_blocks: Optional[int] = None,
    unfreeze_projector: Optional[bool] = None,
    unfreeze_llm: Optional[bool] = None,
    vision_target_momentum: Optional[float] = None,
    use_ema_target: Optional[bool] = None,
    **_: object,
) -> OpenVLA:
    checkpoint_path = Path(os.path.expanduser(str(model_id_or_path)))
    if not checkpoint_path.is_file() or checkpoint_path.suffix != ".pt" or checkpoint_path.parent.name != "checkpoints":
        raise ValueError("JEPA-WAM loading requires a local `runs/.../checkpoints/*.pt` checkpoint.")

    run_dir = checkpoint_path.parents[1]
    config_path = run_dir / "config.json"
    statistics_path = run_dir / "dataset_statistics.json"
    if not config_path.exists() or not statistics_path.exists():
        raise ValueError(f"Checkpoint run directory must contain config.json and dataset_statistics.json: `{run_dir}`")

    with open(config_path, "r") as handle:
        full_cfg = json.load(handle)
    with open(statistics_path, "r") as handle:
        norm_stats = json.load(handle)
    vla_cfg = full_cfg["vla"]

    def _recorded(name: str, default=None):
        """Read a training flag from wherever the run recorded it.

        Which block a setting lands in depends on which dataclass declares it, and
        the split is not something a caller can see: `lambda_latent_ar` is inside
        `vla` while `lambda_temporal_hinge`, `unfreeze_vision_blocks` and
        `fast_lewm_pooler_competitive` are at the top level. Reading only `vla`
        silently falls back to the default and rebuilds a different model -- a run
        trained with competitive pooling reloaded as an ordinary pooler, whose probe
        then reported a representation twice as bad as it was. Every recorded
        setting goes through here so that cannot happen again.
        """
        if name in full_cfg:
            return full_cfg[name]
        return vla_cfg.get(name, default)
    effective_five_tubelet_ar = (
        bool(_recorded("enable_five_tubelet_ar", False))
        if enable_five_tubelet_ar is None else bool(enable_five_tubelet_ar)
    )
    effective_fast_lewm = (
        bool(_recorded("enable_fast_lewm", False))
        if enable_fast_lewm is None else bool(enable_fast_lewm)
    )
    effective_action_conditioned_dynamics = (
        bool(_recorded("enable_action_conditioned_dynamics", False))
        if enable_action_conditioned_dynamics is None else bool(enable_action_conditioned_dynamics)
    )
    if sum((effective_five_tubelet_ar, effective_fast_lewm, effective_action_conditioned_dynamics)) > 1:
        raise ValueError(
            "five-tubelet AR, Fast-LeWM, and action-conditioned dynamics loading modes "
            "are mutually exclusive"
        )

    def _recorded_cross_embodiments() -> dict:
        """Rebuild the co-training robots' dimensions from what the run recorded.

        Only the deployment robot is ever driven at inference, but its action-prefix
        projections live under a per-robot key once co-training is on, so the model
        has to be constructed with the same set of robots or the checkpoint's keys
        will not line up.
        """
        recorded = full_cfg.get("cross_embodiments") or _recorded("cross_embodiments")
        if recorded:
            return {name: tuple(dims) for name, dims in recorded.items()}
        spec = full_cfg.get("co_train_specs") or _recorded("co_train_specs")
        embodiments = cross_embodiments_from_spec(spec)
        if spec and not embodiments:
            overwatch.warning(f"Co-training spec `{spec}` names no known embodiment")
        return embodiments

    effective_unfreeze_llm = bool(
        _recorded("unfreeze_llm", False) if unfreeze_llm is None else unfreeze_llm
    )
    effective_unfreeze_projector = bool(
        _recorded("unfreeze_projector", False) if unfreeze_projector is None else unfreeze_projector
    )
    effective_unfreeze_vision_blocks = int(
        _recorded("unfreeze_vision_blocks", 0) if unfreeze_vision_blocks is None else unfreeze_vision_blocks
    )

    base_source = base_vlm or _recorded("base_vlm")
    if not base_source:
        raise ValueError("Pass a local base VLM run directory through `base_vlm`.")
    _, model_cfg = _resolve_base_config(base_source)

    vision_id = model_cfg.get("vision_backbone_id")
    llm_id = model_cfg.get("llm_backbone_id")
    vision_checkpoint = vjepa_checkpoint_path or _recorded("vjepa_checkpoint_path") or model_cfg.get(
        "vision_checkpoint_path"
    )
    llm_checkpoint = llm_checkpoint_path or full_cfg.get("llm_checkpoint_path") or model_cfg.get("llm_local_path")
    if not vision_checkpoint or not llm_checkpoint:
        raise ValueError("Both V-JEPA and Qwen checkpoint paths are required to reconstruct JEPA-WAM.")
    # These are recorded absolute paths too, so they need the same host-root fallback.
    vision_checkpoint = _resolve_existing_path(vision_checkpoint, what="V-JEPA checkpoint")
    llm_checkpoint = _resolve_existing_path(llm_checkpoint, what="Qwen checkpoint")

    overwatch.info(f"Loading JEPA-WAM checkpoint `{checkpoint_path}`")
    vision_backbone, _ = get_vision_backbone_and_transform(
        vision_id,
        model_cfg.get("image_resize_strategy", "resize-naive"),
        checkpoint_path=str(vision_checkpoint),
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        llm_id,
        llm_max_length=int(model_cfg.get("llm_max_length", 32_768)),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        custom_hf_path=str(llm_checkpoint),
        # All three world-model paths feed Qwen a custom 4D additive attention
        # mask; FlashAttention-2 does not honour those, so they must run on SDPA.
        use_flash_attention_2=not (
            effective_five_tubelet_ar
            or effective_fast_lewm
            or effective_action_conditioned_dynamics
        ),
    )
    if effective_unfreeze_llm:
        overwatch.info("Checkpoint trained Qwen with full parameters; not wrapping it in LoRA.")
    else:
        _apply_lora_to_llm_backbone(llm_backbone, vla_cfg, is_trainable=load_for_training)

    model = OpenVLA.from_pretrained(
        checkpoint_path,
        model_cfg.get("model_id", "prism-qwen25-vjepa21-vitl-384px+0_5b"),
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.get("arch_specifier", "no-align+gelu-mlp"),
        freeze_weights=not load_for_training,
        load_visual_token_cosine_head=load_visual_token_cosine_head,
        enable_official_latent_head=(bool(_recorded("enable_official_latent_head", False))
                                    if enable_official_latent_head is None else enable_official_latent_head),
        lambda_official_latent=(float(_recorded("lambda_official_latent", 0.15))
                               if lambda_official_latent is None else lambda_official_latent),
        reinitialize_world_head=reinitialize_world_head,
        norm_stats=norm_stats,
        d_action=int(_recorded("d_action", 7)),
        d_proprio=int(_recorded("d_proprio", 8)),
        action_horizon=int(_recorded("action_horizon", NUM_ACTIONS_CHUNK)),
        flow_gr00t_placeholder_tokens=int(_recorded("flow_gr00t_placeholder_tokens", 64)),
        fm_hidden_size=int(_recorded("fm_hidden_size", 1024)),
        fm_num_layers=int(_recorded("fm_num_layers", 16)),
        fm_num_inference_timesteps=int(_recorded("fm_num_inference_timesteps", 4)),
        fm_num_timestep_buckets=int(_recorded("fm_num_timestep_buckets", 1000)),
        fm_noise_beta_alpha=float(_recorded("fm_noise_beta_alpha", 1.5)),
        fm_noise_beta_beta=float(_recorded("fm_noise_beta_beta", 1.0)),
        fm_noise_s=float(_recorded("fm_noise_s", 0.999)),
        fm_num_target_vision_tokens=int(_recorded("fm_num_target_vision_tokens", 32)),
        fm_add_pos_embed=bool(_recorded("fm_add_pos_embed", True)),
        fm_max_seq_len=int(_recorded("fm_max_seq_len", 1024)),
        fm_state_dropout=float(_recorded("fm_state_dropout", 0.5)),
        action_prediction_type=(
            str(_recorded("action_prediction_type", "velocity"))
            if action_prediction_type is None else str(action_prediction_type)
        ),
        # The recorded value belongs to the run being resumed; an explicit
        # argument is the caller's own config and must win over it.
        lambda_visual_token_cosine=(
            float(_recorded("lambda_visual_token_cosine", 0.5))
            if lambda_visual_token_cosine is None else float(lambda_visual_token_cosine)
        ),
        enable_five_tubelet_ar=effective_five_tubelet_ar,
        enable_fast_lewm=effective_fast_lewm,
        enable_action_conditioned_dynamics=effective_action_conditioned_dynamics,
        # `None` means "use whatever the run recorded"; an empty dict is an explicit
        # request for a single-robot model, and must not fall back to the recorded set.
        cross_embodiments=(
            _recorded_cross_embodiments() if cross_embodiments is None else cross_embodiments
        ),
        unfreeze_vision_blocks=effective_unfreeze_vision_blocks,
        unfreeze_projector=effective_unfreeze_projector,
        unfreeze_llm=effective_unfreeze_llm,
        vision_target_momentum=float(_recorded("vision_target_momentum", 0.999) if vision_target_momentum is None else vision_target_momentum),
        use_ema_target=bool(_recorded("use_ema_target", True) if use_ema_target is None else use_ema_target),
        latent_patch_merge_size=(
            int(_recorded("latent_patch_merge_size", 2))
            if latent_patch_merge_size is None else int(latent_patch_merge_size)
        ),
        lambda_latent_ar=(
            float(_recorded("lambda_latent_ar", 0.5))
            if lambda_latent_ar is None else float(lambda_latent_ar)
        ),
        lambda_absolute_latent=(
            float(_recorded("lambda_absolute_latent", 0.02))
            if lambda_absolute_latent is None else float(lambda_absolute_latent)
        ),
        lambda_horizon_consistency=(
            float(_recorded("lambda_horizon_consistency", 0.02))
            if lambda_horizon_consistency is None else float(lambda_horizon_consistency)
        ),
        horizon_loss_weights=(
            tuple(_recorded("horizon_loss_weights", (1.0, 0.8, 0.6, 0.4, 0.3)))
            if horizon_loss_weights is None else tuple(horizon_loss_weights)
        ),
        action_horizon_weights=(
            tuple(_recorded("action_horizon_weights", (1.0, 0.8, 0.6, 0.4, 0.3)))
            if action_horizon_weights is None else tuple(action_horizon_weights)
        ),
        fast_lewm_num_prefixes=(
            int(_recorded("fast_lewm_num_prefixes", 5))
            if fast_lewm_num_prefixes is None else int(fast_lewm_num_prefixes)
        ),
        fast_lewm_segment_targets=(
            bool(_recorded("fast_lewm_segment_targets", False))
            if fast_lewm_segment_targets is None else bool(fast_lewm_segment_targets)
        ),
        fast_lewm_prefix_dim=(
            int(_recorded("fast_lewm_prefix_dim", 192))
            if fast_lewm_prefix_dim is None else int(fast_lewm_prefix_dim)
        ),
        fast_lewm_prefix_depth=(
            int(_recorded("fast_lewm_prefix_depth", 3))
            if fast_lewm_prefix_depth is None else int(fast_lewm_prefix_depth)
        ),
        fast_lewm_prefix_heads=(
            int(_recorded("fast_lewm_prefix_heads", 6))
            if fast_lewm_prefix_heads is None else int(fast_lewm_prefix_heads)
        ),
        fast_lewm_prefix_dropout=(
            float(_recorded("fast_lewm_prefix_dropout", 0.0))
            if fast_lewm_prefix_dropout is None else float(fast_lewm_prefix_dropout)
        ),
        fast_lewm_action_conditioning=(
            str(_recorded("fast_lewm_action_conditioning", "ground_truth"))
            if fast_lewm_action_conditioning is None else str(fast_lewm_action_conditioning)
        ),
        fast_lewm_action_gradient_scale=(
            float(_recorded("fast_lewm_action_gradient_scale", 0.0))
            if fast_lewm_action_gradient_scale is None else float(fast_lewm_action_gradient_scale)
        ),
        fast_lewm_query_source=(
            str(_recorded("fast_lewm_query_source", "llm_hidden"))
            if fast_lewm_query_source is None else str(fast_lewm_query_source)
        ),
        fast_lewm_query_cosine=(
            bool(_recorded("fast_lewm_query_cosine", False))
            if fast_lewm_query_cosine is None else bool(fast_lewm_query_cosine)
        ),
        fast_lewm_head_type=(
            str(_recorded("fast_lewm_head_type", "shared_mlp"))
            if fast_lewm_head_type is None else str(fast_lewm_head_type)
        ),
        fast_lewm_transformer_dim=(
            int(_recorded("fast_lewm_transformer_dim", 512))
            if fast_lewm_transformer_dim is None else int(fast_lewm_transformer_dim)
        ),
        fast_lewm_transformer_depth=(
            int(_recorded("fast_lewm_transformer_depth", 6))
            if fast_lewm_transformer_depth is None else int(fast_lewm_transformer_depth)
        ),
        fast_lewm_transformer_heads=(
            int(_recorded("fast_lewm_transformer_heads", 8))
            if fast_lewm_transformer_heads is None else int(fast_lewm_transformer_heads)
        ),
        fast_lewm_transformer_mlp_dim=(
            int(_recorded("fast_lewm_transformer_mlp_dim", 2048))
            if fast_lewm_transformer_mlp_dim is None else int(fast_lewm_transformer_mlp_dim)
        ),
        fast_lewm_transformer_window_sizes=(
            tuple(_recorded("fast_lewm_transformer_window_sizes", (8, 6)))
            if fast_lewm_transformer_window_sizes is None else tuple(fast_lewm_transformer_window_sizes)
        ),
        fast_lewm_transformer_dropout=(
            float(_recorded("fast_lewm_transformer_dropout", 0.1))
            if fast_lewm_transformer_dropout is None else float(fast_lewm_transformer_dropout)
        ),
        fast_lewm_horizon_embedding_std=(
            float(_recorded("fast_lewm_horizon_embedding_std", 0.02))
            if fast_lewm_horizon_embedding_std is None else float(fast_lewm_horizon_embedding_std)
        ),
        fast_lewm_head_gradient_checkpointing=(
            bool(_recorded("fast_lewm_head_gradient_checkpointing", False))
            if fast_lewm_head_gradient_checkpointing is None
            else bool(fast_lewm_head_gradient_checkpointing)
        ),
        lambda_latent_sigreg=(
            float(_recorded("lambda_latent_sigreg", 0.0))
            if lambda_latent_sigreg is None else float(lambda_latent_sigreg)
        ),
        sigreg_num_directions=(
            int(_recorded("sigreg_num_directions", 64))
            if sigreg_num_directions is None else int(sigreg_num_directions)
        ),
        fast_lewm_normalize_target_scale=(
            bool(_recorded("fast_lewm_normalize_target_scale", False))
            if fast_lewm_normalize_target_scale is None
            else bool(fast_lewm_normalize_target_scale)
        ),
        fast_lewm_gate_bias_init=(
            float(_recorded("fast_lewm_gate_bias_init", 0.0))
            if fast_lewm_gate_bias_init is None else float(fast_lewm_gate_bias_init)
        ),
        fast_lewm_pooled_latent_dim=(
            int(_recorded("fast_lewm_pooled_latent_dim", 256))
            if fast_lewm_pooled_latent_dim is None else int(fast_lewm_pooled_latent_dim)
        ),
        fast_lewm_pooled_depth=(
            int(_recorded("fast_lewm_pooled_depth", 6))
            if fast_lewm_pooled_depth is None else int(fast_lewm_pooled_depth)
        ),
        fast_lewm_pooled_hidden_dim=(
            int(_recorded("fast_lewm_pooled_hidden_dim", 2048))
            if fast_lewm_pooled_hidden_dim is None else int(fast_lewm_pooled_hidden_dim)
        ),
        fast_lewm_pooled_fusion_dim=(
            int(_recorded("fast_lewm_pooled_fusion_dim", 768))
            if fast_lewm_pooled_fusion_dim is None else int(fast_lewm_pooled_fusion_dim)
        ),
        lambda_pooled_latent_reg=(
            float(_recorded("lambda_pooled_latent_reg", 0.0))
            if lambda_pooled_latent_reg is None else float(lambda_pooled_latent_reg)
        ),
        fast_lewm_pooled_queries=(
            int(_recorded("fast_lewm_pooled_queries", 1))
            if fast_lewm_pooled_queries is None else int(fast_lewm_pooled_queries)
        ),
        fast_lewm_per_sample_normalization=(
            bool(_recorded("fast_lewm_per_sample_normalization", False))
            if fast_lewm_per_sample_normalization is None
            else bool(fast_lewm_per_sample_normalization)
        ),
        fast_lewm_pooler_type=(
            str(_recorded("fast_lewm_pooler_type", "attention"))
            if fast_lewm_pooler_type is None else str(fast_lewm_pooler_type)
        ),
        fast_lewm_pooler_depth=(
            int(_recorded("fast_lewm_pooler_depth", 2))
            if fast_lewm_pooler_depth is None else int(fast_lewm_pooler_depth)
        ),
        fast_lewm_pooler_heads=(
            int(_recorded("fast_lewm_pooler_heads", 8))
            if fast_lewm_pooler_heads is None else int(fast_lewm_pooler_heads)
        ),
        fast_lewm_pooler_mlp_dim=(
            int(_recorded("fast_lewm_pooler_mlp_dim", 2048))
            if fast_lewm_pooler_mlp_dim is None else int(fast_lewm_pooler_mlp_dim)
        ),
        fast_lewm_directional_loss=(
            bool(_recorded("fast_lewm_directional_loss", False))
            if fast_lewm_directional_loss is None else bool(fast_lewm_directional_loss)
        ),
        lambda_query_diversity=(
            float(_recorded("lambda_query_diversity", 0.0))
            if lambda_query_diversity is None else float(lambda_query_diversity)
        ),
        lambda_pooled_variance=(
            float(_recorded("lambda_pooled_variance", 0.0))
            if lambda_pooled_variance is None else float(lambda_pooled_variance)
        ),
        lambda_pooled_covariance=(
            float(_recorded("lambda_pooled_covariance", 0.0))
            if lambda_pooled_covariance is None else float(lambda_pooled_covariance)
        ),
        lambda_action=(
            float(_recorded("lambda_action", 1.0))
            if lambda_action is None else float(lambda_action)
        ),
        fast_lewm_pooler_competitive=(
            bool(_recorded("fast_lewm_pooler_competitive", False))
            if fast_lewm_pooler_competitive is None else bool(fast_lewm_pooler_competitive)
        ),
        pooled_variance_floor=(
            float(_recorded("pooled_variance_floor", 0.7))
            if pooled_variance_floor is None else float(pooled_variance_floor)
        ),
        lambda_temporal_hinge=(
            float(_recorded("lambda_temporal_hinge", 0.0))
            if lambda_temporal_hinge is None else float(lambda_temporal_hinge)
        ),
        d_jepa=vision_backbone.embed_dim,
    )
    return model
