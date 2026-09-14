"""Fixed public JEPA-WAM training configuration."""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple, Union

from draccus import ChoiceRegistry

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


@dataclass
class VLAConfig(ChoiceRegistry):
    """The released V-JEPA 2.1 + Qwen2.5 + Flow-GR00T recipe."""

    vla_id: str = "jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-qwen25-vjepa21-vitl-384px+0_5b"

    data_mix: str = "libero_4_task_suites_no_noops"
    shuffle_buffer_size: int = 20_000

    max_steps: int = 40_000
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03

    vjepa_checkpoint_path: Optional[str] = None

    d_action: int = 7
    d_proprio: int = 8
    action_horizon: int = NUM_ACTIONS_CHUNK
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    fm_hidden_size: int = 1024
    fm_num_layers: int = 16
    fm_num_inference_timesteps: int = 4
    fm_num_timestep_buckets: int = 1_000
    fm_noise_beta_alpha: float = 1.5
    fm_noise_beta_beta: float = 1.0
    fm_noise_s: float = 0.999
    fm_num_target_vision_tokens: int = 32
    fm_add_pos_embed: bool = True
    fm_max_seq_len: int = 1_024
    fm_state_dropout: float = 0.5
    action_prediction_type: str = "velocity"

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: Union[str, Tuple[str, ...]] = "all-linear"
    visual_token_pair_offset: int = 31
    lambda_visual_token_cosine: float = 0.5
    attention_backend: str = "flash_attention_2"
    # Opt-in auxiliary head on the UNMODIFIED released causal policy path.
    enable_official_latent_head: bool = False
    lambda_official_latent: float = 0.15
    enable_five_tubelet_ar: bool = False
    enable_fast_lewm: bool = False
    enable_action_conditioned_dynamics: bool = False
    latent_patch_merge_size: int = 2
    lambda_latent_ar: float = 0.5
    lambda_absolute_latent: float = 0.02
    lambda_horizon_consistency: float = 0.02
    horizon_loss_weights: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.3)
    action_horizon_weights: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.3)
    fast_lewm_num_prefixes: int = 5
    fast_lewm_segment_targets: bool = False
    fast_lewm_prefix_dim: int = 192
    fast_lewm_prefix_depth: int = 3
    fast_lewm_prefix_heads: int = 6
    fast_lewm_prefix_dropout: float = 0.0
    fast_lewm_action_conditioning: str = "ground_truth"
    fast_lewm_action_gradient_scale: float = 0.0
    fast_lewm_query_cosine: bool = False
    fast_lewm_query_source: str = "llm_hidden"
    fast_lewm_head_type: str = "shared_mlp"
    fast_lewm_transformer_dim: int = 512
    fast_lewm_transformer_depth: int = 6
    fast_lewm_transformer_heads: int = 8
    fast_lewm_transformer_mlp_dim: int = 2048
    fast_lewm_transformer_window_sizes: Tuple[int, ...] = (8, 6)
    fast_lewm_transformer_dropout: float = 0.1
    fast_lewm_horizon_embedding_std: float = 0.02
    fast_lewm_head_gradient_checkpointing: bool = False
    fast_lewm_normalize_target_scale: bool = False
    fast_lewm_gate_bias_init: float = 0.0
    fast_lewm_pooled_latent_dim: int = 256
    fast_lewm_pooled_depth: int = 6
    fast_lewm_pooled_hidden_dim: int = 2048
    fast_lewm_pooled_fusion_dim: int = 768
    fast_lewm_pooled_queries: int = 1
    fast_lewm_per_sample_normalization: bool = False
    fast_lewm_pooler_type: str = "attention"
    fast_lewm_pooler_depth: int = 2
    fast_lewm_pooler_heads: int = 8
    fast_lewm_pooler_mlp_dim: int = 2048
    fast_lewm_directional_loss: bool = False

    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True


@dataclass
class LiberoVLAConfig(VLAConfig):
    """Concrete choice for the released LIBERO recipe."""


Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90 = LiberoVLAConfig


@dataclass
class RoboTwinVLAConfig(VLAConfig):
    """Paper Appendix B.2 recipe for RoboTwin Clean-20."""

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean"
    data_mix: str = "robotwin_paper20_clean"
    max_steps: int = 60_000
    expected_world_size: int = 16
    global_batch_size: int = 128
    per_device_batch_size: int = 8
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    d_action: int = 14
    d_proprio: int = 14
    action_horizon: int = 50
    visual_token_pair_offset: int = 50
    action_prediction_type: str = "x"
    normalization_stats_path: Optional[str] = None
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    enable_photometric_augmentation: bool = False
    photometric_augmentation_probability: float = 0.4
    photometric_augmentation_strength: float = 0.1


@dataclass
class RoboTwinFiveTubeletARConfig(RoboTwinVLAConfig):
    """384px JEPA-WAM with asymmetric five-tubelet AR/action visibility."""

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean-5tubelet-ar"
    enable_five_tubelet_ar: bool = True
    latent_patch_merge_size: int = 1
    lambda_latent_ar: float = 0.15
    lambda_absolute_latent: float = 0.02
    lambda_horizon_consistency: float = 0.02
    visual_token_pair_offset: int = 0
    attention_backend: str = "sdpa"


@dataclass
class RoboTwinFastLeWMConfig(RoboTwinVLAConfig):
    """Action-prefix-conditioned parallel five-horizon latent world head."""

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean-fast-lewm"
    enable_five_tubelet_ar: bool = False
    enable_fast_lewm: bool = True
    latent_patch_merge_size: int = 1
    lambda_latent_ar: float = 0.15
    lambda_absolute_latent: float = 0.02
    lambda_horizon_consistency: float = 0.02
    fast_lewm_num_prefixes: int = 5
    fast_lewm_segment_targets: bool = False
    fast_lewm_prefix_dim: int = 192
    fast_lewm_prefix_depth: int = 3
    fast_lewm_prefix_heads: int = 6
    fast_lewm_prefix_dropout: float = 0.0
    fast_lewm_action_conditioning: str = "predicted"
    fast_lewm_action_gradient_scale: float = 0.1
    fast_lewm_query_cosine: bool = True
    fast_lewm_query_source: str = "llm_hidden"
    visual_token_pair_offset: int = 0
    attention_backend: str = "sdpa"


@dataclass
class RoboTwinFastLeWMTransformerConfig(RoboTwinFastLeWMConfig):
    """Fresh JEPA-WAM initialization with dense prefix-conditioned dynamics attention."""

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean-fast-lewm-transformer"
    fast_lewm_head_type: str = "window_transformer"
    fast_lewm_transformer_dim: int = 512
    fast_lewm_transformer_depth: int = 6
    fast_lewm_transformer_heads: int = 8
    fast_lewm_transformer_mlp_dim: int = 2048
    fast_lewm_transformer_window_sizes: Tuple[int, ...] = (8, 6)
    fast_lewm_transformer_dropout: float = 0.1
    fast_lewm_horizon_embedding_std: float = 0.02
    fast_lewm_head_gradient_checkpointing: bool = False
    fast_lewm_normalize_target_scale: bool = False
    fast_lewm_gate_bias_init: float = 0.0
    fast_lewm_pooled_latent_dim: int = 256
    fast_lewm_pooled_depth: int = 6
    fast_lewm_pooled_hidden_dim: int = 2048
    fast_lewm_pooled_fusion_dim: int = 768
    fast_lewm_pooled_queries: int = 1
    fast_lewm_per_sample_normalization: bool = False
    fast_lewm_pooler_type: str = "attention"
    fast_lewm_pooler_depth: int = 2
    fast_lewm_pooler_heads: int = 8
    fast_lewm_pooler_mlp_dim: int = 2048
    fast_lewm_directional_loss: bool = False


@dataclass
class RoboTwinActionConditionedDynamicsConfig(RoboTwinVLAConfig):
    """Action-conditioned five-horizon dynamics fused inside Qwen's own attention.

    Feeds a causally-cumulative action-block token and a per-horizon video
    query directly into Qwen's input sequence (see
    ``build_action_conditioned_dynamics_mask``), instead of computing the
    world-model query/conditioning in a small head after Qwen (fast-lewm) or
    letting horizons see each other's future (five-tubelet-ar).
    """

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean-action-dynamics"
    enable_five_tubelet_ar: bool = False
    enable_fast_lewm: bool = False
    enable_action_conditioned_dynamics: bool = True
    latent_patch_merge_size: int = 1
    lambda_latent_ar: float = 0.15
    lambda_absolute_latent: float = 0.02
    lambda_horizon_consistency: float = 0.02
    fast_lewm_num_prefixes: int = 5
    fast_lewm_segment_targets: bool = False
    visual_token_pair_offset: int = 0
    attention_backend: str = "sdpa"


@dataclass
class LiberoFastLeWMTransformerConfig(LiberoVLAConfig):
    """The released LIBERO recipe with our patch-level latent world head bolted on.

    Deliberately inherits LiberoVLAConfig rather than the RoboTwin one: the
    released LIBERO checkpoint carries a 7-dim action decoder, an 8-dim state
    encoder and no embodiment adapters, so the RoboTwin dims (14/14, horizon 50)
    would rebuild the 533M action head instead of loading it. Every field shared
    with the released config.json already matches; only the world head is new.
    """

    vla_id: str = "jepawam-qwen25-vjepa-384px+0_5b+libero-fast-lewm-transformer"

    enable_five_tubelet_ar: bool = False
    enable_fast_lewm: bool = True
    latent_patch_merge_size: int = 1
    lambda_latent_ar: float = 0.15
    lambda_absolute_latent: float = 0.02
    lambda_horizon_consistency: float = 0.02
    fast_lewm_num_prefixes: int = 5
    fast_lewm_segment_targets: bool = False
    fast_lewm_prefix_dim: int = 192
    fast_lewm_prefix_depth: int = 3
    fast_lewm_prefix_heads: int = 6
    fast_lewm_prefix_dropout: float = 0.0
    # The released LIBERO head is a velocity-prediction flow head, so there is no
    # direct action estimate to condition on; RoboTwin's "predicted" mode needs an
    # x-prediction head.  Ground-truth conditioning keeps the released action head
    # exactly as trained -- the world loss still shapes the shared representation
    # through `fast_lewm_query_source="llm_hidden"`.
    fast_lewm_action_conditioning: str = "ground_truth"
    fast_lewm_action_gradient_scale: float = 0.1
    fast_lewm_query_cosine: bool = True
    fast_lewm_query_source: str = "llm_hidden"
    # Kept at the released recipe's value rather than the RoboTwin world-model
    # variants' 0: with no paired frames the visual-token cosine term scores
    # nothing (loss pinned near 1.0) yet still carries 0.5 weight, which is most
    # of the gradient once the action loss drops to ~0.007.  Keeping it at 31
    # also makes the world head the single difference from the released recipe.
    visual_token_pair_offset: int = 31
    # Disabled for this arm.  The released head reaches cosine ~0 under our
    # formulation even with its weights loaded and the frame pair repaired, so
    # this fork cannot reproduce their auxiliary target; at weight 0.5 it was
    # 0.49 of a 0.61 total against an action loss of 0.007, i.e. almost all of
    # the gradient, and it would wash out the checkpoint we just loaded.
    # The world head is this arm's representation objective in its place.
    lambda_visual_token_cosine: float = 0.0
    attention_backend: str = "sdpa"

    fast_lewm_head_type: str = "window_transformer"
    fast_lewm_transformer_dim: int = 512
    fast_lewm_transformer_depth: int = 6
    fast_lewm_transformer_heads: int = 8
    fast_lewm_transformer_mlp_dim: int = 2048
    fast_lewm_transformer_window_sizes: Tuple[int, ...] = (8, 6)
    fast_lewm_transformer_dropout: float = 0.1
    fast_lewm_horizon_embedding_std: float = 0.02
    fast_lewm_head_gradient_checkpointing: bool = False
    fast_lewm_normalize_target_scale: bool = False
    fast_lewm_gate_bias_init: float = 0.0
    fast_lewm_pooled_latent_dim: int = 256
    fast_lewm_pooled_depth: int = 6
    fast_lewm_pooled_hidden_dim: int = 2048
    fast_lewm_pooled_fusion_dim: int = 768
    fast_lewm_pooled_queries: int = 1
    fast_lewm_per_sample_normalization: bool = False


RoboTwinVLAConfig.register_subclass(RoboTwinVLAConfig.vla_id, RoboTwinVLAConfig)
RoboTwinFiveTubeletARConfig.register_subclass(RoboTwinFiveTubeletARConfig.vla_id, RoboTwinFiveTubeletARConfig)
RoboTwinFastLeWMConfig.register_subclass(RoboTwinFastLeWMConfig.vla_id, RoboTwinFastLeWMConfig)
RoboTwinFastLeWMTransformerConfig.register_subclass(
    RoboTwinFastLeWMTransformerConfig.vla_id,
    RoboTwinFastLeWMTransformerConfig,
)
RoboTwinActionConditionedDynamicsConfig.register_subclass(
    RoboTwinActionConditionedDynamicsConfig.vla_id,
    RoboTwinActionConditionedDynamicsConfig,
)
LiberoFastLeWMTransformerConfig.register_subclass(
    LiberoFastLeWMTransformerConfig.vla_id,
    LiberoFastLeWMTransformerConfig,
)


@unique
class VLARegistry(Enum):
    JEPAVLA_QWEN25_VJEPA_224PX_0_5B_LIBERO_90 = LiberoVLAConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_ROBOTWIN_PAPER20 = RoboTwinVLAConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_ROBOTWIN_PAPER20_5TUBELET_AR = RoboTwinFiveTubeletARConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_ROBOTWIN_PAPER20_FAST_LEWM = RoboTwinFastLeWMConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_ROBOTWIN_PAPER20_FAST_LEWM_TRANSFORMER = RoboTwinFastLeWMTransformerConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_ROBOTWIN_PAPER20_ACTION_DYNAMICS = RoboTwinActionConditionedDynamicsConfig
    JEPAWAM_QWEN25_VJEPA_384PX_0_5B_LIBERO_FAST_LEWM_TRANSFORMER = LiberoFastLeWMTransformerConfig

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


VLAConfig.register_subclass(LiberoVLAConfig.vla_id, LiberoVLAConfig)
