"""
prismatic.py

PyTorch module implementing the fixed JEPA-WAM policy.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Type, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from prismatic.models.action_heads import VisualTokenCosineHead
from prismatic.models.official_latent_head import OfficialLatentHead, validate_release_state
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.fast_lewm import (
    LatentPooler,
    PerceiverLatentPooler,
    PooledTransformerPredictor,
    PooledActionPrefixPredictor,
    pooled_latent_losses,
    ActionPrefixEncoder,
    MultiEmbodimentActionPrefixEncoder,
    PrefixConditionedWindowTransformerHead,
    SharedHorizonQueryHead,
    build_action_blocks,
    build_action_conditioned_dynamics_mask,
    build_present_action_visibility_mask,
    horizon_weighted_cosine_loss,
    scale_action_conditioning_gradient,
)
from prismatic.models.representation_regularizers import (
    gather_across_ranks,
    temporal_separation_hinge,
    sketched_isotropic_gaussian_regularization,
    variance_covariance_regularization,
    query_diversity_penalty,
    effective_rank,
)
from prismatic.training.checkpoint_migration import drop_extra_embodiments
from prismatic.models.robotwin_latent_ar import (
    LatentPatchCodec,
    build_fixed_anchor_pairs,
    build_global_action_visibility_mask,
    motion_residual_losses,
)
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import MLPProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, NUM_TOKENS, PROPRIO_DIM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Name of the robot the model is deployed on. Extra co-training robots are keyed
# by their dataset name; this one keeps the original single-robot modules.
DEFAULT_EMBODIMENT = "default"

# Prefixes FSDP and activation checkpointing insert into parameter names.
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


def _strip_wrapper_prefixes(name: str) -> str:
    """Parameter name with any wrapper segments removed, so a wrapped module and an
    unwrapped copy of it can be matched parameter for parameter."""
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def _maybe_cuda_mem_snapshot(label: str):
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "label": label,
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def encode_pairs_in_chunks(encoder, pairs: torch.Tensor, chunk: int) -> torch.Tensor:
    """Encode independent frame pairs a slice at a time.

    A ViT has no interaction between samples, so slicing the leading axis is
    bit-identical to one call -- but it caps the peak activation, which grows
    with the whole flattened batch. That batch is B*V*(1+K): at per-device 32 and
    five horizons it is 384 pairs, and the attention tensor for a 16-head ViT-L
    over 576 tokens is then 384*16*576*576 = 2.04e9 elements, within 5% of the
    int32 index limit. Halving it is the difference between running and an
    illegal memory access. Callers are inside torch.no_grad(), so nothing is
    retained across slices.
    """
    if chunk <= 0 or pairs.shape[0] <= chunk:
        return encoder.encode_pair(pairs)
    return torch.cat([encoder.encode_pair(pairs[i : i + chunk]) for i in range(0, pairs.shape[0], chunk)], dim=0)


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # The released base VLM uses the two-layer GELU projector.
        self.arch_specifier = arch_specifier
        if not arch_specifier.endswith("gelu-mlp"):
            raise ValueError(f"The public JEPA-WAM recipe requires a GELU MLP projector, got `{arch_specifier}`.")
        self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Fixed public heads: Flow-GR00T plus final-layer visual-token cosine alignment.
        self.action_placeholder_tokens = kwargs.get("flow_gr00t_placeholder_tokens", NUM_TOKENS)
        self.lambda_visual_token_cosine = kwargs.get("lambda_visual_token_cosine", 0.5)
        self.action_head = FlowMatchingActionHead(
            d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
            d_action=kwargs.get("d_action", ACTION_DIM),
            d_llm=llm_backbone.embed_dim,
            horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
            fm_hidden_size=kwargs.get("fm_hidden_size", 1024),
            fm_num_layers=kwargs.get("fm_num_layers", 16),
            fm_num_inference_timesteps=kwargs.get("fm_num_inference_timesteps", 4),
            fm_num_timestep_buckets=kwargs.get("fm_num_timestep_buckets", 1000),
            fm_noise_beta_alpha=kwargs.get("fm_noise_beta_alpha", 1.5),
            fm_noise_beta_beta=kwargs.get("fm_noise_beta_beta", 1.0),
            fm_noise_s=kwargs.get("fm_noise_s", 0.999),
            fm_num_target_vision_tokens=kwargs.get("fm_num_target_vision_tokens", 32),
            fm_add_pos_embed=kwargs.get("fm_add_pos_embed", True),
            fm_max_seq_len=kwargs.get("fm_max_seq_len", 1024),
            fm_state_dropout=kwargs.get("fm_state_dropout", 0.5),
            prediction_type=kwargs.get("action_prediction_type", "velocity"),
        )
        self.visual_token_cosine_head = VisualTokenCosineHead(
            d_llm=llm_backbone.embed_dim,
            d_target=kwargs.get("d_jepa", vision_backbone.embed_dim),
        )
        self.enable_five_tubelet_ar = bool(kwargs.get("enable_five_tubelet_ar", False))
        self.enable_fast_lewm = bool(kwargs.get("enable_fast_lewm", False))
        self.enable_action_conditioned_dynamics = bool(
            kwargs.get("enable_action_conditioned_dynamics", False)
        )
        self.fast_lewm_query_cosine = bool(kwargs.get("fast_lewm_query_cosine", False))
        if sum((self.enable_five_tubelet_ar, self.enable_fast_lewm, self.enable_action_conditioned_dynamics)) > 1:
            raise ValueError(
                "five-tubelet AR, Fast-LeWM, and action-conditioned dynamics world heads "
                "are mutually exclusive"
            )
        if self.fast_lewm_query_cosine and not self.enable_fast_lewm:
            raise ValueError("Fast-LeWM query cosine requires enable_fast_lewm=True")
        self.lambda_latent_ar = float(kwargs.get("lambda_latent_ar", 0.15))
        self.lambda_absolute_latent = float(kwargs.get("lambda_absolute_latent", 0.02))
        self.lambda_horizon_consistency = float(kwargs.get("lambda_horizon_consistency", 0.02))
        self.horizon_loss_weights = tuple(kwargs.get("horizon_loss_weights", (1.0, 0.8, 0.6, 0.4, 0.3)))
        self.action_horizon_weights = tuple(kwargs.get("action_horizon_weights", (1.0, 0.8, 0.6, 0.4, 0.3)))
        self.fast_lewm_action_conditioning = str(
            kwargs.get("fast_lewm_action_conditioning", "ground_truth")
        )
        self.fast_lewm_action_gradient_scale = float(
            kwargs.get("fast_lewm_action_gradient_scale", 0.0)
        )
        self.fast_lewm_head_type = str(kwargs.get("fast_lewm_head_type", "shared_mlp"))
        if self.fast_lewm_head_type not in {
            "shared_mlp", "window_transformer", "pooled_mlp", "pooled_transformer",
        }:
            raise ValueError(
                "Fast-LeWM head type must be one of 'shared_mlp', "
                "'window_transformer', 'pooled_mlp', 'pooled_transformer'"
            )
        if self.fast_lewm_action_conditioning not in {"ground_truth", "predicted"}:
            raise ValueError("Fast-LeWM action conditioning must be 'ground_truth' or 'predicted'")
        if self.fast_lewm_action_gradient_scale < 0:
            raise ValueError("Fast-LeWM action gradient scale must be non-negative")
        self.fast_lewm_query_source = str(kwargs.get("fast_lewm_query_source", "llm_hidden"))
        if self.fast_lewm_query_source not in {"llm_hidden", "raw_tokens"}:
            raise ValueError("Fast-LeWM query source must be 'llm_hidden' or 'raw_tokens'")
        self.latent_patch_codec = None
        self.fast_lewm_prefix_encoder = None
        self.fast_lewm_query_head = None
        self.dynamics_num_horizons = None
        if self.enable_five_tubelet_ar or self.enable_fast_lewm or self.enable_action_conditioned_dynamics:
            spatial_side = int(
                kwargs.get(
                    "latent_spatial_side",
                    int(getattr(vision_backbone, "default_image_size", 384))
                    // int(getattr(vision_backbone, "patch_size", 16)),
                )
            )
            self.latent_patch_codec = LatentPatchCodec(
                jepa_dim=kwargs.get("d_jepa", vision_backbone.embed_dim),
                llm_dim=llm_backbone.embed_dim,
                spatial_side=spatial_side,
                patch_merge_size=int(kwargs.get("latent_patch_merge_size", 2)),
            )
        # name -> (d_action, d_proprio, action_horizon) for every *extra* robot whose
        # data co-trains this model. Empty by default, which keeps the single-robot
        # modules and checkpoint keys byte-identical to before.
        self.cross_embodiments = dict(kwargs.get("cross_embodiments", {}) or {})

        # Unfreezing the visual encoder makes it produce both the world-model input
        # *and* its regression target, which the model can exploit by degrading the
        # target space instead of predicting better. An EMA copy supplies the target
        # so that shortcut is closed. 0 keeps everything frozen, which is the default
        # and leaves the module list and checkpoint keys unchanged.
        self.unfreeze_vision_blocks = int(kwargs.get("unfreeze_vision_blocks", 0))
        self.unfreeze_projector = bool(kwargs.get("unfreeze_projector", False))
        # Train the language backbone outright instead of through LoRA adapters.
        self.unfreeze_llm = bool(kwargs.get("unfreeze_llm", False))
        # Keep a trainable encoder's latents spread out and decorrelated. Zero by
        # default: a frozen encoder gets this from its pretrained weights, and the
        # terms would only add noise there.
        self.lambda_latent_variance = float(kwargs.get("lambda_latent_variance", 0.0))
        self.lambda_latent_covariance = float(kwargs.get("lambda_latent_covariance", 0.0))
        self.lambda_latent_sigreg = float(kwargs.get("lambda_latent_sigreg", 0.0))
        self.lambda_pooled_latent_reg = float(kwargs.get("lambda_pooled_latent_reg", 0.0))
        # 0.10 is where the untrained pooler starts; the failure is falling below it.
        self.fast_lewm_per_sample_normalization = bool(
            kwargs.get("fast_lewm_per_sample_normalization", False)
        )
        self.fast_lewm_directional_loss = bool(kwargs.get("fast_lewm_directional_loss", False))
        self.pooled_temporal_floor = float(kwargs.get("pooled_temporal_floor", 0.10))
        self.pooled_temporal_weight = float(kwargs.get("pooled_temporal_weight", 10.0))
        self.lambda_temporal_hinge = float(kwargs.get("lambda_temporal_hinge", 0.0))
        self.lambda_query_diversity = float(kwargs.get("lambda_query_diversity", 0.0))
        self.lambda_pooled_variance = float(kwargs.get("lambda_pooled_variance", 0.0))
        # Covariance carries its own weight because it is the term that buys rank.
        # Clearing the variance floor on every dimension is necessary and not
        # sufficient: 256 dimensions can each hold std 0.7 while being perfectly
        # correlated, which is rank one with healthy variance, and only this term
        # objects (see test_the_variance_hinge_alone_does_not_buy_rank). The
        # variance hinge had stalled at 0.45 against the world model by step 9750,
        # so adding weight there fights it head-on; this addresses the rank directly.
        self.lambda_pooled_covariance = float(kwargs.get("lambda_pooled_covariance", 0.0))
        # The action term had no weight at all, which fixed it at 1.0 and left it
        # 0.6% of a total loss the world-model terms carried: 0.0042 against
        # 0.3185 for latent AR and 0.4254 for the pooled regularizers. Note that a
        # weight scales the gradient on every module the action path touches --
        # the action head, the LoRA adapters and the prefix encoder -- so a large
        # value acts like a learning-rate multiplier on them, not just a change of
        # emphasis. Default 1.0 keeps existing runs identical.
        self.lambda_action = float(kwargs.get("lambda_action", 1.0))
        # The pooler ends in a parameter-free LayerNorm, so every pooled vector has
        # unit RMS across its D dimensions and the batch obeys
        #     sum_i E[x_i^2] = D = ||mean||^2 + sum_i var_i.
        # A per-dimension floor of 1.0 is therefore exactly the ceiling: it is met
        # only when the batch mean is exactly zero, so that hinge can never reach
        # zero and a large weight on it steers the representation instead of merely
        # braking it -- which is how lambda 1.0 made the regularizer 91% of the loss
        # before. A floor of f caps the mean at (1 - f^2) of the energy: 0.7 caps it
        # at 51%, against the 99.2% measured at step 8500, and leaves slack.
        self.pooled_variance_floor = float(kwargs.get("pooled_variance_floor", 0.7))
        self.fast_lewm_normalize_target_scale = bool(
            kwargs.get("fast_lewm_normalize_target_scale", False)
        )
        self.sigreg_num_directions = int(kwargs.get("sigreg_num_directions", 64))
        self.latent_variance_floor = float(kwargs.get("latent_variance_floor", 1.0))
        self.vision_target_momentum = float(kwargs.get("vision_target_momentum", 0.999))
        # Whether the world-model targets come from a momentum copy of the encoder.
        # With the variance/covariance terms carrying the anti-collapse duty, the EMA
        # is redundant machinery: it costs a second copy of the encoder, makes the
        # target drift with the online weights, and its state has to survive every
        # checkpoint and resume. Off means a plain stop-gradient target -- the
        # encoder's own current output, which is what VICReg-style objectives use.
        self.use_ema_target = bool(kwargs.get("use_ema_target", True))
        # Pairs per call into the frozen supervision encoder; 0 disables slicing.
        self.supervision_encode_chunk = int(
            kwargs.get("supervision_encode_chunk", os.getenv("VLA_SUPERVISION_ENCODE_CHUNK", "192"))
        )
        self.vision_target_encoder = None
        if self.enable_fast_lewm:
            action_horizon = int(kwargs.get("action_horizon", NUM_ACTIONS_CHUNK))
            action_dim = int(kwargs.get("d_action", ACTION_DIM))
            num_prefixes = int(kwargs.get("fast_lewm_num_prefixes", 5))
            if action_horizon % num_prefixes:
                raise ValueError("Fast-LeWM requires action_horizon divisible by fast_lewm_num_prefixes")
            prefix_dim = int(kwargs.get("fast_lewm_prefix_dim", 192))
            state_dim = kwargs.get("d_jepa", vision_backbone.embed_dim)
            prefix_kwargs = dict(
                prefix_dim=prefix_dim,
                depth=int(kwargs.get("fast_lewm_prefix_depth", 3)),
                num_heads=int(kwargs.get("fast_lewm_prefix_heads", 6)),
                dropout=float(kwargs.get("fast_lewm_prefix_dropout", 0.0)),
                max_prefixes=num_prefixes,
            )
            if self.cross_embodiments:
                # The causal accumulation is the method and stays shared; only the
                # input projections differ per robot. The state token is a pooled
                # V-JEPA summary, so its width is the same for every robot.
                embodiments = {DEFAULT_EMBODIMENT: (state_dim, (action_horizon // num_prefixes) * action_dim)}
                for name, (extra_action, _, extra_horizon) in self.cross_embodiments.items():
                    if extra_horizon % num_prefixes:
                        raise ValueError(
                            f"{name}: action_horizon {extra_horizon} must divide into {num_prefixes} prefixes"
                        )
                    embodiments[name] = (state_dim, (extra_horizon // num_prefixes) * extra_action)
                self.fast_lewm_prefix_encoder = MultiEmbodimentActionPrefixEncoder(
                    embodiments=embodiments, **prefix_kwargs
                )
            else:
                self.fast_lewm_prefix_encoder = ActionPrefixEncoder(
                    state_dim=state_dim,
                    action_block_dim=(action_horizon // num_prefixes) * action_dim,
                    **prefix_kwargs,
                )
            if self.latent_patch_codec.patch_merge_size != 1:
                raise ValueError("Fast-LeWM shared query dynamics requires raw latent_patch_merge_size=1")
            self.fast_lewm_latent_pooler = None
            if self.fast_lewm_head_type in {"pooled_mlp", "pooled_transformer"}:
                # Fast-LeWM predicts one latent per timestep. Pointing the same
                # recipe at a 3 x 576 x 1024 patch grid gave a target 69% of whose
                # direction a ridge probe recovered from the current frame alone,
                # so most of the supervision said nothing about the future.
                patch_dim = int(kwargs.get("d_jepa", vision_backbone.embed_dim))
                latent_dim = int(kwargs.get("fast_lewm_pooled_latent_dim", 256))
                sequence_head = self.fast_lewm_head_type == "pooled_transformer"
                num_queries = int(kwargs.get("fast_lewm_pooled_queries", 16 if sequence_head else 1))
                # The single-head pooler degenerated to a one-hot on one patch with
                # all sixteen queries picking the same one (pairwise cosine 0.99997).
                if str(kwargs.get("fast_lewm_pooler_type", "attention")) == "perceiver":
                    self.fast_lewm_latent_pooler = PerceiverLatentPooler(
                        patch_dim=patch_dim, views=3, latent_dim=latent_dim,
                        num_queries=num_queries,
                        depth=int(kwargs.get("fast_lewm_pooler_depth", 2)),
                        num_heads=int(kwargs.get("fast_lewm_pooler_heads", 8)),
                        mlp_dim=int(kwargs.get("fast_lewm_pooler_mlp_dim", 2048)),
                        competitive=bool(kwargs.get("fast_lewm_pooler_competitive", False)),
                    )
                else:
                    self.fast_lewm_latent_pooler = LatentPooler(
                        patch_dim=patch_dim, views=3, latent_dim=latent_dim,
                        num_queries=num_queries,
                    )
                if sequence_head:
                    self.fast_lewm_query_head = PooledTransformerPredictor(
                        latent_dim=latent_dim,
                        prefix_dim=prefix_dim,
                        num_queries=num_queries,
                        model_dim=int(kwargs.get("fast_lewm_transformer_dim", 768)),
                        depth=int(kwargs.get("fast_lewm_transformer_depth", 12)),
                        num_heads=int(kwargs.get("fast_lewm_transformer_heads", 12)),
                        mlp_dim=int(kwargs.get("fast_lewm_transformer_mlp_dim", 3072)),
                        num_horizons=num_prefixes,
                        dropout=float(kwargs.get("fast_lewm_transformer_dropout", 0.1)),
                    )
                else:
                    self.fast_lewm_query_head = PooledActionPrefixPredictor(
                        latent_dim=latent_dim,
                        prefix_dim=prefix_dim,
                        depth=int(kwargs.get("fast_lewm_pooled_depth", 6)),
                        hidden_dim=int(kwargs.get("fast_lewm_pooled_hidden_dim", 2048)),
                        fusion_dim=int(kwargs.get("fast_lewm_pooled_fusion_dim", 768)),
                        num_horizons=num_prefixes,
                        dropout=float(kwargs.get("fast_lewm_transformer_dropout", 0.1)),
                    )
            elif self.fast_lewm_head_type == "window_transformer":
                self.fast_lewm_query_head = PrefixConditionedWindowTransformerHead(
                    query_dim=llm_backbone.embed_dim,
                    target_dim=kwargs.get("d_jepa", vision_backbone.embed_dim),
                    prefix_dim=prefix_dim,
                    model_dim=int(kwargs.get("fast_lewm_transformer_dim", 512)),
                    depth=int(kwargs.get("fast_lewm_transformer_depth", 6)),
                    num_heads=int(kwargs.get("fast_lewm_transformer_heads", 8)),
                    mlp_dim=int(kwargs.get("fast_lewm_transformer_mlp_dim", 2048)),
                    window_sizes=tuple(kwargs.get("fast_lewm_transformer_window_sizes", (8, 6))),
                    num_horizons=num_prefixes,
                    max_views=3,
                    dropout=float(kwargs.get("fast_lewm_transformer_dropout", 0.1)),
                    horizon_embedding_std=float(
                        kwargs.get("fast_lewm_horizon_embedding_std", 0.02)
                    ),
                    gradient_checkpointing=bool(
                        kwargs.get("fast_lewm_head_gradient_checkpointing", False)
                    ),
                    gate_bias_init=float(kwargs.get("fast_lewm_gate_bias_init", 0.0)),
                )
            else:
                self.fast_lewm_query_head = SharedHorizonQueryHead(
                    query_dim=llm_backbone.embed_dim,
                    target_dim=kwargs.get("d_jepa", vision_backbone.embed_dim),
                    prefix_dim=prefix_dim,
                    num_horizons=num_prefixes,
                )
            self.fast_lewm_num_prefixes = num_prefixes
            self.fast_lewm_segment_targets = bool(
                kwargs.get("fast_lewm_segment_targets", False)
            )
            for name, (extra_action, extra_proprio, extra_horizon) in self.cross_embodiments.items():
                self.action_head.register_embodiment(
                    name, d_action=extra_action, d_proprio=extra_proprio, action_horizon=extra_horizon
                )

        if self.enable_action_conditioned_dynamics:
            if self.latent_patch_codec.patch_merge_size != 1:
                raise ValueError("action-conditioned dynamics requires raw latent_patch_merge_size=1")
            self.dynamics_num_horizons = int(kwargs.get("fast_lewm_num_prefixes", 5))

        if self.unfreeze_vision_blocks and self.use_ema_target:
            import copy

            self.vision_target_encoder = copy.deepcopy(self.vision_backbone)
            self.vision_target_encoder.requires_grad_(False)
            self.vision_target_encoder.eval()

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = [
            "vision_backbone",
            "llm_backbone",
            "projector",
            "action_head",
            "visual_token_cosine_head",
        ]
        self.enable_official_latent_head = bool(kwargs.get("enable_official_latent_head", False))
        self.lambda_official_latent = float(kwargs.get("lambda_official_latent", 0.15))
        self.official_latent_head = None
        if self.enable_official_latent_head:
            if self.enable_fast_lewm or self.enable_five_tubelet_ar or self.enable_action_conditioned_dynamics:
                raise ValueError("Official auxiliary head cannot change the released policy forward path")
            if self.unfreeze_llm or self.unfreeze_projector or self.unfreeze_vision_blocks:
                raise ValueError("Official auxiliary recipe requires frozen base Qwen, projector and vision")
            # Keep the base policy's RNG sequence unchanged when attaching the new module.
            with torch.random.fork_rng(devices=[]):
                self.official_latent_head = OfficialLatentHead(llm_backbone.embed_dim,
                                                             vision_backbone.embed_dim)
            self.all_module_keys.append("official_latent_head")
        if self.latent_patch_codec is not None:
            self.all_module_keys.append("latent_patch_codec")
        if self.fast_lewm_prefix_encoder is not None:
            self.all_module_keys.append("fast_lewm_prefix_encoder")
        if self.fast_lewm_query_head is not None:
            self.all_module_keys.append("fast_lewm_query_head")
        if getattr(self, "fast_lewm_latent_pooler", None) is not None:
            self.all_module_keys.append("fast_lewm_latent_pooler")
        self.trainable_module_keys = []
        overwatch.info("Initialized Flow-GR00T action head with %d placeholder tokens", self.action_placeholder_tokens)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        load_visual_token_cosine_head: bool = True,
        reinitialize_world_head: bool = False,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if vlm.enable_official_latent_head:
            validate_release_state(vlm, model_state_dict)
            if "official_latent_head" in model_state_dict:
                vlm.official_latent_head.load_state_dict(model_state_dict["official_latent_head"], strict=True)
            else:
                overwatch.info("Official release restore: ONLY official_latent_head is newly initialized")
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])

        llm_state = model_state_dict["llm_backbone"]
        expected = set(vlm.llm_backbone.state_dict())
        if expected and not (expected & set(llm_state)):
            # No key in common: the checkpoint trained Qwen outright and this model
            # wraps it in LoRA (or the reverse). The base weights are still the right
            # starting point, so re-address them instead of refusing to load.
            from prismatic.training.checkpoint_migration import adapt_full_parameter_llm_to_lora

            llm_state, mapped, adapters = adapt_full_parameter_llm_to_lora(llm_state, expected)
            overwatch.info(
                "Adapting a full-parameter Qwen checkpoint onto a LoRA-wrapped model: "
                "%d base tensors re-addressed, %d adapter tensors left at their init",
                mapped,
                adapters,
            )
            vlm.llm_backbone.load_state_dict(llm_state, strict=False)
        else:
            vlm.llm_backbone.load_state_dict(llm_state)

        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        if "action_head" in model_state_dict:
            head_state, dropped = drop_extra_embodiments(
                model_state_dict["action_head"], vlm.action_head.state_dict()
            )
            if dropped:
                overwatch.info(
                    "Action head: dropped %d tensors belonging to robots this model does not carry",
                    len(dropped),
                )
            vlm.action_head.load_state_dict(head_state, strict=not dropped)
        if "visual_token_cosine_head" in model_state_dict:
            if load_visual_token_cosine_head:
                missing, unexpected = vlm.visual_token_cosine_head.load_state_dict(
                    model_state_dict["visual_token_cosine_head"],
                    strict=False,
                )
                if missing or unexpected:
                    overwatch.info(
                        "Visual Token Cosine Head checkpoint mismatch ignored "
                        "(missing=%s unexpected=%s)",
                        missing,
                        unexpected,
                    )
            else:
                overwatch.info("Skipping visual_token_cosine_head checkpoint load by request.")
        if vlm.latent_patch_codec is not None and "latent_patch_codec" in model_state_dict:
            vlm.latent_patch_codec.load_state_dict(model_state_dict["latent_patch_codec"])
        if (
            getattr(vlm, "fast_lewm_latent_pooler", None) is not None
            and "fast_lewm_latent_pooler" in model_state_dict
        ):
            from prismatic.training.checkpoint_migration import expand_pooler_queries

            saved_pooler = model_state_dict["fast_lewm_latent_pooler"]
            expected_pooler = vlm.fast_lewm_latent_pooler.state_dict()
            if set(saved_pooler) != set(expected_pooler):
                # A different pooler entirely, not a resized one. Everything else in
                # the checkpoint still fits and is worth keeping -- the encoder, Qwen,
                # the action head, the prefix encoder and, when its shape is
                # unchanged, the predictor itself.
                if not reinitialize_world_head:
                    raise ValueError(
                        "The checkpoint's latent pooler has a different architecture "
                        f"({len(saved_pooler)} tensors against {len(expected_pooler)}). Pass "
                        "reinitialize_world_head=True to rebuild it on purpose."
                    )
                overwatch.info(
                    "Latent pooler rebuilt: the checkpoint's has a different architecture "
                    "(%d tensors against %d); every other module still resumes",
                    len(saved_pooler), len(expected_pooler),
                )
            else:
                pooler_state, rebuilt = expand_pooler_queries(saved_pooler, expected_pooler)
                if rebuilt:
                    overwatch.info(
                        "Latent pooler: %s reshaped for the new query count; its projections "
                        "were carried over unchanged",
                        ", ".join(rebuilt),
                    )
                vlm.fast_lewm_latent_pooler.load_state_dict(pooler_state)
        if vlm.fast_lewm_prefix_encoder is not None:
            if "fast_lewm_prefix_encoder" in model_state_dict:
                prefix_state, dropped = drop_extra_embodiments(
                    model_state_dict["fast_lewm_prefix_encoder"],
                    vlm.fast_lewm_prefix_encoder.state_dict(),
                )
                if dropped:
                    overwatch.info(
                        "Prefix encoder: dropped %d tensors belonging to robots this model does not carry",
                        len(dropped),
                    )
                vlm.fast_lewm_prefix_encoder.load_state_dict(prefix_state, strict=not dropped)
            if vlm.fast_lewm_query_head is not None and "fast_lewm_query_head" in model_state_dict:
                head_state = model_state_dict["fast_lewm_query_head"]
                expected_head = vlm.fast_lewm_query_head.state_dict()
                # Missing and unexpected keys count as a mismatch, not just a
                # changed shape: swapping the head's architecture outright leaves
                # the two key sets almost disjoint, and a shapes-only check would
                # find nothing to complain about and then fail inside a strict load.
                mismatched = [
                    key
                    for key, tensor in expected_head.items()
                    if key in head_state and head_state[key].shape != tensor.shape
                ]
                mismatched += sorted(set(expected_head) ^ set(head_state))
                if mismatched and reinitialize_world_head:
                    # Resizing the world head is a deliberate architecture change:
                    # every other module still resumes, but the head itself has to
                    # start from its own initialization.
                    overwatch.info(
                        "World head resized (%d tensors differ in shape); reinitializing it "
                        "and resuming every other module from the checkpoint",
                        len(mismatched),
                    )
                elif mismatched:
                    raise ValueError(
                        "World-head checkpoint shapes do not match this model "
                        f"({len(mismatched)} tensors, e.g. {mismatched[0]}). Pass "
                        "reinitialize_world_head=True to change the head's size on purpose."
                    )
                else:
                    vlm.fast_lewm_query_head.load_state_dict(head_state)

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_for_training(self) -> None:
        """Freeze the fixed base model and train only Qwen LoRA plus the two public heads."""
        self.vision_backbone.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.llm_backbone.requires_grad_(False)

        lora_param_names = []
        if self.unfreeze_llm:
            # Full-parameter language training: no adapters, the backbone itself moves.
            self.llm_backbone.requires_grad_(True)
        else:
            for name, param in self.llm_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    lora_param_names.append(name)
            if not lora_param_names:
                raise RuntimeError("Qwen must be wrapped with LoRA before calling `freeze_for_training`.")

        self.action_head.requires_grad_(True)
        if self.enable_five_tubelet_ar:
            if self.latent_patch_codec is None:
                raise RuntimeError("five-tubelet AR is enabled without a latent patch codec")
            self.visual_token_cosine_head.requires_grad_(False)
            self.latent_patch_codec.requires_grad_(True)
            self.trainable_module_keys = ["llm_backbone", "action_head", "latent_patch_codec"]
        elif self.enable_fast_lewm:
            if self.fast_lewm_prefix_encoder is None or self.fast_lewm_query_head is None:
                raise RuntimeError("Fast-LeWM is enabled without all required world-head modules")
            self.visual_token_cosine_head.requires_grad_(False)
            # Fast-LeWM feeds Qwen through the pretrained frozen projector, so the
            # codec sits on no forward path here; keeping it trainable would leave
            # FSDP with parameters that require grad but never receive one.
            if self.latent_patch_codec is not None:
                self.latent_patch_codec.requires_grad_(False)
            self.fast_lewm_prefix_encoder.requires_grad_(True)
            self.fast_lewm_query_head.requires_grad_(True)
            if getattr(self, "fast_lewm_latent_pooler", None) is not None:
                self.fast_lewm_latent_pooler.requires_grad_(True)
            self.trainable_module_keys = [
                "llm_backbone",
                "action_head",
                "fast_lewm_prefix_encoder",
                "fast_lewm_query_head",
                *(("fast_lewm_latent_pooler",)
                  if getattr(self, "fast_lewm_latent_pooler", None) is not None else ()),
            ]
        elif self.enable_action_conditioned_dynamics:
            if self.latent_patch_codec is None:
                raise RuntimeError("action-conditioned dynamics is enabled without its required modules")
            self.visual_token_cosine_head.requires_grad_(False)
            self.latent_patch_codec.requires_grad_(True)
            self.trainable_module_keys = [
                "llm_backbone",
                "action_head",
                "latent_patch_codec",
            ]
        else:
            self.visual_token_cosine_head.requires_grad_(True)
            self.trainable_module_keys = ["llm_backbone", "action_head", "visual_token_cosine_head"]
        if getattr(self, "official_latent_head", None) is not None:
            self.official_latent_head.requires_grad_(True)
            self.trainable_module_keys.append("official_latent_head")
        self.vision_backbone_requires_grad = False
        if self.unfreeze_vision_blocks:
            # Only the last N transformer blocks: the early layers hold the generic
            # low-level features that make the frozen encoder survive a domain shift,
            # and they are the most expensive to keep activations for.
            featurizer = getattr(self.vision_backbone, "featurizer", None)
            blocks = getattr(featurizer, "blocks", None)
            if blocks is None:
                raise RuntimeError("this vision backbone does not expose `featurizer.blocks` to unfreeze")
            if self.unfreeze_vision_blocks < 0:
                # Everything, patch embedding and modality embeddings included -- not
                # just the transformer blocks, which would leave the encoder's input
                # stage fixed to what the pretraining data looked like.
                featurizer.requires_grad_(True)
                selected = list(blocks)
            else:
                selected = list(blocks)[-self.unfreeze_vision_blocks :]
                for block in selected:
                    block.requires_grad_(True)
            # The featurizer is cast to bf16 on construction, which is fine only while
            # it is frozen: once it produces gradients they are bf16 among everyone
            # else's fp32, and gradient clipping refuses a mixed-dtype set. Upcasting
            # is lossless, and FSDP's mixed-precision policy still runs compute in bf16.
            self.vision_backbone.featurizer.float()
            if self.vision_target_encoder is not None:
                self.vision_target_encoder.float()
            self.vision_backbone.featurizer.train()
            self.vision_backbone_requires_grad = True
            self.trainable_module_keys = list(self.trainable_module_keys) + ["vision_backbone"]
            if self.vision_target_encoder is not None:
                # Persisted even though it takes no gradient. Rebuilt from the online
                # encoder on resume it would restart with zero lag, and a target equal
                # to its own input is exactly the condition the EMA exists to avoid --
                # for the ~1/(1-momentum) steps it takes to separate again.
                self.trainable_module_keys = list(self.trainable_module_keys) + ["vision_target_encoder"]
            overwatch.info(
                "[TRAINABLE] =>> Vision Backbone: %s (%s)",
                "entire featurizer" if self.unfreeze_vision_blocks < 0
                else f"last {len(selected)} of {len(list(blocks))} blocks",
                # Say which target the world model is actually supervised against;
                # printing a momentum unconditionally reads as an EMA even when the
                # run disabled it.
                f"EMA target, momentum={self.vision_target_momentum:.4f}"
                if self.vision_target_encoder is not None
                else "no EMA: targets are the encoder's own output, stop-gradient",
                ctx_level=1,
            )
        if self.unfreeze_projector:
            # The projector is the vision->language interface the encoder drifts under;
            # leaving it frozen while the encoder moves breaks the alignment it encodes.
            self.projector.requires_grad_(True)
            self.trainable_module_keys = list(self.trainable_module_keys) + ["projector"]
            overwatch.info("[TRAINABLE] =>> Projector `%s`", self.arch_specifier, ctx_level=1)

        if not self.vision_backbone_requires_grad:
            overwatch.info(f"[Frozen] =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
        if not self.unfreeze_projector:
            overwatch.info(f"[Frozen] =>> Projector `{self.arch_specifier}`", ctx_level=1)
        if self.unfreeze_llm:
            overwatch.info("[TRAINABLE] =>> Qwen (full parameters, no LoRA)", ctx_level=1)
        else:
            overwatch.info(
                f"[TRAINABLE] =>> Qwen LoRA (`{len(lora_param_names)}` parameter groups matched)",
                ctx_level=1,
            )
        overwatch.info(
            f"[TRAINABLE] =>> Flow-GR00T Action Head (placeholders={self.action_placeholder_tokens})",
            ctx_level=1,
        )
        if self.enable_five_tubelet_ar:
            overwatch.info(
                "[TRAINABLE] =>> raw 24x24 latent token projection + reconstruction head",
                ctx_level=1,
            )
        elif self.enable_fast_lewm:
            overwatch.info(
                "[TRAINABLE] =>> unified action-conditioned Qwen query dynamics%s",
                " + shared horizon query cosine" if self.fast_lewm_query_cosine else "",
                ctx_level=1,
            )
        elif self.enable_action_conditioned_dynamics:
            overwatch.info(
                "[TRAINABLE] =>> in-Qwen action-conditioned horizon dynamics "
                "(%d horizons, action-gen sub-group visibility)",
                self.dynamics_num_horizons,
                ctx_level=1,
            )
        else:
            overwatch.info("[TRAINABLE] =>> Visual Token Cosine Head", ctx_level=1)

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> projector and fixed action/alignment heads
        head_classes = {
            MLPProjector,
            FlowMatchingActionHead,
            VisualTokenCosineHead,
            OfficialLatentHead,
            LatentPatchCodec,
            ActionPrefixEncoder,
            PrefixConditionedWindowTransformerHead,
            SharedHorizonQueryHead,
        }

        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes=head_classes,
        )

        # Return union (_or_) over constituent policies
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    @staticmethod
    def _select_action_memory(
        llm_hidden: torch.Tensor,
        fused_attention_mask: torch.Tensor,
        num_action_tokens: int,
    ) -> torch.Tensor:
        """Use the final action-placeholder span from the padded sequence."""
        if llm_hidden.shape[1] < num_action_tokens:
            raise ValueError("Input sequence is shorter than the configured action placeholder span.")
        return llm_hidden[:, -num_action_tokens:, :]

    def _split_text_and_action_embeddings(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_lengths = attention_mask.long().sum(dim=1)
        text_lengths = valid_lengths - self.action_placeholder_tokens
        if torch.any(text_lengths < 1):
            raise ValueError("Every sample must contain text before the action placeholders.")
        max_text = int(text_lengths.max().item())
        text = input_embeddings.new_zeros(input_embeddings.shape[0], max_text, input_embeddings.shape[-1])
        text_mask = torch.zeros(
            input_embeddings.shape[0], max_text, dtype=attention_mask.dtype, device=attention_mask.device
        )
        action = input_embeddings.new_empty(
            input_embeddings.shape[0], self.action_placeholder_tokens, input_embeddings.shape[-1]
        )
        for batch_index, text_length_tensor in enumerate(text_lengths):
            text_length = int(text_length_tensor.item())
            text[batch_index, :text_length] = input_embeddings[batch_index, :text_length]
            text_mask[batch_index, :text_length] = 1
            action[batch_index] = input_embeddings[
                batch_index, text_length : text_length + self.action_placeholder_tokens
            ]
        return text, text_mask, action

    @property
    def _target_vision_encoder(self):
        """Whichever encoder supplies world-model targets: the EMA copy when the
        online encoder is training, otherwise the (frozen) online encoder itself."""
        return self.vision_target_encoder if self.vision_target_encoder is not None else self.vision_backbone

    @torch.no_grad()
    def update_vision_target_encoder(self) -> None:
        """Momentum-update the target encoder. Call once per optimizer step.

        Pairing is by name, not by iteration order: FSDP flattens and shards the
        online encoder and its EMA copy independently, so positional zip lines up
        tensors that merely happen to sit at the same index -- a bias against a
        positional embedding, say -- and the in-place update then changes a
        parameter's shape.
        """
        if self.vision_target_encoder is None:
            return
        momentum = self.vision_target_momentum
        # The online encoder is wrapped by FSDP and its parameter names carry the
        # wrapper's prefixes; the target is excluded from the wrap and keeps clean
        # ones. Strip the wrappers so the two address the same parameter.
        online_params = {
            _strip_wrapper_prefixes(name): parameter
            for name, parameter in self.vision_backbone.named_parameters()
        }
        for raw_name, target in self.vision_target_encoder.named_parameters():
            name = _strip_wrapper_prefixes(raw_name)
            online = online_params.get(name)
            if online is None:
                raise RuntimeError(f"EMA target encoder has no counterpart for `{name}`")
            if online.shape != target.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for `{name}`: online {tuple(online.shape)} "
                    f"vs target {tuple(target.shape)}"
                )
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        online_buffers = {
            _strip_wrapper_prefixes(name): buffer
            for name, buffer in self.vision_backbone.named_buffers()
        }
        for raw_name, target in self.vision_target_encoder.named_buffers():
            online = online_buffers.get(_strip_wrapper_prefixes(raw_name))
            if online is not None and online.shape == target.shape:
                target.copy_(online)

    def _resolve_embodiment(self, dataset_names: Optional[list]) -> Optional[str]:
        """Which robot this batch came from, or None for the deployment robot.

        Streams are alternated rather than interleaved, so a batch is homogeneous;
        a mixed batch would silently route every sample through one robot's action
        space, so reject it instead.
        """
        if not self.cross_embodiments or not dataset_names:
            return None
        unique = set(dataset_names)
        if len(unique) > 1:
            raise ValueError(f"a batch must come from a single embodiment, got {sorted(unique)}")
        name = next(iter(unique))
        return name if name in self.cross_embodiments else None

    def present_action_context(
        self,
        current_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Run Qwen over the present-only sequence the world-head recipe trains on.

        Inference and training have to build this identically or the action head
        sees an input it was never fit on: the visual prefix is every current
        camera frame through the frozen projector, and the action placeholders
        read each other bidirectionally through a 4D mask rather than causally.
        Keeping one implementation is what stops `predict_action` and
        `_forward_fast_lewm` from drifting apart. The world branch is training
        only and is simply not called at inference.
        """
        current_tokens = self.projector(current_features)
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        text, text_mask, action_tokens = self._split_text_and_action_embeddings(
            input_embeddings, attention_mask
        )
        prefix = torch.cat((text[:, :1], current_tokens, text[:, 1:]), dim=1)
        current_mask = torch.ones(
            current_tokens.shape[:2], dtype=text_mask.dtype, device=text_mask.device
        )
        prefix_mask = torch.cat((text_mask[:, :1], current_mask, text_mask[:, 1:]), dim=1)
        joint_inputs = torch.cat((prefix, action_tokens), dim=1)
        joint_mask = build_present_action_visibility_mask(
            prefix_mask,
            action_tokens=action_tokens.shape[1],
            dtype=joint_inputs.dtype,
        )
        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=joint_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=joint_inputs,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        hidden = llm_output.hidden_states[-1]
        action_hidden = hidden[:, -self.action_placeholder_tokens :]
        return action_hidden, hidden, current_tokens

    def _forward_fast_lewm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        current_frame_pairs: torch.Tensor,
        future_frames: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        embodiment: Optional[str] = None,
    ) -> dict:
        """Predict five action-conditioned future latents in parallel from one anchor."""
        if (
            self.latent_patch_codec is None
            or self.fast_lewm_prefix_encoder is None
            or self.fast_lewm_query_head is None
        ):
            raise RuntimeError("Fast-LeWM requires its codec, prefix encoder, and shared query head")
        if current_frame_pairs.ndim != 6 or current_frame_pairs.shape[2] != 2:
            raise ValueError("current_frame_pairs must have shape [B,V,2,3,H,W]")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
            raise ValueError("Fast-LeWM pixel_values must have shape [B,V,C,H,W]")
        if pixel_values.shape[:2] != current_frame_pairs.shape[:2]:
            raise ValueError("Fast-LeWM pixel_values must contain the same batch and views")
        if future_frames.ndim != 6 or future_frames.shape[2] != 5:
            raise ValueError("future_frames must have shape [B,V,5,3,H,W]")
        if future_frames.shape[1] != current_frame_pairs.shape[1]:
            raise ValueError("current and future camera counts must match")
        if actions is None or proprio is None:
            raise ValueError("Fast-LeWM training requires actions and proprio")
        if actions.ndim != 3:
            raise ValueError("Fast-LeWM actions must have shape [B,T,D]")

        # The input path may carry gradients (when the encoder is being fine-tuned);
        # the supervision path never does, and reads the EMA copy when one exists.
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            current_features = self.vision_backbone(pixel_values)
        batch, views = current_frame_pairs.shape[:2]
        spatial_side = self.latent_patch_codec.spatial_side
        expected_tokens = views * spatial_side * spatial_side
        if current_features.shape[1:] != (expected_tokens, self.latent_patch_codec.jepa_dim):
            raise ValueError(
                "Fast-LeWM single-image V-JEPA output does not match the configured view grid"
            )
        current_grid = current_features.reshape(
            batch,
            views,
            1,
            spatial_side,
            spatial_side,
            current_features.shape[-1],
        )
        with torch.no_grad():
            fixed_anchor_pairs = build_fixed_anchor_pairs(
                current_frame_pairs,
                future_frames,
                segment_targets=getattr(self, "fast_lewm_segment_targets", False),
            )
            batch, views, horizons, pair_frames, channels, height, width = fixed_anchor_pairs.shape
            flat_pairs = fixed_anchor_pairs.reshape(
                batch * views * horizons, pair_frames, channels, height, width
            )
            anchor = current_frame_pairs[:, :, -1]
            static_pairs = torch.stack((anchor, anchor), dim=2).reshape(
                batch * views, pair_frames, channels, height, width
            )
            flat_supervision_grid = encode_pairs_in_chunks(
                self._target_vision_encoder,
                torch.cat((static_pairs, flat_pairs), dim=0),
                self.supervision_encode_chunk,
            )
            if flat_supervision_grid.shape[1:3] != (1, 1):
                raise ValueError(
                    "independent fixed-anchor encoding must return one view and one temporal token"
                )
            static_grid = flat_supervision_grid[: batch * views].reshape(
                batch, views, 1, *flat_supervision_grid.shape[3:]
            )
            future_grid = flat_supervision_grid[batch * views :, 0, 0].reshape(
                batch,
                views,
                horizons,
                *flat_supervision_grid.shape[3:],
            )

        # Present-only action policy path: no predicted future latent is visible here.
        #
        # The visual tokens entering Qwen come from the *pretrained, frozen* projector,
        # exactly as in the plain VLA path and in the published JEPA-WAM recipe (which
        # only ever adds an output-side head and leaves the input pathway untouched).
        # Routing them through `latent_patch_codec` instead would replace that
        # web-pretrained vision->language interface with a randomly-initialised linear
        # layer fit on clean-only RoboTwin renders -- it fits in-domain (low training
        # loss) but has no reason to survive the clean->randomized shift, and no
        # checkpoint can initialise it because plain-path runs do not contain it.
        # `current_features` above is already the frozen V-JEPA encoding of the
        # three current camera frames, so this reuses it rather than paying for a
        # second ViT forward.
        action_hidden, hidden, current_tokens = self.present_action_context(
            current_features, input_ids, attention_mask
        )
        if actions.shape[1] % len(self.action_horizon_weights):
            raise ValueError("action horizon must be divisible by action_horizon_weights")
        action_step_weights = torch.as_tensor(
            self.action_horizon_weights, dtype=actions.dtype, device=actions.device
        ).repeat_interleave(actions.shape[1] // len(self.action_horizon_weights))
        # Only forward the embodiment when extra robots are configured, so a custom
        # or stubbed action head that predates cross-embodiment support still works.
        action_head_kwargs = {"embodiment": embodiment} if self.cross_embodiments else {}
        loss_action, action_prediction = self.action_head(
            action_hidden,
            proprio,
            actions,
            action_step_weights=action_step_weights,
            **action_head_kwargs,
        )

        # Fast-LeWM world path: causal action-prefix encoding followed by one parallel call.
        anchor_grid = current_grid[:, :, 0]
        state_token = anchor_grid.mean(dim=(1, 2, 3))
        if self.fast_lewm_action_conditioning == "predicted":
            prediction_type = getattr(getattr(self.action_head, "config", None), "prediction_type", "x")
            if prediction_type != "x":
                raise RuntimeError(
                    "predicted-action Fast-LeWM conditioning requires the RoboTwin x-pred action head"
                )
            conditioning_actions = scale_action_conditioning_gradient(
                action_prediction,
                gradient_scale=self.fast_lewm_action_gradient_scale,
            )
        else:
            conditioning_actions = actions
        action_blocks = build_action_blocks(
            conditioning_actions,
            num_prefixes=self.fast_lewm_num_prefixes,
        )
        if self.cross_embodiments:
            action_prefixes = self.fast_lewm_prefix_encoder(
                embodiment or DEFAULT_EMBODIMENT, state_token, action_blocks
            )
        else:
            action_prefixes = self.fast_lewm_prefix_encoder(state_token, action_blocks)

        if self.fast_lewm_latent_pooler is not None:
            # Fast-LeWM's own formulation: one latent per timestep, predicted with
            # a plain squared error against the raw future latent. No residual
            # against a static anchor, no per-patch spatial weighting -- both were
            # ours, and the probe showed the patch-level residual was 69%
            # recoverable from the current frame without seeing the future.
            batch_size, views = future_grid.shape[0], future_grid.shape[1]
            patches = future_grid.shape[3] * future_grid.shape[4]
            patch_dim = future_grid.shape[-1]
            current_latent = self.fast_lewm_latent_pooler(
                static_grid[:, :, 0].reshape(batch_size, views, patches, patch_dim)
            )
            # The encoder side stays under no_grad above; the pooler is run with
            # gradient here so the anti-collapse term below can act on it. The
            # regression target is the detached copy -- the pooler must not be able
            # to lower the loss by moving the target it is scored against.
            future_latents = torch.stack(
                [
                    self.fast_lewm_latent_pooler(
                        future_grid[:, :, horizon].reshape(
                            batch_size, views, patches, patch_dim
                        )
                    )
                    for horizon in range(future_grid.shape[2])
                ],
                dim=1,
            )
            target_latents = future_latents.detach()
            # Current and future latents together, because the failure to prevent
            # is temporal: a unit-norm latent cannot shrink, but the pooler can
            # still send every frame of a clip to the same point on the sphere, and
            # it did -- the target's motion RMS fell 0.105 -> 0.015 over 750 steps
            # while each latent kept unit norm. Constraining one timestep's
            # distribution cannot see that; constraining the stack of six can.
            pooled_for_regularization = torch.cat(
                (current_latent[:, None], future_latents), dim=1
            ).reshape(-1, current_latent.shape[-1])
            latent_prediction = self.fast_lewm_query_head(current_latent, action_prefixes)
            pooled_losses = pooled_latent_losses(
                latent_prediction, target_latents, current_latent,
                per_sample_normalization=self.fast_lewm_per_sample_normalization,
                directional=self.fast_lewm_directional_loss,
                live_target=future_latents,
            )
            zero = latent_prediction.sum() * 0.0
            motion_losses = {
                "motion": pooled_losses["latent_mse"],
                "absolute": zero,
                "consistency": zero,
                "target_motion_rms": pooled_losses["target_motion_rms"],
                "residual_cosine": pooled_losses["residual_cosine"],
                "motion_agreement": pooled_losses["motion_agreement"],
                "centred_agreement": pooled_losses["centred_agreement"],
            }
            loss_visual_token_cosine = None
        else:
            # The shared visual grid is the sole forward-dynamics state.  Five
            # action prefixes modulate it into five fixed-horizon V-JEPA targets.
            # `fast_lewm_query_source` picks where that grid comes from:
            #   "llm_hidden" (default) -- Qwen's own output at the visual-token
            #     positions, i.e. current_tokens after a full LLM forward pass.
            #     World-model loss gradients flow back through the Qwen backbone.
            #   "raw_tokens" -- current_tokens itself (the projected visual tokens,
            #     pre-Qwen). Decouples the world-model loss from Qwen's weights,
            #     at the cost of losing whatever language/instruction grounding
            #     the LLM forward pass would have added to the query.
            visual_token_count = current_tokens.shape[1]
            side = int(self.vision_backbone.default_image_size) // int(self.vision_backbone.patch_size)
            expected_tokens = current_frame_pairs.shape[1] * side * side
            if self.fast_lewm_query_source == "raw_tokens":
                query_source = current_tokens
            else:
                query_source = hidden[:, 1 : 1 + visual_token_count]
            if query_source.shape[1] != expected_tokens:
                raise ValueError(
                    f"Fast-LeWM shared query token mismatch: got {query_source.shape[1]}, expected {expected_tokens}"
                )
            shared_query = query_source.reshape(
                query_source.shape[0],
                current_frame_pairs.shape[1],
                side,
                side,
                query_source.shape[-1],
            )
            latent_prediction = self.fast_lewm_query_head(shared_query, action_prefixes)
            motion_losses = motion_residual_losses(
                latent_prediction,
                future_grid,
                static_grid,
                horizon_weights=self.horizon_loss_weights,
                normalize_target_scale=self.fast_lewm_normalize_target_scale,
            )
            loss_visual_token_cosine = None
            if self.fast_lewm_query_cosine:
                loss_visual_token_cosine = horizon_weighted_cosine_loss(
                    latent_prediction,
                    future_grid,
                    horizon_weights=self.horizon_loss_weights,
                )
        total_loss = (
            self.lambda_action * loss_action
            + self.lambda_latent_ar * motion_losses["motion"]
            + self.lambda_absolute_latent * motion_losses["absolute"]
            + self.lambda_horizon_consistency * motion_losses["consistency"]
        )
        if loss_visual_token_cosine is not None:
            total_loss = total_loss + self.lambda_visual_token_cosine * loss_visual_token_cosine
        # Applied to the encoder's own output, which is where the narrowing shows --
        # not to the world head's prediction, which can stay varied while the space
        # it predicts into quietly shrinks.
        loss_pooled_reg = None
        if self.fast_lewm_latent_pooler is not None and self.lambda_pooled_latent_reg:
            pooled_rows = gather_across_ranks(pooled_for_regularization)
            pooled_sigreg = sketched_isotropic_gaussian_regularization(
                pooled_rows, num_directions=self.sigreg_num_directions
            )
            pooled_variance, pooled_covariance = variance_covariance_regularization(
                pooled_rows, variance_floor=self.pooled_variance_floor
            )
            # The distributional terms above cover the collapse modes a marginal
            # can see; the hinge covers the mode it cannot.
            #
            # These two are weighted separately because they behave differently. The
            # distributional terms are always non-zero, so their weight steers the
            # representation: at 1.0 they were 91% of the total loss and the world
            # model 7%, and the latent was shaped to be isotropic rather than
            # predictable. The hinge is exactly zero while the latent keeps its
            # horizons apart, so a large weight on it never steers anything -- it
            # only brakes. Sharing one weight forced a choice between a regularizer
            # that dominates and one that cannot hold: at 0.05 for both, the pooled
            # target's motion fell 0.0154 -> 0.0015 in 600 steps.
            pooled_temporal = temporal_separation_hinge(
                current_latent, future_latents, floor=self.pooled_temporal_floor
            )
            # The query axis, which the flattened distributional terms cannot see:
            # sixteen queries had collapsed onto one vector (off-diagonal cosine
            # 0.99993) while those terms read healthy, because duplicate rows leave
            # both per-dimension variance and between-dimension covariance intact.
            pooled_stack = torch.cat((current_latent[:, None], future_latents), dim=1)
            pooled_query_diversity = query_diversity_penalty(pooled_stack)
            # Diagnostics only, under no_grad: the two numbers that made the pooled
            # collapse legible offline, so the next one is visible as it happens.
            pooled_rank = effective_rank(pooled_stack)
            encoder_rank = effective_rank(static_grid[:, :, 0])
            loss_pooled_reg = (
                self.lambda_pooled_latent_reg * pooled_sigreg
                + self.lambda_pooled_covariance * pooled_covariance
                + self.lambda_pooled_variance * pooled_variance
                + self.lambda_temporal_hinge * pooled_temporal
                + self.lambda_query_diversity * pooled_query_diversity
            )
            total_loss = total_loss + loss_pooled_reg
        loss_latent_sigreg = None
        if self.lambda_latent_sigreg:
            loss_latent_sigreg = sketched_isotropic_gaussian_regularization(
                current_features, num_directions=self.sigreg_num_directions
            )
            total_loss = total_loss + self.lambda_latent_sigreg * loss_latent_sigreg
        loss_latent_variance = loss_latent_covariance = None
        if self.lambda_latent_variance or self.lambda_latent_covariance:
            loss_latent_variance, loss_latent_covariance = variance_covariance_regularization(
                current_features, variance_floor=self.latent_variance_floor
            )
            total_loss = (
                total_loss
                + self.lambda_latent_variance * loss_latent_variance
                + self.lambda_latent_covariance * loss_latent_covariance
            )
        output = {
            "loss": total_loss,
            "loss_action": loss_action,
            "loss_latent_ar": motion_losses["motion"],
            "loss_motion_residual": motion_losses["motion"],
            "loss_absolute_latent": motion_losses["absolute"],
            "loss_horizon_consistency": motion_losses["consistency"],
            "loss_latent_variance": loss_latent_variance,
            "loss_latent_covariance": loss_latent_covariance,
            "loss_latent_sigreg": loss_latent_sigreg,
            "loss_pooled_reg": loss_pooled_reg,
            "loss_temporal_hinge": pooled_temporal if self.fast_lewm_latent_pooler is not None
                                   and self.lambda_pooled_latent_reg else None,
            "pooled_variance": pooled_variance if self.fast_lewm_latent_pooler is not None
                               and self.lambda_pooled_latent_reg else None,
            "pooled_covariance": pooled_covariance if self.fast_lewm_latent_pooler is not None
                                 and self.lambda_pooled_latent_reg else None,
            "pooled_sigreg": pooled_sigreg if self.fast_lewm_latent_pooler is not None
                             and self.lambda_pooled_latent_reg else None,
            "query_diversity": pooled_query_diversity if self.fast_lewm_latent_pooler is not None
                               and self.lambda_pooled_latent_reg else None,
            "pooled_effective_rank": pooled_rank if self.fast_lewm_latent_pooler is not None
                                     and self.lambda_pooled_latent_reg else None,
            "encoder_effective_rank": encoder_rank if self.fast_lewm_latent_pooler is not None
                                      and self.lambda_pooled_latent_reg else None,
            "target_motion_rms": motion_losses["target_motion_rms"],
            "residual_cosine": motion_losses["residual_cosine"],
            "motion_agreement": motion_losses.get("motion_agreement"),
            "centred_agreement": motion_losses.get("centred_agreement"),
            "action_prediction": action_prediction,
            "latent_prediction": latent_prediction,
            "query_latent_prediction": latent_prediction,
            "action_prefixes": action_prefixes,
            "fast_lewm_conditioning": self.fast_lewm_action_conditioning,
            "llm_hidden": hidden,
        }
        if loss_visual_token_cosine is not None:
            output["loss_visual_token_cosine"] = loss_visual_token_cosine
        return output

    def _forward_five_tubelet_ar(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_pairs: torch.Tensor,
        future_frames: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict:
        if self.latent_patch_codec is None:
            raise RuntimeError("five-tubelet AR requires latent_patch_codec")
        if current_frame_pairs.ndim != 6 or current_frame_pairs.shape[2] != 2:
            raise ValueError("current_frame_pairs must have shape [B,V,2,3,H,W]")
        if future_frames.ndim != 6 or future_frames.shape[2] != 5:
            raise ValueError("future_frames must have shape [B,V,5,3,H,W]")
        if future_frames.shape[1] != current_frame_pairs.shape[1]:
            raise ValueError("current and future camera counts must match")
        if actions is None or proprio is None:
            raise ValueError("five-tubelet AR training requires actions and proprio")

        with torch.no_grad():
            current_grid = self.vision_backbone.encode_pair(current_frame_pairs)
            fixed_anchor_pairs = build_fixed_anchor_pairs(current_frame_pairs, future_frames)
            batch, views, horizons, pair_frames, channels, height, width = fixed_anchor_pairs.shape
            flat_pairs = fixed_anchor_pairs.reshape(
                batch * views * horizons, pair_frames, channels, height, width
            )
            anchor = current_frame_pairs[:, :, -1]
            static_pairs = torch.stack((anchor, anchor), dim=2).reshape(
                batch * views, pair_frames, channels, height, width
            )
            flat_supervision_grid = self.vision_backbone.encode_pair(
                torch.cat((static_pairs, flat_pairs), dim=0)
            )
            if flat_supervision_grid.shape[1:3] != (1, 1):
                raise ValueError(
                    "independent fixed-anchor encoding must return one view and one temporal token"
                )
            static_grid = flat_supervision_grid[: batch * views].reshape(
                batch, views, 1, *flat_supervision_grid.shape[3:]
            )
            future_grid = flat_supervision_grid[batch * views :, 0, 0].reshape(
                batch,
                views,
                horizons,
                *flat_supervision_grid.shape[3:],
            )
        current_tokens = self.latent_patch_codec(current_grid, direction="encode").squeeze(1)
        horizon_queries = self.latent_patch_codec(current_tokens, direction="query")

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        text, text_mask, action_tokens = self._split_text_and_action_embeddings(
            input_embeddings, attention_mask
        )
        prefix = torch.cat((text[:, :1], current_tokens, text[:, 1:]), dim=1)
        current_mask = torch.ones(
            current_tokens.shape[:2], dtype=text_mask.dtype, device=text_mask.device
        )
        prefix_mask = torch.cat((text_mask[:, :1], current_mask, text_mask[:, 1:]), dim=1)
        video_length = horizon_queries.shape[1] * horizon_queries.shape[2]
        video_inputs = horizon_queries.reshape(
            horizon_queries.shape[0], video_length, horizon_queries.shape[-1]
        )
        joint_inputs = torch.cat((prefix, video_inputs, action_tokens), dim=1)
        joint_mask = build_global_action_visibility_mask(
            prefix_mask,
            n_horizons=5,
            video_tokens_per_horizon=horizon_queries.shape[2],
            action_tokens=action_tokens.shape[1],
            dtype=joint_inputs.dtype,
        )
        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=joint_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=joint_inputs,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        hidden = llm_output.hidden_states[-1]
        video_hidden = hidden[:, prefix.shape[1] : prefix.shape[1] + video_length].reshape_as(
            horizon_queries
        )
        latent_prediction = self.latent_patch_codec(
            video_hidden, direction="decode", views=3, tubelets=5
        )
        motion_losses = motion_residual_losses(
            latent_prediction,
            future_grid,
            static_grid,
            horizon_weights=self.horizon_loss_weights,
        )

        action_hidden = hidden[:, -self.action_placeholder_tokens :]
        motion_hidden = (video_hidden - horizon_queries).reshape(
            batch, horizons, views, -1, video_hidden.shape[-1]
        )
        motion_summaries = motion_hidden.mean(dim=3).reshape(batch, horizons * views, -1)
        action_memory = torch.cat((motion_summaries, action_hidden), dim=1)
        if actions.shape[1] % len(self.action_horizon_weights):
            raise ValueError("action horizon must be divisible by action_horizon_weights")
        action_step_weights = torch.as_tensor(
            self.action_horizon_weights, dtype=actions.dtype, device=actions.device
        ).repeat_interleave(actions.shape[1] // len(self.action_horizon_weights))
        loss_action, action_prediction = self.action_head(
            action_memory,
            proprio,
            actions,
            action_step_weights=action_step_weights,
        )
        total_loss = (
            loss_action
            + self.lambda_latent_ar * motion_losses["motion"]
            + self.lambda_absolute_latent * motion_losses["absolute"]
            + self.lambda_horizon_consistency * motion_losses["consistency"]
        )
        return {
            "loss": total_loss,
            "loss_action": loss_action,
            "loss_latent_ar": motion_losses["motion"],
            "loss_motion_residual": motion_losses["motion"],
            "loss_absolute_latent": motion_losses["absolute"],
            "loss_horizon_consistency": motion_losses["consistency"],
            "action_prediction": action_prediction,
            "latent_prediction": latent_prediction,
            "motion_summaries": motion_summaries,
            "llm_hidden": hidden,
        }

    def _forward_action_conditioned_dynamics(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_pairs: torch.Tensor,
        future_frames: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict:
        """Predict five action-conditioned future latents in parallel, fused inside Qwen.

        Unlike ``_forward_fast_lewm`` (query + conditioning computed by a small
        head after Qwen) and ``_forward_five_tubelet_ar`` (video horizons see
        each other's future and are *not* action-conditioned), this path feeds
        a per-horizon video query directly into Qwen's own input sequence,
        placed *after* the action-generation placeholders, with a custom mask
        (``build_action_conditioned_dynamics_mask``) so:
          - horizon k's video tokens see the current observation + the
            *hidden states* of action-generation sub-group 0..k (cumulative),
            never another horizon's video tokens (all five are still predicted
            in parallel, not chained) -- no ground-truth action values are fed
            in anywhere, so this is identical at train and inference time;
          - the action-generation placeholders still see each other as one
            joint block (unchanged flow-matching convention) plus the prefix.
        Qwen's own attention performs the forward-dynamics computation; there
        is no separate world-model transformer.
        """
        if self.latent_patch_codec is None:
            raise RuntimeError("action-conditioned dynamics requires latent_patch_codec")
        if current_frame_pairs.ndim != 6 or current_frame_pairs.shape[2] != 2:
            raise ValueError("current_frame_pairs must have shape [B,V,2,3,H,W]")
        if future_frames.ndim != 6 or future_frames.shape[2] != 5:
            raise ValueError("future_frames must have shape [B,V,5,3,H,W]")
        if future_frames.shape[1] != current_frame_pairs.shape[1]:
            raise ValueError("current and future camera counts must match")
        if actions is None or proprio is None:
            raise ValueError("action-conditioned dynamics training requires actions and proprio")

        with torch.no_grad():
            current_grid = self.vision_backbone.encode_pair(current_frame_pairs)
            fixed_anchor_pairs = build_fixed_anchor_pairs(current_frame_pairs, future_frames)
            batch, views, horizons, pair_frames, channels, height, width = fixed_anchor_pairs.shape
            flat_pairs = fixed_anchor_pairs.reshape(
                batch * views * horizons, pair_frames, channels, height, width
            )
            anchor = current_frame_pairs[:, :, -1]
            static_pairs = torch.stack((anchor, anchor), dim=2).reshape(
                batch * views, pair_frames, channels, height, width
            )
            flat_supervision_grid = self.vision_backbone.encode_pair(
                torch.cat((static_pairs, flat_pairs), dim=0)
            )
            if flat_supervision_grid.shape[1:3] != (1, 1):
                raise ValueError(
                    "independent fixed-anchor encoding must return one view and one temporal token"
                )
            static_grid = flat_supervision_grid[: batch * views].reshape(
                batch, views, 1, *flat_supervision_grid.shape[3:]
            )
            future_grid = flat_supervision_grid[batch * views :, 0, 0].reshape(
                batch,
                views,
                horizons,
                *flat_supervision_grid.shape[3:],
            )

        if horizons != self.dynamics_num_horizons:
            raise ValueError("supervision horizon count does not match dynamics_num_horizons")
        current_tokens = self.latent_patch_codec(current_grid, direction="encode").squeeze(1)
        horizon_queries = self.latent_patch_codec(current_tokens, direction="query")

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        text, text_mask, action_tokens = self._split_text_and_action_embeddings(
            input_embeddings, attention_mask
        )
        prefix = torch.cat((text[:, :1], current_tokens, text[:, 1:]), dim=1)
        current_mask = torch.ones(
            current_tokens.shape[:2], dtype=text_mask.dtype, device=text_mask.device
        )
        prefix_mask = torch.cat((text_mask[:, :1], current_mask, text_mask[:, 1:]), dim=1)
        video_length = horizon_queries.shape[1] * horizon_queries.shape[2]
        video_inputs = horizon_queries.reshape(
            horizon_queries.shape[0], video_length, horizon_queries.shape[-1]
        )
        # action-gen placeholders come *before* the video queries so horizon k's
        # video tokens can attend to (still-undecoded) action-gen hidden states
        # instead of needing real action values as input.
        joint_inputs = torch.cat((prefix, action_tokens, video_inputs), dim=1)
        joint_mask = build_action_conditioned_dynamics_mask(
            prefix_mask,
            n_horizons=self.dynamics_num_horizons,
            video_tokens_per_horizon=video_length // self.dynamics_num_horizons,
            action_tokens=action_tokens.shape[1],
            dtype=joint_inputs.dtype,
        )
        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=joint_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=joint_inputs,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        hidden = llm_output.hidden_states[-1]
        action_hidden = hidden[:, prefix.shape[1] : prefix.shape[1] + self.action_placeholder_tokens]
        video_start = prefix.shape[1] + self.action_placeholder_tokens
        video_hidden = hidden[:, video_start : video_start + video_length].reshape_as(horizon_queries)
        latent_prediction = self.latent_patch_codec(
            video_hidden, direction="decode", views=3, tubelets=self.dynamics_num_horizons
        )
        motion_losses = motion_residual_losses(
            latent_prediction,
            future_grid,
            static_grid,
            horizon_weights=self.horizon_loss_weights,
        )
        if actions.shape[1] % len(self.action_horizon_weights):
            raise ValueError("action horizon must be divisible by action_horizon_weights")
        action_step_weights = torch.as_tensor(
            self.action_horizon_weights, dtype=actions.dtype, device=actions.device
        ).repeat_interleave(actions.shape[1] // len(self.action_horizon_weights))
        loss_action, action_prediction = self.action_head(
            action_hidden,
            proprio,
            actions,
            action_step_weights=action_step_weights,
        )
        total_loss = (
            loss_action
            + self.lambda_latent_ar * motion_losses["motion"]
            + self.lambda_absolute_latent * motion_losses["absolute"]
            + self.lambda_horizon_consistency * motion_losses["consistency"]
        )
        return {
            "loss": total_loss,
            "loss_action": loss_action,
            "loss_latent_ar": motion_losses["motion"],
            "loss_motion_residual": motion_losses["motion"],
            "loss_absolute_latent": motion_losses["absolute"],
            "loss_horizon_consistency": motion_losses["consistency"],
            "action_prediction": action_prediction,
            "latent_prediction": latent_prediction,
            "llm_hidden": hidden,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Union[torch.FloatTensor, Dict[str, torch.Tensor]],
        pair_pixel_values: Optional[Union[torch.FloatTensor, Dict[str, torch.Tensor]]] = None,
        current_frame_pairs: Optional[torch.FloatTensor] = None,
        future_frames: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[list] = None,
    ) -> dict:
        """Run the fixed JEPA-WAM action and visual-alignment forward pass."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected input_ids and attention_mask with matching [B, L] shapes.")
        # A world-model architecture with no `current_frame_pairs` in the batch
        # means the dataset was built without the current-pair/future-frame
        # layout (see the `enable_five_tubelet_ar` dataset flag in train.py).
        # Falling through to the plain VLA path here would silently train a
        # completely different model than the config asks for, so fail loudly.
        if (
            self.enable_fast_lewm
            or self.enable_five_tubelet_ar
            or self.enable_action_conditioned_dynamics
        ) and current_frame_pairs is None:
            raise ValueError(
                "world-model forward paths require `current_frame_pairs` in the batch; "
                "the dataset must be constructed with the current-pair/future-frame layout"
            )
        if self.enable_fast_lewm and current_frame_pairs is not None:
            if future_frames is None:
                raise ValueError("future_frames are required with current_frame_pairs")
            return self._forward_fast_lewm(
                input_ids,
                attention_mask,
                pixel_values,
                current_frame_pairs,
                future_frames,
                actions,
                proprio,
                self._resolve_embodiment(dataset_names),
            )
        if self.enable_five_tubelet_ar and current_frame_pairs is not None:
            if future_frames is None:
                raise ValueError("future_frames are required with current_frame_pairs")
            return self._forward_five_tubelet_ar(
                input_ids,
                attention_mask,
                current_frame_pairs,
                future_frames,
                actions,
                proprio,
            )
        if self.enable_action_conditioned_dynamics and current_frame_pairs is not None:
            if future_frames is None:
                raise ValueError("future_frames are required with current_frame_pairs")
            return self._forward_action_conditioned_dynamics(
                input_ids,
                attention_mask,
                current_frame_pairs,
                future_frames,
                actions,
                proprio,
            )

        if getattr(self, "enable_official_latent_head", False) and self.training and pair_pixel_values is None:
            raise ValueError("Official auxiliary training requires the released (t,t+31) target pairs")

        memory_stats = [] if getattr(self, "debug_memory_stats", False) else None

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            patch_features = self.vision_backbone(pixel_values)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_vision_encode")) is not None:
            memory_stats.append(snap)

        pair_vjepa_target = None
        if pair_pixel_values is not None:
            if not hasattr(self.vision_backbone, "encode_pair"):
                raise TypeError("The configured vision backbone does not implement paired-frame encoding.")
            with torch.no_grad():
                pair_vjepa_target = self.vision_backbone.encode_pair(pair_pixel_values)

        projected_patch_embeddings = self.projector(patch_features)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_projector")) is not None:
            memory_stats.append(snap)

        projected_patch_attention_mask = torch.ones(
            projected_patch_embeddings.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        fused_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        fused_attention_mask = torch.cat(
            [
                attention_mask[:, :1],
                projected_patch_attention_mask,
                attention_mask[:, 1:],
            ],
            dim=1,
        )

        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_llm_forward")) is not None:
            memory_stats.append(snap)
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states.")

        llm_hidden = llm_output.hidden_states[-1]
        vision_token_count = projected_patch_embeddings.shape[1]
        vision_memory = llm_hidden[:, 1 : 1 + vision_token_count, :]
        action_memory = self._select_action_memory(
            llm_hidden,
            fused_attention_mask,
            self.action_placeholder_tokens,
        )

        if isinstance(pixel_values, torch.Tensor):
            num_views = pixel_values.shape[1] if pixel_values.ndim == 5 else 1
        else:
            example = next(iter(pixel_values.values()))
            num_views = example.shape[1] if example.ndim == 5 else 1
        if vision_token_count % num_views != 0:
            raise ValueError(
                f"Vision token count {vision_token_count} is not divisible by num views {num_views}."
            )

        total_loss = llm_hidden.new_zeros(())
        loss_action = None
        loss_visual_token_cosine = None

        if (actions is None) != (proprio is None):
            raise ValueError("Actions and proprio must be provided together.")
        if actions is not None:
            if actions.ndim == 4 and actions.shape[1] == 1:
                actions = actions.squeeze(1)
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)
            if proprio.ndim == 3 and proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)

            loss_action, _ = self.action_head(
                action_memory,
                proprio,
                actions,
            )
            total_loss = total_loss + loss_action

        if pair_vjepa_target is not None and self.training:
            if pair_vjepa_target.shape[2] != 1:
                raise ValueError(
                    "Visual-token cosine supervision expects one temporal target token, "
                    f"got {tuple(pair_vjepa_target.shape)}."
                )
            target_grid = pair_vjepa_target.squeeze(2)
            target_visual_tokens = target_grid.reshape(
                target_grid.shape[0],
                target_grid.shape[1] * target_grid.shape[2] * target_grid.shape[3],
                target_grid.shape[-1],
            ).detach()
            if vision_memory.shape[1] != target_visual_tokens.shape[1]:
                raise ValueError(
                    f"Visual token count mismatch: prediction={vision_memory.shape[1]}, "
                    f"target={target_visual_tokens.shape[1]}."
                )

            loss_visual_token_cosine, _ = self.visual_token_cosine_head(
                vision_memory,
                target_visual_tokens,
            )
            total_loss = total_loss + self.lambda_visual_token_cosine * loss_visual_token_cosine
            if memory_stats is not None and (
                snap := _maybe_cuda_mem_snapshot("after_visual_token_cosine_head")
            ) is not None:
                memory_stats.append(snap)

        loss_official_latent = None
        if getattr(self, "official_latent_head", None) is not None and self.training and pair_vjepa_target is not None:
            loss_official_latent, _ = self.official_latent_head(
                vision_memory, target_visual_tokens, num_views
            )
            total_loss = total_loss + self.lambda_official_latent * loss_official_latent
        output = {"loss": total_loss, "llm_hidden": llm_hidden}
        if loss_official_latent is not None:
            output["loss_world"] = loss_official_latent
        if loss_action is not None:
            output["loss_action"] = loss_action
        if loss_visual_token_cosine is not None:
            output["loss_visual_token_cosine"] = loss_visual_token_cosine
        if memory_stats is not None:
            output["memory_stats"] = memory_stats
        return output
