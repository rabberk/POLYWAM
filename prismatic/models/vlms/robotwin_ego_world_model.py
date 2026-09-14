"""Joint RoboTwin qpos14 action and Ego five-tubelet JEPA-WAM model."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from prismatic.models.b300_fa4_attention import build_fa4_block_mask
from prismatic.models.ego_latent_ar import (
    EgoLatentARHead,
    build_block_causal_attention_mask,
    build_teacher_forcing_latents,
    expand_spatial_latent_grid,
    pool_spatial_latent_grid,
    standardized_mse,
)
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.vlms.ego_world_model import Ego5TubeletWorldModel
from prismatic.util.nn_utils import MLPProjector


def load_ego_fullft_state(model: nn.Module, checkpoint_path: str | Path) -> dict[str, int]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("model", checkpoint)
    modules = {
        "llm_backbone": model.llm_backbone,
        "projector": model.projector,
        "latent_ar_head": model.latent_ar_head,
    }
    summary = {}
    for prefix, module in modules.items():
        component = {
            key[len(prefix) + 1 :]: value
            for key, value in state.items()
            if key.startswith(prefix + ".")
        }
        if not component:
            raise ValueError(f"Ego checkpoint is missing required component {prefix}")
        incompatible = module.load_state_dict(component, strict=False)
        allowed_missing = {"llm.lm_head.weight"} if prefix == "llm_backbone" else set()
        invalid_missing = sorted(set(incompatible.missing_keys).difference(allowed_missing))
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Ego checkpoint component {prefix} mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        if "llm.lm_head.weight" in incompatible.missing_keys:
            input_weight = module.llm.get_input_embeddings().weight
            output_weight = module.llm.get_output_embeddings().weight
            if input_weight is not output_weight:
                raise RuntimeError("Ego checkpoint omits lm_head.weight but Qwen embeddings are not tied")
        summary[prefix] = len(component)
    return summary


class RoboTwinEgo5TubeletVLA(Ego5TubeletWorldModel):
    """Shared Qwen with paper action supervision and per-view full-token AR."""

    def __init__(
        self,
        vision_backbone: nn.Module,
        llm_backbone: nn.Module,
        projector: nn.Module,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        *,
        action_head: Optional[nn.Module] = None,
        action_placeholder_tokens: int = 64,
        lambda_world: float = 0.5,
        image_size: int = 384,
        patch_size: int = 16,
        patch_merge_size: int = 1,
        enforce_public_geometry: bool = True,
    ) -> None:
        super().__init__(
            vision_backbone,
            llm_backbone,
            projector,
            latent_mean,
            latent_std,
            image_size=image_size,
            patch_size=patch_size,
            patch_merge_size=patch_merge_size,
            n_future_tubelets=5,
            enforce_public_geometry=enforce_public_geometry,
        )
        if action_placeholder_tokens <= 0:
            raise ValueError("action_placeholder_tokens must be positive")
        if lambda_world < 0:
            raise ValueError("lambda_world must be non-negative")
        self.action_placeholder_tokens = int(action_placeholder_tokens)
        self.lambda_world = float(lambda_world)
        self.action_head = action_head or FlowMatchingActionHead(
            d_proprio=14,
            d_action=14,
            d_llm=int(llm_backbone.embed_dim),
            horizon=50,
            fm_hidden_size=1024,
            fm_num_layers=16,
            fm_num_inference_timesteps=4,
            fm_num_timestep_buckets=1000,
            fm_noise_beta_alpha=1.5,
            fm_noise_beta_beta=1.0,
            fm_noise_s=0.999,
            fm_num_target_vision_tokens=32,
            fm_add_pos_embed=True,
            fm_max_seq_len=1024,
            fm_state_dropout=0.5,
            prediction_type="x",
        )
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector", "latent_ar_head", "action_head"]
        self.trainable_module_keys: list[str] = []
        self.llm_transformer_layer_cls = llm_backbone.transformer_layer_cls

    def configure_for_robotwin_training(self, full_finetune_llm: bool = False) -> None:
        self.requires_grad_(False)
        if full_finetune_llm:
            self.llm_backbone.requires_grad_(True)
            self._ego_finetune_mode = "full"
        else:
            lora_names = []
            for name, parameter in self.llm_backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
                    lora_names.append(name)
            if not lora_names:
                raise RuntimeError("Qwen must have fresh LoRA adapters before RoboTwin post-training")
            self._ego_finetune_mode = "lora"
        self.latent_ar_head.requires_grad_(True)
        self.action_head.requires_grad_(True)
        self.trainable_module_keys = ["llm_backbone", "latent_ar_head", "action_head"]
        self.vision_backbone.eval()
        self.projector.eval()

    def freeze_for_training(self) -> None:
        self.configure_for_robotwin_training()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.vision_backbone.eval()
        self.projector.eval()
        return self

    def get_fsdp_wrapping_policy(self):
        policies = []
        if hasattr(self.vision_backbone, "get_fsdp_wrapping_policy"):
            policies.append(self.vision_backbone.get_fsdp_wrapping_policy())
        if hasattr(self.llm_backbone, "get_fsdp_wrapping_policy"):
            policies.append(self.llm_backbone.get_fsdp_wrapping_policy())
        policies.append(
            partial(
                _module_wrap_policy,
                module_classes={MLPProjector, EgoLatentARHead, FlowMatchingActionHead},
            )
        )
        return partial(_or_policy, policies=policies)

    @staticmethod
    def _select_action_memory(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        num_action_tokens: int,
    ) -> torch.Tensor:
        """Gather the final *valid* tokens per sample, ignoring right padding."""
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("hidden states and attention mask must have matching [B,L] dimensions")
        valid_lengths = attention_mask.long().sum(dim=1)
        if torch.any(valid_lengths < num_action_tokens):
            raise ValueError("Qwen sequence is shorter than the action-placeholder span")
        token_offsets = torch.arange(num_action_tokens, device=hidden.device)
        indices = valid_lengths[:, None] - num_action_tokens + token_offsets[None, :]
        return hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))

    @staticmethod
    def _flatten_joint_view_latents(
        grid: torch.Tensor,
        expected_views: int,
        expected_tubelets: int,
    ) -> tuple[torch.Tensor, int]:
        """Flatten `[B,V,T,H,W,D]` in time-major, then view-major order."""
        if (
            grid.ndim != 6
            or grid.shape[1] != expected_views
            or grid.shape[2] != expected_tubelets
        ):
            raise ValueError(
                f"expected V-JEPA grid [B,{expected_views},{expected_tubelets},H,W,D], "
                f"got {tuple(grid.shape)}"
            )
        spatial_tokens = grid.shape[3] * grid.shape[4]
        tokens_per_joint_tubelet = expected_views * spatial_tokens
        flattened = grid.permute(0, 2, 1, 3, 4, 5).reshape(
            grid.shape[0], expected_tubelets * tokens_per_joint_tubelet, grid.shape[-1]
        )
        return flattened, spatial_tokens

    def _pool_flat_view_latents(
        self,
        latents: torch.Tensor,
        views: int,
        raw_spatial_side: int,
    ) -> torch.Tensor:
        """Pool flattened view-major V-JEPA tokens and preserve their ordering."""
        if latents.ndim != 3:
            raise ValueError("flat view latents must have shape [B,V*H*W,D]")
        expected_tokens = views * raw_spatial_side * raw_spatial_side
        if latents.shape[1] != expected_tokens:
            raise ValueError(
                f"flat view latents contain {latents.shape[1]} tokens, expected {expected_tokens}"
            )
        grid = latents.reshape(
            latents.shape[0],
            views,
            raw_spatial_side,
            raw_spatial_side,
            latents.shape[-1],
        )
        pooled = pool_spatial_latent_grid(grid, self.ar_config.patch_merge_size)
        return pooled.reshape(latents.shape[0], -1, latents.shape[-1])

    def expand_world_prediction_for_decoder(self, prediction: torch.Tensor) -> torch.Tensor:
        """Expand pooled `[B,V,T*P,D]` predictions back to the raw V-JEPA grid."""
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("world prediction must have shape [B,3,T*P,D]")
        pooled_side = self.ar_config.spatial_side
        expected_tokens = self.ar_config.n_future_tubelets * pooled_side * pooled_side
        if prediction.shape[2] != expected_tokens:
            raise ValueError(
                f"world prediction contains {prediction.shape[2]} tokens per view, expected {expected_tokens}"
            )
        grid = prediction.reshape(
            prediction.shape[0],
            prediction.shape[1],
            self.ar_config.n_future_tubelets,
            pooled_side,
            pooled_side,
            prediction.shape[-1],
        )
        expanded = expand_spatial_latent_grid(grid, self.ar_config.patch_merge_size)
        return expanded.reshape(prediction.shape[0], prediction.shape[1], -1, prediction.shape[-1])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        world_input_ids: torch.Tensor,
        world_attention_mask: torch.Tensor,
        current_frame_pairs: torch.Tensor,
        future_frames: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if current_frame_pairs.ndim != 6 or current_frame_pairs.shape[1:3] != (3, 2):
            raise ValueError("current_frame_pairs must have shape [B,3,2,3,H,W]")
        if future_frames.ndim != 6 or future_frames.shape[1:3] != (3, 10):
            raise ValueError("future_frames must have shape [B,3,10,3,H,W]")
        del world_input_ids, world_attention_mask
        batch_size, views = current_frame_pairs.shape[:2]
        action_observation = current_frame_pairs[:, :, 1]

        with torch.no_grad():
            action_visual_raw = self.vision_backbone(action_observation)
            current_grid = self.vision_backbone.encode_pair(current_frame_pairs)
            future_grid = self.vision_backbone.encode_pair(future_frames)
        action_visual_raw = self._pool_flat_view_latents(
            action_visual_raw,
            views,
            self.ar_config.raw_spatial_side,
        )
        current_grid = pool_spatial_latent_grid(current_grid, self.ar_config.patch_merge_size)
        future_grid = pool_spatial_latent_grid(future_grid, self.ar_config.patch_merge_size)
        current_raw, spatial_tokens = self._flatten_joint_view_latents(current_grid, views, 1)
        future_raw, future_spatial_tokens = self._flatten_joint_view_latents(
            future_grid, views, self.ar_config.n_future_tubelets
        )
        if future_spatial_tokens != spatial_tokens:
            raise ValueError("current and future V-JEPA grids have different spatial geometry")
        tokens_per_joint_tubelet = views * spatial_tokens
        current_standardized = self._standardize(current_raw)
        future_target = self._standardize(future_raw).detach()
        teacher_forcing = build_teacher_forcing_latents(
            current_standardized,
            future_target,
            tokens_per_joint_tubelet,
        )

        projector_dtype = next(self.projector.parameters()).dtype
        text_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        action_visual = self.projector(action_visual_raw.to(projector_dtype)).to(text_embeddings.dtype)
        current_visual = self.projector(current_raw.to(projector_dtype)).to(text_embeddings.dtype)
        action_visual_mask = torch.ones(
            action_visual.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        action_prefix = torch.cat(
            (text_embeddings[:, :1], action_visual, text_embeddings[:, 1:]), dim=1
        )
        action_prefix_mask = torch.cat(
            (attention_mask[:, :1], action_visual_mask, attention_mask[:, 1:]), dim=1
        )

        current_mask = torch.ones(
            current_visual.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        prefix_embeddings = torch.cat((action_prefix, current_visual), dim=1)
        prefix_mask = torch.cat((action_prefix_mask, current_mask), dim=1)
        latent_embeddings = self.latent_ar_head(
            teacher_forcing.to(text_embeddings.dtype), direction="project_input"
        )
        inputs_embeds = torch.cat((prefix_embeddings, latent_embeddings), dim=1)
        if getattr(getattr(self.llm_backbone.llm, "config", None), "_attn_implementation", None) == "b300_fa4":
            attention_mask_for_qwen = build_fa4_block_mask(
                prefix_mask,
                self.ar_config.n_future_tubelets,
                tokens_per_joint_tubelet,
            )
        else:
            attention_mask_for_qwen = build_block_causal_attention_mask(
                prefix_mask,
                self.ar_config.n_future_tubelets,
                tokens_per_joint_tubelet,
                inputs_embeds.dtype,
            )
        output = self.llm_backbone(
            input_ids=None,
            attention_mask=attention_mask_for_qwen,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        hidden = output.hidden_states[-1]

        action_hidden = hidden[:, : action_prefix.shape[1]]
        action_memory = self._select_action_memory(
            action_hidden,
            action_prefix_mask,
            self.action_placeholder_tokens,
        )
        loss_action, action_prediction = self.action_head(action_memory, proprio, actions)

        future_hidden = hidden[:, -future_target.shape[1] :]
        joint_prediction = self.latent_ar_head(future_hidden, direction="predict")
        loss_world = standardized_mse(joint_prediction, future_target)
        world_prediction = joint_prediction.reshape(
            batch_size,
            self.ar_config.n_future_tubelets,
            views,
            spatial_tokens,
            joint_prediction.shape[-1],
        ).permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            views,
            self.ar_config.n_future_tubelets * spatial_tokens,
            joint_prediction.shape[-1],
        )
        return {
            "loss": loss_action + self.lambda_world * loss_world,
            "loss_action": loss_action,
            "loss_world": loss_world,
            "action_prediction": action_prediction,
            "world_prediction": world_prediction,
        }
