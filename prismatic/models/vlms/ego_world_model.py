"""Action-free JEPA-WAM adaptation for full-token Ego future prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from prismatic.models.ego_latent_ar import (
    EgoLatentARConfig,
    EgoLatentARHead,
    build_block_causal_attention_mask,
    build_teacher_forcing_latents,
    standardized_mse,
)


def load_pretrained_vlm_components(checkpoint_path: str | Path) -> dict[str, Mapping[str, torch.Tensor]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = checkpoint.get("model", checkpoint)
    missing = [name for name in ("llm_backbone", "projector") if name not in model]
    if missing:
        raise ValueError(f"pretrained VLM checkpoint is missing required components: {missing}")
    return {"llm_backbone": model["llm_backbone"], "projector": model["projector"]}


class Ego5TubeletWorldModel(nn.Module):
    """Frozen JEPA-WAM perception plus LoRA-conditioned continuous latent AR."""

    def __init__(
        self,
        vision_backbone: nn.Module,
        llm_backbone: nn.Module,
        projector: nn.Module,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        image_size: int = 384,
        patch_size: int = 16,
        patch_merge_size: int = 1,
        n_future_tubelets: int = 5,
        enforce_public_geometry: bool = True,
    ) -> None:
        super().__init__()
        if enforce_public_geometry and (image_size, patch_size, n_future_tubelets) != (384, 16, 5):
            raise ValueError("public Ego recipe requires image_size=384, patch_size=16, n_future_tubelets=5")
        self.ar_config = EgoLatentARConfig(
            image_size=image_size,
            patch_size=patch_size,
            n_future_tubelets=n_future_tubelets,
            patch_merge_size=patch_merge_size,
        )
        self.vision_backbone = vision_backbone
        self.llm_backbone = llm_backbone
        self.projector = projector
        self.latent_ar_head = EgoLatentARHead(
            jepa_dim=int(vision_backbone.embed_dim),
            llm_dim=int(llm_backbone.embed_dim),
        )
        mean = torch.as_tensor(latent_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(latent_std, dtype=torch.float32).flatten()
        if mean.numel() != int(vision_backbone.embed_dim) or std.shape != mean.shape:
            raise ValueError("latent mean/std must match the V-JEPA embedding dimension")
        if not torch.all(std > 0):
            raise ValueError("latent standard deviations must be positive")
        self.register_buffer("latent_mean", mean, persistent=True)
        self.register_buffer("latent_std", std, persistent=True)
        self._ego_finetune_mode = "lora"

    def configure_for_ego_training(self, mode: str) -> None:
        if mode not in {"lora", "full"}:
            raise ValueError(f"unsupported Ego fine-tune mode: {mode}")
        self.requires_grad_(False)
        if mode == "lora":
            lora_names = []
            for name, parameter in self.llm_backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
                    lora_names.append(name)
            if not lora_names:
                raise RuntimeError("Qwen must have LoRA adapters before Ego training")
        else:
            if any("lora_" in name for name, _ in self.llm_backbone.named_parameters()):
                raise RuntimeError("full fine-tuning must not contain LoRA parameters")
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
        self.latent_ar_head.requires_grad_(True)
        self._ego_finetune_mode = mode
        self.vision_backbone.eval()
        if mode == "lora":
            self.projector.eval()

    def freeze_for_ego_training(self) -> None:
        self.configure_for_ego_training("lora")

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        if self._ego_finetune_mode == "lora":
            self.projector.eval()
        return self

    def _standardize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent.float() - self.latent_mean) / self.latent_std

    @staticmethod
    def _flatten_latents(grid: torch.Tensor, expected_tubelets: int) -> torch.Tensor:
        if grid.ndim != 6 or grid.shape[1] != 1 or grid.shape[2] != expected_tubelets:
            raise ValueError(
                f"expected V-JEPA grid [B,1,{expected_tubelets},H,W,D], got {tuple(grid.shape)}"
            )
        return grid[:, 0].reshape(grid.shape[0], -1, grid.shape[-1])

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        current_frames: torch.Tensor,
        future_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if current_frames.shape[1] != 2 or future_frames.shape[1] != 10:
            raise ValueError("expected two current frames and ten future frames")
        with torch.no_grad():
            current_grid = self.vision_backbone.encode_pair(current_frames)
            future_grid = self.vision_backbone.encode_pair(future_frames)
        current_raw = self._flatten_latents(current_grid, 1)
        future_raw = self._flatten_latents(future_grid, self.ar_config.n_future_tubelets)
        if current_raw.shape[1] != self.ar_config.tokens_per_tubelet:
            raise ValueError(
                f"current tubelet has {current_raw.shape[1]} tokens, expected {self.ar_config.tokens_per_tubelet}"
            )
        if future_raw.shape[1] != self.ar_config.total_future_tokens:
            raise ValueError(
                f"future has {future_raw.shape[1]} tokens, expected {self.ar_config.total_future_tokens}"
            )

        current_standardized = self._standardize(current_raw)
        future_target = self._standardize(future_raw).detach()
        teacher_forcing = build_teacher_forcing_latents(
            current_standardized,
            future_target,
            self.ar_config.tokens_per_tubelet,
        )

        current_projected = self.projector(current_raw.to(next(self.projector.parameters()).dtype))
        text_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        latent_embeddings = self.latent_ar_head.project_input(teacher_forcing.to(text_embeddings.dtype))
        prefix_embeddings = torch.cat(
            (text_embeddings[:, :1], current_projected.to(text_embeddings.dtype), text_embeddings[:, 1:]), dim=1
        )
        current_mask = torch.ones(
            input_ids.shape[0], current_projected.shape[1], dtype=torch.bool, device=input_ids.device
        )
        prefix_mask = torch.cat((attention_mask[:, :1].bool(), current_mask, attention_mask[:, 1:].bool()), dim=1)
        inputs_embeds = torch.cat((prefix_embeddings, latent_embeddings), dim=1)
        additive_mask = build_block_causal_attention_mask(
            prefix_mask,
            self.ar_config.n_future_tubelets,
            self.ar_config.tokens_per_tubelet,
            inputs_embeds.dtype,
        )

        output = self.llm_backbone(
            input_ids=None,
            attention_mask=additive_mask,
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
        future_hidden = output.hidden_states[-1][:, -self.ar_config.total_future_tokens :]
        prediction = self.latent_ar_head.predict(future_hidden)
        loss = standardized_mse(prediction, future_target)
        return {"loss": loss, "prediction": prediction, "target": future_target}


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}
