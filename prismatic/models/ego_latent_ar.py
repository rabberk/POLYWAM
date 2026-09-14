"""Full-spatial-token teacher-forcing utilities for the Ego world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


def pool_spatial_latent_grid(grid: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Average non-overlapping spatial blocks in a latent grid ending in ``[H,W,D]``."""
    if grid.ndim < 3:
        raise ValueError("latent grid must end in [H,W,D]")
    merge = int(merge_size)
    if merge <= 0:
        raise ValueError("merge_size must be positive")
    height, width, dim = grid.shape[-3:]
    if height % merge or width % merge:
        raise ValueError(
            f"latent grid ({height}, {width}) must be divisible by merge_size={merge}"
        )
    if merge == 1:
        return grid
    blocked = grid.reshape(
        *grid.shape[:-3],
        height // merge,
        merge,
        width // merge,
        merge,
        dim,
    )
    return blocked.mean(dim=(-4, -2))


def expand_spatial_latent_grid(grid: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Repeat pooled latent tokens back onto their original spatial grid for decoding."""
    if grid.ndim < 3:
        raise ValueError("latent grid must end in [H,W,D]")
    merge = int(merge_size)
    if merge <= 0:
        raise ValueError("merge_size must be positive")
    if merge == 1:
        return grid
    return grid.repeat_interleave(merge, dim=-3).repeat_interleave(merge, dim=-2)


def autoregressive_rollout_tubelets(
    seed_tubelet: torch.Tensor,
    predict_next: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_tubelets: int,
    tokens_per_tubelet: int,
) -> torch.Tensor:
    """Generate tubelets using only the GT seed and earlier predictions."""
    if seed_tubelet.ndim != 3:
        raise ValueError("seed_tubelet must have shape [B,N,D]")
    if n_tubelets <= 0 or tokens_per_tubelet <= 0:
        raise ValueError("n_tubelets and tokens_per_tubelet must be positive")
    if seed_tubelet.shape[1] != tokens_per_tubelet:
        raise ValueError("seed_tubelet must contain exactly one tubelet")
    history = seed_tubelet
    predictions = []
    expected_shape = (seed_tubelet.shape[0], tokens_per_tubelet, seed_tubelet.shape[2])
    for _ in range(n_tubelets):
        prediction = predict_next(history)
        if tuple(prediction.shape) != expected_shape:
            raise ValueError(f"predict_next returned {tuple(prediction.shape)}, expected {expected_shape}")
        if prediction.is_floating_point() and not torch.isfinite(prediction).all():
            raise ValueError("predict_next returned non-finite values")
        predictions.append(prediction)
        history = torch.cat((history, prediction), dim=1)
    return torch.cat(predictions, dim=1)


@dataclass(frozen=True)
class EgoLatentARConfig:
    image_size: int = 384
    patch_size: int = 16
    tubelet_size: int = 2
    n_future_tubelets: int = 5
    patch_merge_size: int = 1

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.tubelet_size != 2:
            raise ValueError("tubelet_size must be 2 for the Ego five-tubelet recipe")
        if self.n_future_tubelets != 5:
            raise ValueError("n_future_tubelets must be 5 for the Ego five-tubelet recipe")
        if self.patch_merge_size not in {1, 2}:
            raise ValueError("patch_merge_size must be 1 or 2")
        if (self.image_size // self.patch_size) % self.patch_merge_size:
            raise ValueError("raw spatial side must be divisible by patch_merge_size")

    @property
    def raw_spatial_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def spatial_side(self) -> int:
        return self.raw_spatial_side // self.patch_merge_size

    @property
    def tokens_per_tubelet(self) -> int:
        return self.spatial_side * self.spatial_side

    @property
    def total_future_tokens(self) -> int:
        return self.n_future_tubelets * self.tokens_per_tubelet


def build_teacher_forcing_latents(
    current_latents: torch.Tensor,
    future_targets: torch.Tensor,
    tokens_per_tubelet: int,
) -> torch.Tensor:
    """Right-shift complete spatial tubelets, using the current tubelet as BOS."""
    if current_latents.ndim != 3 or future_targets.ndim != 3:
        raise ValueError("current_latents and future_targets must both have shape [B, L, D]")
    if current_latents.shape[0] != future_targets.shape[0] or current_latents.shape[2] != future_targets.shape[2]:
        raise ValueError("current and future latent batch/embedding dimensions must match")
    if tokens_per_tubelet <= 0:
        raise ValueError("tokens_per_tubelet must be positive")
    if current_latents.shape[1] < tokens_per_tubelet:
        raise ValueError("current_latents does not contain one complete tubelet")
    if future_targets.shape[1] == 0 or future_targets.shape[1] % tokens_per_tubelet:
        raise ValueError("future_targets must contain a whole number of tubelets")

    current_bos = current_latents[:, -tokens_per_tubelet:]
    return torch.cat((current_bos, future_targets[:, :-tokens_per_tubelet]), dim=1)


def build_block_causal_attention_mask(
    prefix_attention_mask: torch.Tensor,
    n_tubelets: int,
    tokens_per_tubelet: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build `[B, 1, S, S]` additive attention with block-causal future groups.

    Prefix rows retain Qwen's native causal ordering. Future rows see all valid
    prefix keys, every earlier future group, and every token in their own group.
    """
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B, P]")
    if n_tubelets <= 0 or tokens_per_tubelet <= 0:
        raise ValueError("n_tubelets and tokens_per_tubelet must be positive")
    if not dtype.is_floating_point:
        raise ValueError("attention mask dtype must be floating point")

    prefix_valid = prefix_attention_mask.bool()
    batch_size, prefix_length = prefix_valid.shape
    future_length = n_tubelets * tokens_per_tubelet
    sequence_length = prefix_length + future_length
    device = prefix_valid.device

    allowed = torch.zeros(batch_size, sequence_length, sequence_length, dtype=torch.bool, device=device)

    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal.unsqueeze(0) & prefix_valid[:, None, :]

    allowed[:, prefix_length:, :prefix_length] = prefix_valid[:, None, :]
    group_ids = torch.arange(future_length, device=device) // tokens_per_tubelet
    future_allowed = group_ids[:, None] >= group_ids[None, :]
    allowed[:, prefix_length:, prefix_length:] = future_allowed.unsqueeze(0)

    additive = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


class EgoLatentARHead(nn.Module):
    """Bidirectional projection pair between standardized V-JEPA and Qwen spaces."""

    def __init__(self, jepa_dim: int, llm_dim: int) -> None:
        super().__init__()
        if jepa_dim <= 0 or llm_dim <= 0:
            raise ValueError("jepa_dim and llm_dim must be positive")
        self.input_projection = nn.Linear(jepa_dim, llm_dim)
        self.output_norm = nn.LayerNorm(llm_dim)
        self.output_projection = nn.Linear(llm_dim, jepa_dim)

    def project_input(self, latent: torch.Tensor) -> torch.Tensor:
        return self.input_projection(latent)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.output_norm(hidden))

    def forward(self, tensor: torch.Tensor, *, direction: str = "predict") -> torch.Tensor:
        """Run either projection through ``forward`` so FSDP hooks materialize weights."""
        if direction == "project_input":
            return self.project_input(tensor)
        if direction == "predict":
            return self.predict(tensor)
        raise ValueError(f"unsupported latent AR head direction: {direction!r}")


def standardized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {tuple(prediction.shape)} != {tuple(target.shape)}")
    return (prediction.float() - target.float()).square().mean()
