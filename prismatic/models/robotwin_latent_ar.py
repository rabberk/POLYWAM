"""Five-tubelet latent AR components for the paper-compatible RoboTwin recipe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_residual_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    static: torch.Tensor,
    *,
    horizon_weights: tuple[float, ...],
    normalize_target_scale: bool = False,
) -> dict[str, torch.Tensor]:
    """Supervise fixed-anchor motion while down-weighting static background patches."""
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError("prediction and target must share shape [B,V,K,H,W,D]")
    if static.ndim != 6 or static.shape[:2] != target.shape[:2] or static.shape[2] != 1:
        raise ValueError("static must have shape [B,V,1,H,W,D]")
    if static.shape[3:] != target.shape[3:]:
        raise ValueError("static and target spatial/feature geometry must match")
    if len(horizon_weights) != target.shape[2] or any(weight <= 0 for weight in horizon_weights):
        raise ValueError("horizon_weights must contain one positive value per horizon")

    feature_shape = (target.shape[-1],)
    pred_norm = F.layer_norm(prediction.float(), feature_shape)
    target_norm = F.layer_norm(target.detach().float(), feature_shape)
    static_norm = F.layer_norm(static.detach().float(), feature_shape)
    prediction_residual = pred_norm - static_norm
    target_residual = target_norm - static_norm

    # How far the encoder currently places a frame one second ahead from the same
    # frame now. It is the supervision signal itself, and it is reported because
    # it is the quantity that decides whether a falling loss means anything: the
    # encoder produces both sides here, and nothing in an absolute L1 stops it
    # from lowering the loss by encoding the future like the present. Measured on
    # the unnormalised residual so the number stays comparable across steps.
    # These residuals are [B,V,K,H,W,D] -- 707M elements at the training shape, so
    # every full-tensor temporary here costs ~2.8 GB and the first version of this
    # block OOM'd all eight ranks. `vector_norm` reduces without materialising a
    # squared copy, and the cosine is read off one batch element, which is far more
    # samples than a running diagnostic needs.
    with torch.no_grad():
        target_motion_rms = (
            torch.linalg.vector_norm(target_residual)
            / math.sqrt(target_residual.numel())
        )
        residual_cosine = F.cosine_similarity(
            prediction_residual[:1], target_residual[:1], dim=-1
        ).mean()

    if normalize_target_scale:
        # Divide both sides by the target's own scale, so shrinking the target
        # buys exactly nothing. The spatial weights below are ratios of the
        # target's magnitude to its spatial mean, so they are untouched by this.
        scale = target_motion_rms.clamp_min(1e-3)
        # In place on both: these are 707M-element tensors at the training shape,
        # so an out-of-place divide costs another 2.8 GB apiece at exactly the
        # moment activation memory peaks. Subtraction does not need its output for
        # backward, so scaling it in place is safe -- the gradient check in
        # tests/test_target_scale_normalization.py pins that down.
        prediction_residual = prediction_residual.div_(scale)
        target_residual = target_residual.div_(scale)

    magnitude = target_residual.square().mean(dim=-1).sqrt()
    spatial_mean = magnitude.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    spatial_weights = (0.25 + magnitude / spatial_mean).clamp(max=4.0)
    horizon = torch.as_tensor(
        horizon_weights, dtype=spatial_weights.dtype, device=spatial_weights.device
    ).view(1, 1, -1, 1, 1)
    combined_weights = spatial_weights * horizon
    patch_error = F.smooth_l1_loss(prediction_residual, target_residual, reduction="none").mean(dim=-1)
    motion = (patch_error * combined_weights).sum() / combined_weights.sum().clamp_min(1e-6)

    cosine_error = (1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1)).clamp_min(0.0)
    absolute_weights = horizon.expand_as(cosine_error)
    absolute = (cosine_error * absolute_weights).sum() / absolute_weights.sum().clamp_min(1e-6)

    if target.shape[2] > 1:
        pred_delta = prediction_residual[:, :, 1:] - prediction_residual[:, :, :-1]
        target_delta = target_residual[:, :, 1:] - target_residual[:, :, :-1]
        consistency = F.smooth_l1_loss(pred_delta, target_delta)
    else:
        consistency = motion.new_zeros(())
    return {
        "motion": motion,
        "absolute": absolute,
        "consistency": consistency,
        "prediction_residual": prediction_residual,
        "target_residual": target_residual,
        "spatial_weights": spatial_weights,
        "target_motion_rms": target_motion_rms,
        "residual_cosine": residual_cosine,
    }


class LatentPatchCodec(nn.Module):
    """Learned 24->12 patch embed and 12->24 full-grid reconstruction head."""

    def __init__(
        self,
        jepa_dim: int,
        llm_dim: int,
        *,
        spatial_side: int = 24,
        patch_merge_size: int = 2,
        num_horizons: int = 5,
    ) -> None:
        super().__init__()
        if jepa_dim <= 0 or llm_dim <= 0:
            raise ValueError("jepa_dim and llm_dim must be positive")
        if spatial_side <= 0 or spatial_side % patch_merge_size:
            raise ValueError("spatial_side must be divisible by patch_merge_size")
        if patch_merge_size not in (1, 2):
            raise ValueError("the RoboTwin AR recipe supports raw tokens or a 2x2 latent patch embedding")
        self.jepa_dim = int(jepa_dim)
        self.llm_dim = int(llm_dim)
        self.spatial_side = int(spatial_side)
        self.patch_merge_size = int(patch_merge_size)
        self.num_horizons = int(num_horizons)
        if self.num_horizons <= 0:
            raise ValueError("num_horizons must be positive")
        self.compressed_side = self.spatial_side // self.patch_merge_size
        if self.patch_merge_size == 1:
            self.patch_embed = nn.Linear(self.jepa_dim, self.llm_dim)
        else:
            self.patch_embed = nn.Conv2d(
                self.jepa_dim,
                self.llm_dim,
                kernel_size=self.patch_merge_size,
                stride=self.patch_merge_size,
            )
        self.output_norm = nn.LayerNorm(self.llm_dim)
        if self.patch_merge_size == 1:
            self.reconstruct = nn.Linear(self.llm_dim, self.jepa_dim)
        else:
            self.reconstruct = nn.ConvTranspose2d(
                self.llm_dim,
                self.jepa_dim,
                kernel_size=self.patch_merge_size,
                stride=self.patch_merge_size,
            )
        self.horizon_embeddings = nn.Parameter(
            torch.empty(self.num_horizons, 1, self.llm_dim)
        )
        nn.init.normal_(self.horizon_embeddings, mean=0.0, std=0.02)

    @property
    def tokens_per_view(self) -> int:
        return self.compressed_side**2

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode `[B,V,T,24,24,Dj]` to time-major Qwen tokens."""
        if grid.ndim != 6:
            raise ValueError("latent grid must have shape [B,V,T,H,W,D]")
        batch, views, tubelets, height, width, dim = grid.shape
        if (height, width, dim) != (self.spatial_side, self.spatial_side, self.jepa_dim):
            raise ValueError("latent grid does not match the configured V-JEPA geometry")
        if self.patch_merge_size == 1:
            encoded = self.patch_embed(grid)
            return encoded.permute(0, 2, 1, 3, 4, 5).reshape(
                batch, tubelets, views * self.tokens_per_view, self.llm_dim
            )
        image = grid.permute(0, 1, 2, 5, 3, 4).reshape(batch * views * tubelets, dim, height, width)
        encoded = self.patch_embed(image)
        return encoded.reshape(
            batch, views, tubelets, self.llm_dim, self.compressed_side, self.compressed_side
        ).permute(0, 2, 1, 4, 5, 3).reshape(
            batch, tubelets, views * self.tokens_per_view, self.llm_dim
        )

    def decode(self, tokens: torch.Tensor, *, views: int, tubelets: int) -> torch.Tensor:
        """Decode time-major Qwen tokens to `[B,V,T,24,24,Dj]`."""
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,V*P,D]")
        batch, token_tubelets, count, dim = tokens.shape
        expected_count = views * self.tokens_per_view
        if token_tubelets != tubelets or count != expected_count or dim != self.llm_dim:
            raise ValueError("tokens do not match the configured tubelet geometry")
        normalized = self.output_norm(tokens)
        if self.patch_merge_size == 1:
            decoded = self.reconstruct(
                normalized.reshape(
                    batch, tubelets, views, self.spatial_side, self.spatial_side, self.llm_dim
                )
            )
            return decoded.permute(0, 2, 1, 3, 4, 5)
        image = normalized.reshape(
            batch, tubelets, views, self.compressed_side, self.compressed_side, self.llm_dim
        ).permute(0, 2, 1, 5, 3, 4).reshape(
            batch * views * tubelets,
            self.llm_dim,
            self.compressed_side,
            self.compressed_side,
        )
        decoded = self.reconstruct(image)
        return decoded.reshape(
            batch, views, tubelets, self.jepa_dim, self.spatial_side, self.spatial_side
        ).permute(0, 1, 2, 4, 5, 3)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate legacy 2x2 codec weights during recursive whole-model loading."""
        patch_key = f"{prefix}patch_embed.weight"
        reconstruct_key = f"{prefix}reconstruct.weight"
        patch_weight = state_dict.get(patch_key)
        if self.patch_merge_size == 1 and patch_weight is not None and patch_weight.ndim == 4:
            state_dict[patch_key] = patch_weight.mean(dim=(-1, -2))
            state_dict[reconstruct_key] = (
                state_dict[reconstruct_key].mean(dim=(-1, -2)).transpose(0, 1)
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def build_queries(self, current: torch.Tensor, *, horizons: int) -> torch.Tensor:
        """Repeat the observed latent into independently identifiable horizon queries."""
        if current.ndim != 3 or current.shape[-1] != self.llm_dim:
            raise ValueError("current tokens must have shape [B,N,Dq]")
        if horizons != self.num_horizons:
            raise ValueError(
                f"configured for {self.num_horizons} horizons, requested {horizons}"
            )
        return current[:, None] + self.horizon_embeddings[None, :horizons]

    def forward(
        self,
        value: torch.Tensor,
        *,
        direction: str,
        views: int = 3,
        tubelets: int = 5,
    ) -> torch.Tensor:
        """Route both paths through forward so FSDP materializes sharded weights."""
        if direction == "encode":
            return self.encode(value)
        if direction == "decode":
            return self.decode(value, views=views, tubelets=tubelets)
        if direction == "query":
            return self.build_queries(value, horizons=tubelets)
        raise ValueError(f"unsupported latent patch codec direction: {direction!r}")


def build_fixed_anchor_pairs(
    current_frame_pairs: torch.Tensor,
    future_frames: torch.Tensor,
    segment_targets: bool = False,
) -> torch.Tensor:
    """Pair every future frame with the frame its transition starts from.

    By default that is the present frame for all K horizons, which extends
    JEPA-WAM's single `(O_t, O_{t+delta})` target. With `segment_targets` the
    start is the previous horizon instead, giving K equal-length transitions.
    """
    if current_frame_pairs.ndim != 6 or current_frame_pairs.shape[2] != 2:
        raise ValueError("current_frame_pairs must have shape [B,V,2,C,H,W]")
    if future_frames.ndim != 6:
        raise ValueError("future_frames must have shape [B,V,K,C,H,W]")
    if current_frame_pairs.shape[:2] != future_frames.shape[:2]:
        raise ValueError("current and future camera axes must match")
    if current_frame_pairs.shape[3:] != future_frames.shape[3:]:
        raise ValueError("current and future frame geometry must match")
    anchor = current_frame_pairs[:, :, 1][:, :, None]
    if segment_targets:
        # Each horizon spans from the previous one rather than from the present,
        # so every target is an equal-length step instead of a growing reach.
        # Fixed anchors make the K targets share the whole `t -> t+h_{k-1}`
        # prefix, which is redundant, and leaves the far horizons carrying much
        # larger motion than the near ones for the weights to compensate.
        previous = torch.cat((anchor, future_frames[:, :, :-1]), dim=2)
        return torch.stack((previous, future_frames), dim=3)
    return torch.stack((anchor.expand_as(future_frames), future_frames), dim=3)


def right_shift_tubelets(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Right shift complete joint-view tubelets for teacher forcing."""
    if current.ndim != 3 or future.ndim != 4:
        raise ValueError("current/future must have shapes [B,N,D] and [B,T,N,D]")
    if current.shape[0] != future.shape[0] or current.shape[1:] != future.shape[2:]:
        raise ValueError("current and future tubelet shapes do not match")
    return torch.cat((current[:, None], future[:, :-1]), dim=1)


def split_action_conditioning_groups(tokens: torch.Tensor, groups: int = 5) -> tuple[torch.Tensor, ...]:
    """Split placeholder hidden states into balanced, contiguous temporal groups."""
    if tokens.ndim != 3 or groups <= 0 or tokens.shape[1] < groups:
        raise ValueError("tokens must be [B,N,D] with at least one token per group")
    base, remainder = divmod(tokens.shape[1], groups)
    sizes = tuple(base + (1 if index < remainder else 0) for index in range(groups))
    return tuple(tokens.split(sizes, dim=1))


def build_block_causal_attention_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    n_tubelets: int,
    tokens_per_tubelet: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask with a causal prefix and block-causal future tubelets."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if n_tubelets <= 0 or tokens_per_tubelet <= 0:
        raise ValueError("tubelet counts must be positive")
    if not dtype.is_floating_point:
        raise ValueError("attention dtype must be floating point")
    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    future_length = n_tubelets * tokens_per_tubelet
    total = prefix_length + future_length
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)
    causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = causal[None] & valid[:, None, :]
    allowed[:, prefix_length:, :prefix_length] = valid[:, None, :]
    groups = torch.arange(future_length, device=device) // tokens_per_tubelet
    allowed[:, prefix_length:, prefix_length:] = (groups[:, None] >= groups[None, :])[None]
    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


def build_joint_visibility_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    n_tubelets: int,
    video_tokens_per_tubelet: int,
    action_group_sizes: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build asymmetric `[prefix | video groups | action groups]` visibility.

    Video group ``k`` sees valid prefix keys and video groups through ``k``.
    Action group ``k`` additionally sees action groups through ``k`` but never
    a later video group. Video rows never see action keys.
    """
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if n_tubelets <= 0 or video_tokens_per_tubelet <= 0:
        raise ValueError("tubelet counts must be positive")
    if len(action_group_sizes) != n_tubelets or any(size <= 0 for size in action_group_sizes):
        raise ValueError("one positive action group size is required per tubelet")
    if not dtype.is_floating_point:
        raise ValueError("attention dtype must be floating point")

    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    video_length = n_tubelets * video_tokens_per_tubelet
    action_length = sum(action_group_sizes)
    total = prefix_length + video_length + action_length
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)

    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal[None] & valid[:, None, :]

    video_start = prefix_length
    action_start = video_start + video_length
    video_groups = torch.arange(video_length, device=device) // video_tokens_per_tubelet
    video_allowed = video_groups[:, None] >= video_groups[None, :]
    allowed[:, video_start:action_start, :prefix_length] = valid[:, None, :]
    allowed[:, video_start:action_start, video_start:action_start] = video_allowed[None]

    action_groups = torch.repeat_interleave(
        torch.arange(n_tubelets, device=device),
        torch.tensor(action_group_sizes, device=device),
    )
    allowed[:, action_start:, :prefix_length] = valid[:, None, :]
    allowed[:, action_start:, video_start:action_start] = (
        action_groups[:, None] >= video_groups[None, :]
    )[None]
    allowed[:, action_start:, action_start:] = (
        action_groups[:, None] >= action_groups[None, :]
    )[None]

    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


def build_parallel_joint_visibility_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    n_horizons: int,
    video_tokens_per_horizon: int,
    action_group_sizes: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Isolate parallel horizon queries and expose each only to its action group."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if n_horizons <= 0 or video_tokens_per_horizon <= 0:
        raise ValueError("horizon counts and token counts must be positive")
    if len(action_group_sizes) != n_horizons or any(size <= 0 for size in action_group_sizes):
        raise ValueError("one positive action group size is required per horizon")
    if not dtype.is_floating_point:
        raise ValueError("attention dtype must be floating point")

    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    video_length = n_horizons * video_tokens_per_horizon
    action_length = sum(action_group_sizes)
    total = prefix_length + video_length + action_length
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)

    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal[None] & valid[:, None, :]

    video_start = prefix_length
    action_start = video_start + video_length
    video_groups = torch.arange(video_length, device=device) // video_tokens_per_horizon
    same_video_group = video_groups[:, None] == video_groups[None, :]
    allowed[:, video_start:action_start, :prefix_length] = valid[:, None, :]
    allowed[:, video_start:action_start, video_start:action_start] = same_video_group[None]

    action_groups = torch.repeat_interleave(
        torch.arange(n_horizons, device=device),
        torch.tensor(action_group_sizes, device=device),
    )
    allowed[:, action_start:, :prefix_length] = valid[:, None, :]
    allowed[:, action_start:, video_start:action_start] = (
        action_groups[:, None] == video_groups[None, :]
    )[None]
    allowed[:, action_start:, action_start:] = (
        action_groups[:, None] == action_groups[None, :]
    )[None]

    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


def build_global_action_visibility_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    n_horizons: int,
    video_tokens_per_horizon: int,
    action_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Isolate video horizons while exposing all of them to one global action memory."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if n_horizons <= 0 or video_tokens_per_horizon <= 0 or action_tokens <= 0:
        raise ValueError("horizon, video-token, and action-token counts must be positive")
    if not dtype.is_floating_point:
        raise ValueError("attention dtype must be floating point")

    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    video_length = n_horizons * video_tokens_per_horizon
    total = prefix_length + video_length + action_tokens
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)

    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal[None] & valid[:, None, :]

    video_start = prefix_length
    action_start = video_start + video_length
    video_groups = torch.arange(video_length, device=device) // video_tokens_per_horizon
    same_video_group = video_groups[:, None] == video_groups[None, :]
    allowed[:, video_start:action_start, :prefix_length] = valid[:, None, :]
    allowed[:, video_start:action_start, video_start:action_start] = same_video_group[None]

    allowed[:, action_start:, :prefix_length] = valid[:, None, :]
    allowed[:, action_start:, video_start:action_start] = True
    allowed[:, action_start:, action_start:] = True

    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


def standardized_latent_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Compare reconstructed full-resolution grids in standardized V-JEPA space."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target latent grids must have identical shapes")
    safe_std = std.to(target.device, torch.float32).clamp_min(1e-6)
    mean = mean.to(target.device, torch.float32)
    pred = (prediction.float() - mean) / safe_std
    truth = (target.detach().float() - mean) / safe_std
    return (pred - truth).square().mean()
