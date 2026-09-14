"""Fast-LeWorldModel-style action-prefix dynamics for RoboTwin V-JEPA latents."""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
import torch.nn as nn
import torch.nn.functional as F


def scale_action_conditioning_gradient(
    predicted_actions: torch.Tensor,
    *,
    gradient_scale: float,
) -> torch.Tensor:
    """Use predicted actions in forward while scaling WM gradients into the policy."""
    gradient_scale = float(gradient_scale)
    if gradient_scale < 0:
        raise ValueError("Fast-LeWM action gradient scale must be non-negative")
    detached = predicted_actions.detach()
    return detached + gradient_scale * (predicted_actions - detached)


def build_action_blocks(actions: torch.Tensor, *, num_prefixes: int) -> torch.Tensor:
    """Pack a fixed action chunk into equal consecutive blocks.

    A causal prefix encoder turns block token ``k`` into a representation of
    blocks ``0..k``; the blocks themselves are intentionally not cumulative.
    """
    if actions.ndim != 3:
        raise ValueError("actions must have shape [B,T,D]")
    if num_prefixes <= 0 or actions.shape[1] % num_prefixes:
        raise ValueError("the action horizon must divide evenly across prefixes")
    batch, horizon, action_dim = actions.shape
    block_steps = horizon // num_prefixes
    return actions.reshape(batch, num_prefixes, block_steps * action_dim)


class ActionPrefixEncoder(nn.Module):
    """Encode state-conditioned cumulative action prefixes with causal attention."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_block_dim: int,
        prefix_dim: int = 192,
        depth: int = 3,
        num_heads: int = 6,
        dropout: float = 0.0,
        max_prefixes: int = 5,
    ) -> None:
        super().__init__()
        if min(state_dim, action_block_dim, prefix_dim, depth, num_heads, max_prefixes) <= 0:
            raise ValueError("Fast-LeWM dimensions and depths must be positive")
        if prefix_dim % num_heads:
            raise ValueError("prefix_dim must be divisible by num_heads")
        self.prefix_dim = int(prefix_dim)
        self.max_prefixes = int(max_prefixes)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 4 * prefix_dim),
            nn.GELU(),
            nn.Linear(4 * prefix_dim, prefix_dim),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_block_dim),
            nn.Linear(action_block_dim, 4 * prefix_dim),
            nn.GELU(),
            nn.Linear(4 * prefix_dim, prefix_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, max_prefixes + 1, prefix_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=prefix_dim,
            nhead=num_heads,
            dim_feedforward=4 * prefix_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(prefix_dim)

    def forward(self, state: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("state must have shape [B,D]")
        if action_blocks.ndim != 3 or action_blocks.shape[0] != state.shape[0]:
            raise ValueError("action_blocks must have shape [B,K,D]")
        prefixes = action_blocks.shape[1]
        if prefixes <= 0 or prefixes > self.max_prefixes:
            raise ValueError("action prefix count exceeds configured maximum")
        state_token = self.state_projection(state).unsqueeze(1)
        action_tokens = self.action_projection(action_blocks)
        tokens = torch.cat((state_token, action_tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : prefixes + 1].to(tokens.dtype)
        causal_mask = torch.ones(
            prefixes + 1,
            prefixes + 1,
            dtype=torch.bool,
            device=tokens.device,
        ).triu(diagonal=1)
        hidden = self.encoder(tokens, mask=causal_mask)
        return self.output_norm(hidden[:, 1:])


class MultiEmbodimentActionPrefixEncoder(nn.Module):
    """One causal prefix encoder shared across robots with different action spaces.

    The world head never sees raw actions -- it only sees the fixed-width prefix
    produced here. So co-training across embodiments needs nothing more than a
    per-dataset input projection: RoboTwin's 14-D dual-arm qpos and DROID's 8-D
    Franka qpos each get their own `state`/`action` projection, while the causal
    accumulation (which is the method, not the robot) stays shared.

    `embodiments` maps a dataset name to its `(state_dim, action_block_dim)`.
    Horizons should be matched in *time* rather than step count when the source
    datasets differ in frame rate.
    """

    def __init__(
        self,
        *,
        embodiments: dict[str, tuple[int, int]],
        prefix_dim: int = 192,
        depth: int = 3,
        num_heads: int = 6,
        dropout: float = 0.0,
        max_prefixes: int = 5,
    ) -> None:
        super().__init__()
        if not embodiments:
            raise ValueError("at least one embodiment must be configured")
        if min(prefix_dim, depth, num_heads, max_prefixes) <= 0:
            raise ValueError("prefix dimensions and depths must be positive")
        if prefix_dim % num_heads:
            raise ValueError("prefix_dim must be divisible by num_heads")
        self.prefix_dim = int(prefix_dim)
        self.max_prefixes = int(max_prefixes)

        def projection(in_dim: int) -> nn.Module:
            if in_dim <= 0:
                raise ValueError(f"embodiment input dims must be positive, got {in_dim}")
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 4 * prefix_dim),
                nn.GELU(),
                nn.Linear(4 * prefix_dim, prefix_dim),
            )

        self.state_projections = nn.ModuleDict(
            {name: projection(state_dim) for name, (state_dim, _) in embodiments.items()}
        )
        self.action_projections = nn.ModuleDict(
            {name: projection(action_dim) for name, (_, action_dim) in embodiments.items()}
        )

        # Everything below is shared across embodiments.
        self.position_embedding = nn.Parameter(torch.empty(1, max_prefixes + 1, prefix_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=prefix_dim,
            nhead=num_heads,
            dim_feedforward=4 * prefix_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(prefix_dim)

    @property
    def embodiments(self) -> list[str]:
        return sorted(self.state_projections.keys())

    def forward(self, embodiment: str, state: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        if embodiment not in self.state_projections:
            raise KeyError(f"unknown embodiment {embodiment!r}; configured: {self.embodiments}")
        if state.ndim != 2:
            raise ValueError("state must have shape [B,D]")
        if action_blocks.ndim != 3 or action_blocks.shape[0] != state.shape[0]:
            raise ValueError("action_blocks must have shape [B,K,D] matching the state batch")
        prefixes = action_blocks.shape[1]
        if prefixes <= 0 or prefixes > self.max_prefixes:
            raise ValueError("action prefix count exceeds configured maximum")

        state_token = self.state_projections[embodiment](state).unsqueeze(1)
        action_tokens = self.action_projections[embodiment](action_blocks)
        tokens = torch.cat((state_token, action_tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : prefixes + 1].to(tokens.dtype)
        causal_mask = torch.ones(
            prefixes + 1, prefixes + 1, dtype=torch.bool, device=tokens.device
        ).triu(diagonal=1)
        hidden = self.encoder(tokens, mask=causal_mask)
        return self.output_norm(hidden[:, 1:])


class SharedHorizonQueryHead(nn.Module):
    """Use one Qwen visual grid as the action-conditioned forward dynamics model."""

    def __init__(
        self,
        *,
        query_dim: int,
        target_dim: int,
        prefix_dim: int,
        num_horizons: int = 5,
    ) -> None:
        super().__init__()
        if min(query_dim, target_dim, prefix_dim, num_horizons) <= 0:
            raise ValueError("shared horizon query dimensions must be positive")
        self.query_dim = int(query_dim)
        self.target_dim = int(target_dim)
        self.prefix_dim = int(prefix_dim)
        self.num_horizons = int(num_horizons)
        self.horizon_embeddings = nn.Parameter(
            torch.empty(1, self.num_horizons, self.prefix_dim)
        )
        nn.init.normal_(self.horizon_embeddings, mean=0.0, std=0.02)
        self.condition = nn.Sequential(
            nn.LayerNorm(self.prefix_dim),
            nn.Linear(self.prefix_dim, 2 * self.query_dim),
        )
        self.query_norm = nn.LayerNorm(self.query_dim, elementwise_affine=False)
        self.projection = nn.Sequential(
            nn.Linear(self.query_dim, 2 * self.target_dim),
            nn.GELU(),
            nn.Linear(2 * self.target_dim, self.target_dim),
        )

    def forward(
        self,
        shared_query: torch.Tensor,
        action_prefixes: torch.Tensor,
    ) -> torch.Tensor:
        if shared_query.ndim != 5 or shared_query.shape[-1] != self.query_dim:
            raise ValueError("shared_query must have shape [B,V,H,W,Dq]")
        if action_prefixes.shape != (
            shared_query.shape[0],
            self.num_horizons,
            self.prefix_dim,
        ):
            raise ValueError("action_prefixes must have shape [B,K,Dp]")
        condition_tokens = action_prefixes + self.horizon_embeddings.to(action_prefixes.dtype)
        scale, shift = self.condition(condition_tokens).chunk(2, dim=-1)
        scale = scale[:, None, :, None, None]
        shift = shift[:, None, :, None, None]
        query = self.query_norm(shared_query).unsqueeze(2)
        return self.projection(query * (1.0 + scale) + shift)


class _SDPASelfAttention(nn.Module):
    """Minimal self-attention that dispatches to PyTorch's fused SDPA kernels."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(
        self, tokens: torch.Tensor, *, is_causal: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # A boolean mask and `is_causal` are alternatives, never both: the pooled
        # sequence is causal across timesteps but fully connected within one, which
        # a triangular mask over the flat token axis cannot express.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None if attn_mask is None else ~attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.output(attended)


def _window_partition(tokens: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert [B,V,K,H,W,D] grids into independent window sequences."""
    batch, views, horizons, height, width, dim = tokens.shape
    if height % window_size or width % window_size:
        raise ValueError(
            f"grid {(height, width)} must be divisible by window size {window_size}"
        )
    return (
        tokens.reshape(
            batch,
            views,
            horizons,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 2, 3, 5, 4, 6, 7)
        .reshape(-1, window_size * window_size, dim)
    )


def _window_reverse(
    windows: torch.Tensor,
    *,
    shape: tuple[int, int, int, int, int, int],
    window_size: int,
) -> torch.Tensor:
    batch, views, horizons, height, width, dim = shape
    return (
        windows.reshape(
            batch,
            views,
            horizons,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 2, 3, 5, 4, 6, 7)
        .reshape(batch, views, horizons, height, width, dim)
    )


class _PrefixConditionedWindowBlock(nn.Module):
    """Spatial-window and causal-horizon attention with AdaLN-Zero conditioning."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        window_size: int,
        dropout: float,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.spatial_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.horizon_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.spatial_attention = _SDPASelfAttention(dim, num_heads, dropout)
        self.horizon_attention = _SDPASelfAttention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        # AdaLN-Zero, with the gates opened. The three gates here multiply the whole
        # sublayer output, not just the conditioning, so zero-initialising all nine
        # chunks does not merely make conditioning start as identity -- it switches
        # the transformer off and leaves the head a shallow map of its input. It
        # never recovered: gates sat at 0.04 after 15k steps on one line and at
        # 0.027 after 3k on another, so each of the 36 residual branches contributed
        # under 3% and 205M of attention and MLP weights moved 4.6% per 1500 steps
        # while every metric stayed flat. The horizon attention is gated the same
        # way, which is the only path by which the five horizons interact at all.
        #
        # DiT gets away with this because its condition -- timestep and class -- is
        # informative from step one, so the gates have something to open for. Here
        # the condition is an action prefix that barely varies across samples or
        # horizons, so the gates have no reason to leave zero.
        #
        # Zeroing the weight still makes conditioning start as identity (shift and
        # scale are zero, so the modulation is the identity); a unit bias on the
        # gate chunks starts the sublayers at full strength instead of off.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        if gate_bias_init:
            with torch.no_grad():
                bias = self.adaLN_modulation[-1].bias
                for chunk in (2, 5, 8):  # gate_spatial, gate_horizon, gate_mlp
                    bias[chunk * dim : (chunk + 1) * dim].fill_(float(gate_bias_init))

    @staticmethod
    def _modulate(
        tokens: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return tokens * (1.0 + scale) + shift

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 6 or condition.ndim != 3:
            raise ValueError("window block expects [B,V,K,H,W,D] tokens and [B,K,D] condition")
        batch, views, horizons, height, width, dim = tokens.shape
        if condition.shape != (batch, horizons, dim):
            raise ValueError("window block condition shape does not match tokens")
        modulation = self.adaLN_modulation(condition)
        (
            shift_spatial,
            scale_spatial,
            gate_spatial,
            shift_horizon,
            scale_horizon,
            gate_horizon,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(9, dim=-1)

        def broadcast(value: torch.Tensor) -> torch.Tensor:
            return value[:, None, :, None, None]

        spatial = self._modulate(
            self.spatial_norm(tokens),
            broadcast(shift_spatial),
            broadcast(scale_spatial),
        )
        spatial_windows = _window_partition(spatial, self.window_size)
        spatial_windows = self.spatial_attention(spatial_windows)
        spatial = _window_reverse(
            spatial_windows,
            shape=(batch, views, horizons, height, width, dim),
            window_size=self.window_size,
        )
        tokens = tokens + broadcast(gate_spatial) * spatial

        horizon = self._modulate(
            self.horizon_norm(tokens),
            broadcast(shift_horizon),
            broadcast(scale_horizon),
        )
        horizon = horizon.permute(0, 1, 3, 4, 2, 5).reshape(
            batch * views * height * width, horizons, dim
        )
        horizon = self.horizon_attention(horizon, is_causal=True)
        horizon = horizon.reshape(batch, views, height, width, horizons, dim).permute(
            0, 1, 4, 2, 3, 5
        )
        tokens = tokens + broadcast(gate_horizon) * horizon

        mlp_output = self.mlp(
            self._modulate(
                self.mlp_norm(tokens),
                broadcast(shift_mlp),
                broadcast(scale_mlp),
            )
        )
        return tokens + broadcast(gate_mlp) * mlp_output


class PrefixConditionedWindowTransformerHead(nn.Module):
    """Parallel dense Fast-LeWM predictor with fused window/horizon attention."""

    def __init__(
        self,
        *,
        query_dim: int,
        target_dim: int,
        prefix_dim: int,
        model_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        window_sizes: tuple[int, ...] = (8, 6),
        num_horizons: int = 5,
        max_views: int = 3,
        dropout: float = 0.1,
        additive_conditioning: bool = True,
        horizon_embedding_std: float = 0.02,
        gradient_checkpointing: bool = False,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        dimensions = (
            query_dim,
            target_dim,
            prefix_dim,
            model_dim,
            depth,
            num_heads,
            mlp_dim,
            num_horizons,
            max_views,
        )
        if min(dimensions) <= 0 or not window_sizes or min(window_sizes) <= 0:
            raise ValueError("window Transformer dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.query_dim = int(query_dim)
        self.target_dim = int(target_dim)
        self.prefix_dim = int(prefix_dim)
        self.model_dim = int(model_dim)
        self.depth = int(depth)
        self.num_horizons = int(num_horizons)
        self.max_views = int(max_views)
        self.window_sizes = tuple(int(size) for size in window_sizes)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, model_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim),
            nn.Linear(prefix_dim, model_dim),
        )
        self.horizon_embeddings = nn.Parameter(
            torch.empty(1, num_horizons, model_dim)
        )
        # A second, additive route for the conditioning. Everything distinguishing
        # one horizon or one action from another used to reach the tokens only
        # through the AdaLN gates, which start at zero -- and measured after 15k
        # steps they sat at ~0.04 and had stopped growing, leaving the head a linear
        # map of the current frame that ignored both. Adding the condition into the
        # tokens gives that information a path which cannot be gated off, so the
        # horizons differ from the first step and the gates are left to modulate
        # rather than to carry.
        self.additive_conditioning = bool(additive_conditioning)
        self.view_embeddings = nn.Parameter(torch.empty(1, max_views, 1, 1, 1, model_dim))
        # The five horizons share one copy of the visual tokens, so this embedding
        # plus the action prefix is the *only* thing telling them apart. At the
        # 0.02 default it lands at ~2.5% of the conditioning magnitude, small
        # enough that weight decay outpaced its gradient and it never grew --
        # measured at std 0.0197 after 5000 steps. Scale it to the conditioning
        # it is added to and the horizons differ from step one.
        if float(horizon_embedding_std) <= 0:
            raise ValueError("horizon_embedding_std must be positive")
        self.horizon_embedding_std = float(horizon_embedding_std)
        nn.init.normal_(self.horizon_embeddings, std=self.horizon_embedding_std)
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            _PrefixConditionedWindowBlock(
                dim=model_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                window_size=self.window_sizes[index % len(self.window_sizes)],
                dropout=dropout,
                gate_bias_init=gate_bias_init,
            )
            for index in range(depth)
        )
        self.view_norm = nn.LayerNorm(model_dim)
        self.view_attention = _SDPASelfAttention(model_dim, num_heads, dropout)
        self.view_gate = nn.Parameter(torch.zeros(1, 1, num_horizons, 1, 1, model_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, target_dim),
        )
        nn.init.normal_(self.output[-1].weight, std=0.02)
        nn.init.zeros_(self.output[-1].bias)
        # Each block holds activations for V*K*H*W tokens (8640 per sample at three
        # views and five horizons), so a deep head is activation-bound long before
        # it is parameter-bound. Recomputing them in backward trades ~30% step time
        # for roughly the whole activation budget.
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gate_bias_init = float(gate_bias_init)

    def forward(
        self,
        shared_query: torch.Tensor,
        action_prefixes: torch.Tensor,
    ) -> torch.Tensor:
        if shared_query.ndim != 5 or shared_query.shape[-1] != self.query_dim:
            raise ValueError("shared_query must have shape [B,V,H,W,Dq]")
        batch, views, height, width, _ = shared_query.shape
        if views > self.max_views:
            raise ValueError("shared_query view count exceeds max_views")
        if action_prefixes.shape != (batch, self.num_horizons, self.prefix_dim):
            raise ValueError("action_prefixes must have shape [B,K,Dp]")
        for window_size in self.window_sizes:
            if height % window_size or width % window_size:
                raise ValueError(
                    f"grid {(height, width)} must be divisible by every window size {self.window_sizes}"
                )

        condition = self.condition_projection(action_prefixes)
        condition = condition + self.horizon_embeddings.to(condition.dtype)
        tokens = self.query_projection(shared_query)
        tokens = tokens[:, :, None].expand(-1, -1, self.num_horizons, -1, -1, -1)
        tokens = tokens + self.view_embeddings[:, :views].to(tokens.dtype)
        if self.additive_conditioning:
            # [B,K,D] -> [B,1,K,1,1,D]: same condition for every view and patch,
            # different per horizon, so horizons are distinguishable before block 0.
            tokens = tokens + condition[:, None, :, None, None].to(tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = gradient_checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)

        pooled_views = self.view_norm(tokens.mean(dim=(3, 4)))
        pooled_views = pooled_views.permute(0, 2, 1, 3).reshape(
            batch * self.num_horizons, views, self.model_dim
        )
        pooled_views = self.view_attention(pooled_views)
        pooled_views = pooled_views.reshape(
            batch, self.num_horizons, views, self.model_dim
        ).permute(0, 2, 1, 3)[:, :, :, None, None]
        tokens = tokens + self.view_gate * pooled_views
        return self.output(tokens)


def horizon_weighted_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    horizon_weights: tuple[float, ...],
) -> torch.Tensor:
    """Align query dynamics with frozen GT V-JEPA targets at each horizon."""
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError("prediction and target must share shape [B,V,K,H,W,D]")
    if len(horizon_weights) != prediction.shape[2] or any(weight <= 0 for weight in horizon_weights):
        raise ValueError("horizon_weights must contain one positive value per horizon")
    cosine_error = 1.0 - F.cosine_similarity(
        prediction.float(),
        target.detach().float(),
        dim=-1,
    )
    weights = torch.as_tensor(
        horizon_weights,
        dtype=cosine_error.dtype,
        device=cosine_error.device,
    ).view(1, 1, -1, 1, 1)
    weights = weights.expand_as(cosine_error)
    return (cosine_error * weights).sum() / weights.sum().clamp_min(1e-6)


def build_action_conditioned_dynamics_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    n_horizons: int,
    video_tokens_per_horizon: int,
    action_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Qwen-internal, action-conditioned, horizon-parallel forward dynamics.

    Sequence layout: [prefix | action_gen | video_horizon_0..K-1]. No ground-truth
    action values are fed as input anywhere -- action-conditioning comes from
    letting horizon k's video tokens attend to the *hidden states* of the
    action-generation placeholders (position-split into K causally-cumulative
    sub-groups) before those placeholders are ever decoded into real numbers by
    the action head. This works identically at train and inference time (no
    future ground truth required), unlike feeding real action values in.

    Visibility:
      - action-gen placeholders: see the prefix and *all* of each other (the
        flow-matching action head still consumes them as one joint block --
        internal causal staging is not imposed here, only which video horizon
        may look at which sub-group);
      - video horizon k: sees the prefix, action-gen sub-groups 0..k only
        (cumulative -- "how much of the action decision has committed by
        horizon k"), and its own horizon group -- never another horizon's
        video tokens (all five are still predicted in parallel, not chained).
    """
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if n_horizons <= 0 or video_tokens_per_horizon <= 0 or action_tokens <= 0:
        raise ValueError("horizon, video-token, and action-token counts must be positive")
    if not dtype.is_floating_point:
        raise ValueError("attention dtype must be floating point")

    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    video_length = n_horizons * video_tokens_per_horizon
    action_gen_start = prefix_length
    video_start = action_gen_start + action_tokens
    total = video_start + video_length
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)

    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal[None] & valid[:, None, :]

    allowed[:, action_gen_start:video_start, :prefix_length] = valid[:, None, :]
    allowed[:, action_gen_start:video_start, action_gen_start:video_start] = True

    video_groups = torch.arange(video_length, device=device) // video_tokens_per_horizon
    same_video_group = video_groups[:, None] == video_groups[None, :]
    allowed[:, video_start:, :prefix_length] = valid[:, None, :]
    allowed[:, video_start:, video_start:] = same_video_group[None]

    chunk_bounds = [round(i * action_tokens / n_horizons) for i in range(n_horizons + 1)]
    for horizon in range(n_horizons):
        row_lo = video_start + horizon * video_tokens_per_horizon
        row_hi = row_lo + video_tokens_per_horizon
        col_hi = action_gen_start + chunk_bounds[horizon + 1]
        allowed[:, row_lo:row_hi, action_gen_start:col_hi] = True

    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


def build_present_action_visibility_mask(
    prefix_attention_mask: torch.Tensor,
    *,
    action_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Let action queries read present context and each other, never future tokens."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must have shape [B,P]")
    if action_tokens <= 0 or not dtype.is_floating_point:
        raise ValueError("action_tokens must be positive and dtype floating point")
    valid = prefix_attention_mask.bool()
    batch, prefix_length = valid.shape
    total = prefix_length + action_tokens
    device = valid.device
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=device)
    prefix_causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device).tril()
    allowed[:, :prefix_length, :prefix_length] = prefix_causal[None] & valid[:, None, :]
    allowed[:, prefix_length:, :prefix_length] = valid[:, None, :]
    allowed[:, prefix_length:, prefix_length:] = True
    additive = torch.full(
        (batch, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(allowed[:, None], 0.0)


class LatentPooler(nn.Module):
    """Pool a patch grid into one compact latent per observation.

    Fast-LeWM predicts a single latent vector per timestep; this head was pointed
    at a 3 x 576 x 1024 patch grid instead, and that difference turned out to
    matter more than any of the architecture knobs. A ridge probe fitted from the
    current frame's own features recovered 69% of the patch-level target's
    direction *without seeing the future at all* -- the residual there is
    dominated by a systematic per-patch transformation of the current frame, so
    most of the supervision carried no information about what happens next.
    Averaging that away is the point of pooling, not an efficiency measure.

    Attention pooling rather than a mean: one learned query per view weighs the
    576 patches, which keeps the arm and the objects from being averaged into the
    table.
    """

    def __init__(
        self, *, patch_dim: int, views: int = 3, latent_dim: int = 256,
        num_queries: int = 1,
    ) -> None:
        super().__init__()
        if min(patch_dim, views, latent_dim, num_queries) <= 0:
            raise ValueError("LatentPooler dimensions must be positive")
        self.patch_dim = int(patch_dim)
        self.views = int(views)
        self.latent_dim = int(latent_dim)
        # More than one query when a sequence model reads the result: a single
        # vector per timestep leaves a transformer six tokens to attend over,
        # which is not a sequence. Each query is free to settle on a different
        # part of the scene.
        self.num_queries = int(num_queries)
        self.query = nn.Parameter(
            torch.randn(views, num_queries, patch_dim) * patch_dim**-0.5
        )
        self.key = nn.Linear(patch_dim, patch_dim, bias=False)
        self.value = nn.Linear(patch_dim, patch_dim, bias=False)
        # The trailing LayerNorm is not cosmetic. The target latent is produced by
        # this same module and then detached, so the gradient cannot see what the
        # pooler does to the target -- and with the scale left free it drifted: the
        # target's motion RMS fell 0.153 -> 0.124 over 200 steps, the same
        # shrink-the-target failure the patch-level path had. Fast-LeWM pins the
        # latent distribution with SIGReg on Z; at eight samples per device against
        # a 256-d latent a distributional test has nothing to work with, so the
        # scale is fixed by construction instead. No learnable affine: that would
        # hand the drift straight back.
        self.projection = nn.Sequential(
            nn.LayerNorm(views * patch_dim),
            nn.Linear(views * patch_dim, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """`[B,V,P,D]` patch tokens -> `[B, latent_dim]`."""
        if patches.ndim != 4 or patches.shape[-1] != self.patch_dim:
            raise ValueError("LatentPooler expects [B,V,P,D] patch tokens")
        views = patches.shape[1]
        if views > self.views:
            raise ValueError("patch grid has more views than the pooler was built for")
        key, value = self.key(patches), self.value(patches)
        weights = torch.einsum(
            "vqd,bvpd->bvqp", self.query[:views].to(key.dtype), key
        ) * self.patch_dim**-0.5
        pooled = torch.einsum("bvqp,bvpd->bvqd", weights.softmax(dim=-1), value)
        # [B,V,Q,D] -> [B,Q,V*D] -> [B,Q,latent]; squeezed to [B,latent] when a
        # single query is asked for, so the original callers are unchanged.
        pooled = pooled.permute(0, 2, 1, 3).reshape(
            patches.shape[0], self.num_queries, views * self.patch_dim
        )
        latents = self.projection(pooled)
        return latents.squeeze(1) if self.num_queries == 1 else latents


class _ActionModulatedResidualLayer(nn.Module):
    """One residual MLP layer whose shift/scale/gate come from the action prefix."""

    def __init__(
        self, *, latent_dim: int, hidden_dim: int, fusion_dim: int,
        dropout: float, modulation_weight_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(fusion_dim, 3 * latent_dim))
        # Neither half of AdaLN-Zero survives contact with this task.
        #
        # A zero *bias* on the gates switches the sublayer off, not just the
        # conditioning: measured on the window head the gates sat at 0.027 after
        # 3000 steps, so twelve layers and 205M parameters contributed under 3%.
        # A unit gate bias starts them at full strength instead.
        #
        # A zero *weight* makes the condition -- action prefix and horizon
        # embedding alike -- have no effect whatsoever at initialization, so the
        # five horizons begin identical and only a gradient through an all-zero
        # map can separate them. On the window head that gradient never arrived:
        # the horizon embedding moved 0.208% in 1500 steps, a hundred and twenty
        # times less than a consistent gradient would give, and the five horizons
        # ended at cosine 0.9975 against targets differing at 0.9080. A small
        # normal init lets the condition act from step one.
        nn.init.normal_(self.modulation[-1].weight, std=modulation_weight_std)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():
            self.modulation[-1].bias[2 * latent_dim :].fill_(1.0)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return latent + gate * self.mlp(self.norm(latent) * (1.0 + scale) + shift)


class PooledActionPrefixPredictor(nn.Module):
    """Fast-LeWM's predictor: a shallow action-modulated residual MLP on one latent.

    Every horizon reads the same current latent and its own action prefix, so the
    five predictions are produced by five separate passes through the same layers
    rather than by broadcasting one token stream and nudging it with a per-horizon
    bias -- which is why the window head's five horizons agreed at cosine 0.9975
    against targets that differ at 0.9080.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        prefix_dim: int,
        depth: int = 6,
        hidden_dim: int = 2048,
        fusion_dim: int = 768,
        num_horizons: int = 5,
        dropout: float = 0.1,
        modulation_weight_std: float = 0.02,
    ) -> None:
        super().__init__()
        if min(latent_dim, prefix_dim, depth, hidden_dim, fusion_dim, num_horizons) <= 0:
            raise ValueError("predictor dimensions and depths must be positive")
        self.latent_dim = int(latent_dim)
        self.num_horizons = int(num_horizons)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, fusion_dim)
        )
        self.horizon_embeddings = nn.Parameter(torch.randn(1, num_horizons, fusion_dim) * 0.3)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim)
        )
        self.layers = nn.ModuleList(
            _ActionModulatedResidualLayer(
                latent_dim=latent_dim, hidden_dim=hidden_dim,
                fusion_dim=fusion_dim, dropout=dropout,
                modulation_weight_std=modulation_weight_std,
            )
            for _ in range(depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))
        # Anchored on the current latent, which is what the paper describes:
        # "Fast-LeWM anchors every prediction at the current observed latent z_t".
        # Reading z_t through one random projection and writing the answer through
        # another leaves the prediction uncorrelated with z_t at initialization --
        # measured, that is an MSE of 0.37 where simply echoing z_t scores 0.015,
        # and after 200 steps the predictor had not yet clawed its way back to the
        # echo. Zero-initialising the delta means it starts exactly at the echo and
        # spends its capacity on the change instead.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, latent: torch.Tensor, action_prefixes: torch.Tensor) -> torch.Tensor:
        """`[B,D]` current latent and `[B,K,P]` prefixes -> `[B,K,D]` future latents."""
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("predictor expects a [B, latent_dim] current latent")
        if action_prefixes.ndim != 3 or action_prefixes.shape[1] != self.num_horizons:
            raise ValueError("predictor expects [B, num_horizons, prefix_dim] prefixes")
        condition = self.condition_projection(action_prefixes)
        condition = condition + self.horizon_embeddings.to(condition.dtype)
        anchor = latent[:, None].expand(-1, self.num_horizons, -1)
        hidden = self.input_projection(latent)[:, None].expand(-1, self.num_horizons, -1)
        for layer in self.layers:
            hidden = layer(hidden, condition)
        return anchor + self.output(hidden)


def pooled_latent_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    *,
    per_sample_normalization: bool = False,
    directional: bool = False,
    centre_motion: bool = True,
    magnitude_weight: float = 0.1,
    live_target: torch.Tensor | None = None,
    floor_fraction: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Fast-LeWM's objective: squared error against the raw future latent.

    Accepts one latent per timestep `[B,K,D]` or several pooled tokens per
    timestep `[B,K,Q,D]`.

    `per_sample_normalization` divides each sample's error by how far that sample
    actually moved. A plain MSE weights samples by motion energy, and across
    embodiments that is not a detail: the co-training streams move the latent
    about 0.85 against RoboTwin's 0.157, so squared and taken at a 90/10 sampling
    ratio, RoboTwin was contributing 0.4% of this loss and the model predicted it
    worse than echoing the present.

    `live_target` is the same future latent still attached to the graph, and it is
    what the divisor is computed from. Detaching the divisor -- the first attempt --
    turns this term into a collapse *accelerator* rather than a defence: the
    numerator can be reduced by dragging the current latent onto the future one,
    the detached divisor does not grow back to cancel that, and the smaller it gets
    the harder the remaining gradient pushes. Measured, the pooled target's motion
    fell 0.76 to 0.02 and the gradient norm reached 1e22 before going infinite. With
    the divisor live, a collapse shrinks numerator and divisor alike and buys
    nothing, and there is mild pressure the other way.
    """
    if prediction.shape != target.shape or prediction.ndim not in (3, 4):
        raise ValueError("prediction and target must both be [B,K,D] or [B,K,Q,D]")
    detached_target = target.detach()
    anchor = current.detach().float()
    anchor = anchor[:, None] if anchor.ndim == prediction.ndim - 1 else anchor
    target_motion = detached_target.float() - anchor
    squared_error = (prediction.float() - detached_target.float()).pow(2)
    if directional:
        # Predict which way the latent moves, not how far. Every scaling of the
        # objective tried before had an exploitable direction: a plain MSE weights
        # samples by motion energy, so RoboTwin contributed 0.4% of it against the
        # co-training streams and was never learned; dividing by a *detached*
        # per-sample energy rewards collapsing the motion and amplifies its own
        # gradient as it shrinks, which ran away to a gradient norm of 1e22; and
        # dividing by a *live* energy rewards the opposite, which is how the pooler
        # ended up attending to the single most volatile patch in the image.
        #
        # A cosine has no scale to exploit in either direction, and it is already
        # the quantity used to judge whether this head is learning anything. The
        # magnitude term is small and keeps the prediction from being a direction
        # with an arbitrary length.
        predicted_motion = prediction.float() - anchor
        # Centred across the batch before the cosine, because the raw motion is
        # dominated by a direction every clip shares: measured at initialisation,
        # before any training, different clips' latent motions already agreed at
        # cosine 0.64, and ten steps of training took that to 0.97. A model that
        # emits only the common direction scores 0.95 on the uncentred cosine and
        # has predicted nothing about *this* clip. Subtracting the batch mean asks
        # the only question worth asking -- how does this clip move differently
        # from the average one -- and the shared component earns nothing.
        if centre_motion and target_motion.shape[0] > 1:
            scored_target = target_motion - target_motion.mean(dim=0, keepdim=True)
            scored_prediction = predicted_motion - predicted_motion.mean(dim=0, keepdim=True)
        else:
            scored_target, scored_prediction = target_motion, predicted_motion
        cosine_loss = (
            1.0 - F.cosine_similarity(scored_prediction, scored_target, dim=-1)
        ).mean()
        reduce_over = tuple(range(1, target_motion.ndim))
        predicted_scale = predicted_motion.pow(2).mean(dim=reduce_over).clamp_min(1e-8).sqrt()
        target_scale = target_motion.pow(2).mean(dim=reduce_over).clamp_min(1e-8).sqrt()
        # On the log ratio, and Huber rather than squared. A plain (ratio - 1)^2 is
        # unbounded above, and the first step after swapping the pooler -- where the
        # inherited predictor's delta was ~1600x the new target's motion -- put this
        # term at 2.6e5. Logs make it symmetric between over- and under-shooting,
        # and the Huber tail keeps a bad first step from producing a ruinous
        # gradient into weights that took 7000 steps to train.
        log_ratio = predicted_scale.log() - target_scale.log()
        magnitude_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio))
        loss = cosine_loss + magnitude_weight * magnitude_loss
    elif per_sample_normalization:
        divisor_source = target if live_target is None else live_target
        live_motion = divisor_source.float() - (
            current.float()[:, None] if current.ndim == prediction.ndim - 1 else current.float()
        )
        reduce_over = tuple(range(1, live_motion.ndim))
        energy = live_motion.pow(2).mean(dim=reduce_over, keepdim=True)
        # A batch-relative floor bounds one still sample's share; it cannot save a
        # batch that collapses as a whole, which is why the divisor is live.
        floor = floor_fraction * energy.mean().detach().clamp_min(1e-8)
        loss = (squared_error / energy.clamp_min(floor)).mean()
    else:
        loss = squared_error.mean()
    with torch.no_grad():
        motion_rms = torch.linalg.vector_norm(target_motion) / math.sqrt(target_motion.numel())
        cosine = F.cosine_similarity(
            prediction.float() - anchor, target_motion, dim=-1
        ).mean()
        # How alike the *targets* are across samples, which is the degenerate
        # solution a cosine objective leaves open: if the encoder sends every clip's
        # latent off in the same direction, the predictor scores ~1 by emitting that
        # one direction and has learned nothing. Healthy is near zero -- different
        # clips move differently -- and the residual cosine only means something
        # while this stays low.
        flat = target_motion.reshape(target_motion.shape[0], -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gram = flat @ flat.T
        off_diagonal = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
        motion_agreement = gram[off_diagonal].mean() if gram.shape[0] > 1 else gram.new_zeros(())
        # The same number after removing the shared direction. This is what the
        # centred objective actually scores against, so it is the one that says
        # whether anything clip-specific is left to predict.
        if target_motion.shape[0] > 1:
            centred = target_motion - target_motion.mean(dim=0, keepdim=True)
            centred = centred.reshape(centred.shape[0], -1)
            centred = centred / centred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            centred_gram = centred @ centred.T
            centred_agreement = centred_gram[off_diagonal].mean()
        else:
            centred_agreement = motion_agreement.new_zeros(())
    return {
        "latent_mse": loss,
        "target_motion_rms": motion_rms,
        "residual_cosine": cosine,
        "motion_agreement": motion_agreement,
        "centred_agreement": centred_agreement,
    }


class PooledTransformerPredictor(nn.Module):
    """A sequence predictor over pooled latent tokens, anchored on the present.

    LeWM's predictor is a transformer; Fast-LeWM's prefix variant uses a residual
    MLP because its latent is a single vector. With several pooled tokens per
    timestep there is a real sequence to attend over -- `num_queries` scene tokens
    for the current observation and one horizon's worth for each future step --
    so this reads them jointly instead of pushing one vector through an MLP.

    Attention is causal along the horizon axis: horizon k may look at the present
    and at every earlier horizon, never at a later one. Every token also sees the
    whole scene at its own timestep, which is what the per-timestep MLP could not
    do at all.

    Two things are kept from the MLP version because they were measured, not
    assumed. The prediction is the current latent plus a zero-initialised delta:
    reading the present through one random projection and writing the answer
    through another leaves the output uncorrelated with it, worth an MSE of 0.37
    where echoing scores 0.015. And the action conditioning is modulated in with a
    small non-zero init, because a zero-initialised modulation makes the action and
    the horizon inert at step one, which is how the previous head's five horizons
    ended up agreeing at cosine 0.9975.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        prefix_dim: int,
        num_queries: int = 16,
        model_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        num_horizons: int = 5,
        dropout: float = 0.1,
        modulation_weight_std: float = 0.02,
    ) -> None:
        super().__init__()
        if min(latent_dim, prefix_dim, num_queries, model_dim, depth, num_heads,
               mlp_dim, num_horizons) <= 0:
            raise ValueError("predictor dimensions and depths must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.latent_dim = int(latent_dim)
        self.num_queries = int(num_queries)
        self.num_horizons = int(num_horizons)
        self.model_dim = int(model_dim)

        self.input_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim)
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, model_dim)
        )
        # Timestep 0 is the present; 1..K are the horizons.
        self.timestep_embeddings = nn.Parameter(
            torch.randn(1, num_horizons + 1, 1, model_dim) * 0.3
        )
        self.query_embeddings = nn.Parameter(
            torch.randn(1, 1, num_queries, model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            _ActionModulatedTransformerBlock(
                model_dim=model_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                dropout=dropout, modulation_weight_std=modulation_weight_std,
            )
            for _ in range(depth)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer(
            "attention_mask", self._causal_horizon_mask(num_horizons + 1, num_queries),
            persistent=False,
        )

    @staticmethod
    def _causal_horizon_mask(timesteps: int, queries: int) -> torch.Tensor:
        """True where attention is forbidden: any token at a later timestep."""
        step = torch.arange(timesteps).repeat_interleave(queries)
        return step[None, :] > step[:, None]

    def forward(self, latent: torch.Tensor, action_prefixes: torch.Tensor) -> torch.Tensor:
        """`[B,Q,D]` current tokens and `[B,K,P]` prefixes -> `[B,K,Q,D]` futures."""
        if latent.ndim == 2:
            latent = latent[:, None]
        if latent.shape[1] != self.num_queries or latent.shape[-1] != self.latent_dim:
            raise ValueError("predictor expects [B, num_queries, latent_dim] tokens")
        if action_prefixes.shape[1] != self.num_horizons:
            raise ValueError("predictor expects one action prefix per horizon")
        batch = latent.shape[0]

        tokens = self.input_projection(latent)[:, None]                    # [B,1,Q,M]
        tokens = tokens.expand(-1, self.num_horizons + 1, -1, -1)
        tokens = tokens + self.timestep_embeddings + self.query_embeddings

        # The present carries no action; horizon k carries the prefix of the first
        # k action blocks, which is the unit Fast-LeWM predicts from.
        condition = self.condition_projection(action_prefixes)             # [B,K,M]
        condition = torch.cat((torch.zeros_like(condition[:, :1]), condition), dim=1)

        hidden = tokens.reshape(batch, (self.num_horizons + 1) * self.num_queries, -1)
        for block in self.blocks:
            hidden = block(hidden, condition, self.attention_mask, self.num_queries)
        hidden = hidden.reshape(batch, self.num_horizons + 1, self.num_queries, -1)
        return latent[:, None] + self.output(hidden[:, 1:])


class _ActionModulatedTransformerBlock(nn.Module):
    """Self-attention and MLP, both modulated by the timestep's action prefix."""

    def __init__(
        self, *, model_dim: int, num_heads: int, mlp_dim: int,
        dropout: float, modulation_weight_std: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attention = _SDPASelfAttention(model_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, model_dim), nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 6 * model_dim))
        nn.init.normal_(self.modulation[-1].weight, std=modulation_weight_std)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():  # unit gates: the sublayers start on, conditioning at identity
            self.modulation[-1].bias[2 * model_dim : 3 * model_dim].fill_(1.0)
            self.modulation[-1].bias[5 * model_dim :].fill_(1.0)

    def forward(self, tokens, condition, attention_mask, num_queries):
        modulation = self.modulation(condition).repeat_interleave(num_queries, dim=1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, dim=-1)
        attended = self.attention_norm(tokens) * (1 + scale_a) + shift_a
        tokens = tokens + gate_a * self.attention(attended, attn_mask=attention_mask)
        mlp_input = self.mlp_norm(tokens) * (1 + scale_m) + shift_m
        return tokens + gate_m * self.mlp(mlp_input)


class _CrossAttention(nn.Module):
    """Multi-head attention from a small query set onto a large token set."""

    def __init__(self, *, dim: int, num_heads: int, dropout: float, competitive: bool = False) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads, self.head_dim = int(num_heads), dim // int(num_heads)
        self.to_query = nn.Linear(dim, dim, bias=False)
        self.to_key = nn.Linear(dim, dim, bias=False)
        self.to_value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.dropout = float(dropout)
        self.competitive = bool(competitive)

    def attention(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Attention weights `[B, H, Q, P]`, normalised over patches either way.

        Ordinary cross-attention softmaxes over the patch axis, so each query picks
        its patches independently of the others and nothing stops all of them
        picking the same one. That is what happened: sixteen queries put their mass
        on a single patch out of 576 and their outputs had pairwise cosine 0.99993,
        with the pooled latent's effective rank at 1.06 of 256 while the encoder
        behind it held 780.

        Slot Attention's normalisation instead makes the queries compete: softmax
        over the *query* axis, so the queries divide each patch between them and one
        taking a patch leaves less of it for the rest. Dividing by the patch-axis sum
        afterwards turns the assignment back into a weighted mean, so a query that
        wins few patches still reads them at full strength rather than being starved.

        This constrains which query explains which patch, not how much distinct
        content there is to explain, so it closes the structural hole rather than
        guaranteeing a high rank.
        """
        shape = lambda x: x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        query, key = shape(self.to_query(queries)), shape(self.to_key(tokens))
        logits = query @ key.transpose(-1, -2) * self.head_dim**-0.5
        if not self.competitive:
            return logits.softmax(dim=-1)
        assignment = logits.softmax(dim=-2)
        return assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_queries, dim = queries.shape
        shape = lambda x: x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        if not self.competitive:
            attended = F.scaled_dot_product_attention(
                shape(self.to_query(queries)), shape(self.to_key(tokens)), shape(self.to_value(tokens)),
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            weights = self.attention(queries, tokens)
            weights = F.dropout(weights, self.dropout, training=self.training)
            attended = weights @ shape(self.to_value(tokens))
        return self.output(attended.transpose(1, 2).reshape(batch, num_queries, dim))


class _SelfAttentionBlock(nn.Module):
    """Self-attention and an MLP over the query set itself."""

    def __init__(self, *, dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = _SDPASelfAttention(dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim)
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        queries = queries + self.attention(self.norm(queries))
        return queries + self.mlp(self.mlp_norm(queries))


class PerceiverLatentPooler(nn.Module):
    """Resample a patch grid onto a small set of latent tokens.

    The single-head, single-layer pooler this replaces degenerated completely:
    its attention became a one-hot on one patch out of 576, and all sixteen
    queries picked the *same* patch -- pairwise cosine 0.99997. Sixteen pooled
    tokens were one vector copied sixteen times, reading one square of the image.

    Two structural reasons, both addressed here. A single softmax head over 576
    patches makes winner-take-all cheap, so the cross-attention is multi-head.
    And nothing made the queries specialise -- they were initialised by tiling one
    trained query and never separated -- so they now attend to each other between
    cross-attention layers, which is what lets them divide the scene up.

    The incentive that drove them onto the most volatile patch lives in the loss,
    not here; see `pooled_latent_losses`.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        views: int = 3,
        latent_dim: int = 256,
        num_queries: int = 16,
        depth: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
        competitive: bool = False,
    ) -> None:
        super().__init__()
        if min(patch_dim, views, latent_dim, num_queries, depth, num_heads, mlp_dim) <= 0:
            raise ValueError("PerceiverLatentPooler dimensions must be positive")
        self.patch_dim, self.views = int(patch_dim), int(views)
        self.latent_dim, self.num_queries = int(latent_dim), int(num_queries)
        self.queries = nn.Parameter(torch.randn(1, num_queries, patch_dim) * patch_dim**-0.5)
        self.view_embeddings = nn.Parameter(torch.randn(1, views, 1, patch_dim) * 0.02)
        self.token_norm = nn.LayerNorm(patch_dim)
        self.query_norm = nn.LayerNorm(patch_dim)
        self.cross = nn.ModuleList(
            _CrossAttention(dim=patch_dim, num_heads=num_heads, dropout=dropout,
                            competitive=competitive)
            for _ in range(depth)
        )
        self.blocks = nn.ModuleList(
            _SelfAttentionBlock(dim=patch_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
        )

    def attention_maps(self, patches: torch.Tensor) -> torch.Tensor:
        """`[B,Q,V*P]` first-layer attention, averaged over heads -- for diagnosis."""
        tokens, queries = self._prepare(patches)
        # Through the same code path as forward, so the diagnosis cannot silently
        # describe a different normalisation from the one being trained.
        return self.cross[0].attention(queries, tokens).mean(dim=1)

    def _prepare(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 4 or patches.shape[-1] != self.patch_dim:
            raise ValueError("pooler expects [B,V,P,D] patch tokens")
        batch, views = patches.shape[0], patches.shape[1]
        if views > self.views:
            raise ValueError("patch grid has more views than the pooler was built for")
        tokens = patches + self.view_embeddings[:, :views].to(patches.dtype)
        tokens = self.token_norm(tokens.reshape(batch, -1, self.patch_dim))
        queries = self.query_norm(self.queries.to(patches.dtype)).expand(batch, -1, -1)
        return tokens, queries

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        tokens, queries = self._prepare(patches)
        for cross, block in zip(self.cross, self.blocks):
            queries = block(queries + cross(queries, tokens))
        latents = self.projection(queries)
        return latents.squeeze(1) if self.num_queries == 1 else latents
