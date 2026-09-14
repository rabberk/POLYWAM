"""Auxiliary dense prediction without changing the released policy's token graph.

The head consumes final Qwen visual-position hidden states, independently per
camera. Its sole target is the SAME detached (t, t+31) V-JEPA grid used by the
released cosine head. No action labels, new Qwen tokens, or custom masks enter
the policy forward. This is not the five-horizon Fast-LeWM forward path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class OfficialLatentHead(nn.Module):
    def __init__(self, d_llm, d_target, dim=512, depth=6, heads=8,
                 window_sizes=(8, 6), gradient_checkpointing=True):
        super().__init__()
        self.input = nn.Linear(d_llm, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_target))
        self.window_sizes = tuple(window_sizes)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden, target, num_views):
        if hidden.ndim != 3 or target.shape[:2] != hidden.shape[:2]:
            raise ValueError('Expected matching [batch, view*patch, channels] tensors')
        b, n, _ = hidden.shape
        if num_views < 1 or n % num_views:
            raise ValueError('Token count must be divisible by camera count')
        side = math.isqrt(n // num_views)
        if side * side * num_views != n or any(side % w for w in self.window_sizes):
            raise ValueError('Expected square camera grids divisible by window sizes')
        x = self.input(hidden).reshape(b * num_views, side, side, -1)
        for i, block in enumerate(self.blocks):
            w = self.window_sizes[i % len(self.window_sizes)]
            c = x.shape[-1]
            windows = x.reshape(b * num_views, side // w, w, side // w, w, c)
            windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                windows = checkpoint(block, windows, use_reentrant=False)
            else:
                windows = block(windows)
            x = windows.reshape(b * num_views, side // w, side // w, w, w, c)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_views, side, side, c)
        pred = self.output(x.reshape(b, n, -1))
        # FP32 reductions; normalize feature channels, not space/cameras/batch.
        loss = F.smooth_l1_loss(F.layer_norm(pred.float(), (pred.shape[-1],)),
                                F.layer_norm(target.detach().float(), (target.shape[-1],)))
        return loss, pred


def validate_release_state(model, state):
    """No silent head omission, adapter migration, or partial existing-module load."""
    for key in ('llm_backbone', 'projector', 'action_head', 'visual_token_cosine_head'):
        if key not in state:
            raise ValueError(f'Released checkpoint is missing required module {key}')
        expected = getattr(model, key).state_dict()
        missing = set(expected) - set(state[key])
        extra = set(state[key]) - set(expected)
        shapes = [name for name in set(expected) & set(state[key])
                  if expected[name].shape != state[key][name].shape]
        if missing or extra or shapes:
            raise ValueError(f'Strict release restore failed for {key}: '
                             f'missing={sorted(missing)} extra={sorted(extra)} shapes={shapes}')
