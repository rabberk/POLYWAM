"""B300-only FlashAttention-4 adapter for Qwen's attention interface."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from torch.nn.attention.flex_attention import FlexKernelOptions, create_block_mask, flex_attention


_BLOCK_SIZE = 256
_fa4_registration: Any | None = None


def _fa4_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs: Any,
) -> Any:
    """Run FlexAttention only while Dynamo is lowering it to a fused kernel."""
    if not torch.compiler.is_dynamo_compiling():
        raise RuntimeError(
            "B300 FA4 must execute as a compiled fused kernel; eager dense fallback is disabled. "
            "Ensure TORCHDYNAMO_DISABLE is unset."
        )
    return flex_attention(query, key, value, **kwargs)


_compiled_flex_attention = torch.compile(_fa4_flex_attention, dynamic=False, fullgraph=True)


def _flex_kernel_options_metadata() -> set[str]:
    return set(getattr(FlexKernelOptions, "__annotations__", {}))


def validate_fa4_runtime() -> None:
    """Fail before FSDP setup unless this PyTorch exposes the required FA4 stack."""
    if "BACKEND" not in _flex_kernel_options_metadata():
        raise RuntimeError("installed PyTorch FlexAttention lacks BACKEND kernel metadata")
    if importlib.util.find_spec("flash_attn.cute") is None:
        raise RuntimeError("FA4 requires the flash_attn.cute package")

    global _fa4_registration
    if _fa4_registration is None:
        from torch.nn.attention._fa4 import register_flash_attention_fa4

        _fa4_registration = register_flash_attention_fa4()


def build_fa4_block_mask(
    prefix_valid: torch.Tensor,
    n_tubelets: int,
    tokens_per_tubelet: int,
    block_size: int = _BLOCK_SIZE,
):
    """Create a padded BlockMask with the joint prefix and tubelet-causal semantics."""
    if prefix_valid.ndim != 2:
        raise ValueError("prefix_valid must have shape [B, P]")
    if n_tubelets <= 0 or tokens_per_tubelet <= 0:
        raise ValueError("n_tubelets and tokens_per_tubelet must be positive")
    if block_size != _BLOCK_SIZE:
        raise ValueError("B300 FA4 requires block_size=256")

    prefix_valid = prefix_valid.bool()
    batch_size, prefix_length = prefix_valid.shape
    if prefix_length == 0:
        raise ValueError("prefix_valid must contain at least one prefix token")
    logical_length = prefix_length + n_tubelets * tokens_per_tubelet
    padded_length = ((logical_length + block_size - 1) // block_size) * block_size

    def mask_mod(batch: torch.Tensor, head: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        del head
        in_bounds = (query < logical_length) & (key < logical_length)
        prefix_query = query < prefix_length
        prefix_key = key < prefix_length
        prefix_key_valid = prefix_valid[batch, key.clamp(max=prefix_length - 1)]
        prefix_allowed = prefix_query & prefix_key & (query >= key) & prefix_key_valid
        future_query = ~prefix_query
        future_key = ~prefix_key
        future_group_query = (query - prefix_length) // tokens_per_tubelet
        future_group_key = (key - prefix_length) // tokens_per_tubelet
        future_allowed = future_query & (
            prefix_key & prefix_key_valid
            | future_key & (future_group_query >= future_group_key)
        )
        padding_sentinel = (query >= logical_length) & (key == 0)
        return padding_sentinel | (in_bounds & (prefix_allowed | future_allowed))

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=prefix_valid.device,
        BLOCK_SIZE=block_size,
    )


def b300_fa4_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hugging Face attention adapter backed exclusively by FlexAttention FA4."""
    del module
    if dropout:
        raise RuntimeError("B300 FA4 attention requires Qwen attention_dropout=0")
    if attention_mask is None:
        raise ValueError("B300 FA4 attention requires a BlockMask")

    logical_length = query.shape[-2]
    padded_length = ((logical_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    padding = padded_length - logical_length
    if padding:
        query = torch.nn.functional.pad(query, (0, 0, 0, padding))
        key = torch.nn.functional.pad(key, (0, 0, 0, padding))
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))

    output, lse = _compiled_flex_attention(
        query,
        key,
        value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
        return_lse=True,
        kernel_options={"BACKEND": "FLASH"},
    )
    return output[:, :, :logical_length].transpose(1, 2).contiguous(), lse[:, :, :logical_length]


def register_b300_fa4_attention() -> str:
    """Register the B300 adapter under the Qwen attention implementation name."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("b300_fa4", b300_fa4_attention_forward)
    return "b300_fa4"
