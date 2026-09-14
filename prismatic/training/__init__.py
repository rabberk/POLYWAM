"""Training API with optional stacks loaded only when requested."""


def __getattr__(name):
    if name == "get_fsdp_strategy":
        from .materialize import get_fsdp_strategy

        return get_fsdp_strategy
    if name == "VLAMetrics":
        from .metrics import VLAMetrics

        return VLAMetrics
    raise AttributeError(name)


__all__ = ["get_fsdp_strategy", "VLAMetrics"]
