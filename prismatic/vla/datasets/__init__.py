"""Dataset API with lazy RLDS imports."""


def __getattr__(name):
    if name in {"RLDSDataset", "VLABatchTransform"}:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(name)


__all__ = ["RLDSDataset", "VLABatchTransform"]
