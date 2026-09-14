"""VLA data API, lazily loaded to keep Ego training independent of TensorFlow."""


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(name)


__all__ = ["get_vla_dataset_and_collator"]
