"""Public package API with lazy heavyweight model imports."""


def __getattr__(name):
    if name == "load_vla":
        from .models import load_vla

        return load_vla
    raise AttributeError(name)


__all__ = ["load_vla"]
