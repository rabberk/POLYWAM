"""Model factories, imported lazily so utility modules stay lightweight."""


def __getattr__(name):
    if name == "load_vla":
        from .load import load_vla

        return load_vla
    if name in {"get_llm_backbone_and_tokenizer", "get_vision_backbone_and_transform", "get_vlm"}:
        from . import materialize

        return getattr(materialize, name)
    if name == "RoboTwinEgo5TubeletVLA":
        from .vlms.robotwin_ego_world_model import RoboTwinEgo5TubeletVLA

        return RoboTwinEgo5TubeletVLA
    raise AttributeError(name)


__all__ = [
    "load_vla",
    "get_llm_backbone_and_tokenizer",
    "get_vision_backbone_and_transform",
    "get_vlm",
    "RoboTwinEgo5TubeletVLA",
]
