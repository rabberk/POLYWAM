"""VLM implementations exposed lazily to avoid loading optional data stacks."""


def __getattr__(name):
    if name == "PrismaticVLM":
        from .prismatic import PrismaticVLM

        return PrismaticVLM
    if name == "Ego5TubeletWorldModel":
        from .ego_world_model import Ego5TubeletWorldModel

        return Ego5TubeletWorldModel
    if name == "RoboTwinEgo5TubeletVLA":
        from .robotwin_ego_world_model import RoboTwinEgo5TubeletVLA

        return RoboTwinEgo5TubeletVLA
    raise AttributeError(name)


__all__ = ["PrismaticVLM", "Ego5TubeletWorldModel", "RoboTwinEgo5TubeletVLA"]
