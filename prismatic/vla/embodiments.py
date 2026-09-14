"""Action-space dimensions per co-training robot, with no heavy imports.

Both the trainer and the inference loader need to know a run's robot set: the
action-prefix projections and action-head adapters are keyed per robot, so a model
rebuilt without them cannot load a co-trained checkpoint. The trainer can afford to
reach into the dataset modules for this; the inference server runs in a leaner
environment, so the table lives here instead, free of torch, datasets or configs.

Values mirror the dataset definitions -- change them together.
"""

from __future__ import annotations

# name -> (action dim, proprio dim, action chunk in steps)
#
# The chunk is the robot's frame rate times the shared 1.0 s horizon, so each robot
# predicts the same span of wall-clock time: RoboTwin 50 fps, AgiBot 30, the others 15.
EMBODIMENT_DIMENSIONS: dict[str, tuple[int, int, int]] = {
    "agibot_qpos16": (16, 16, 30),
    "galaxea_qpos26": (26, 21, 15),
    "droid_qpos8": (8, 8, 15),
}


def parse_co_train_spec_names(spec: str | None) -> list[str]:
    """Robot names from a `<name>:<root>:<stats>[;...]` spec string."""
    if not spec or not spec.strip():
        return []
    names = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if entry:
            names.append(entry.split(":")[0].strip())
    return names


def cross_embodiments_from_spec(spec: str | None) -> dict[str, tuple[int, int, int]]:
    """Robot name -> dimensions, for every name in the spec that is known here."""
    return {
        name: EMBODIMENT_DIMENSIONS[name]
        for name in parse_co_train_spec_names(spec)
        if name in EMBODIMENT_DIMENSIONS
    }
