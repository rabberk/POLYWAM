"""Load a single-robot checkpoint into a cross-embodiment / fine-tuned-encoder model.

Turning on co-training or unfreezing the encoder changes the module graph, so a
checkpoint written before either is not a drop-in match:

* the action-prefix encoder gains one projection pair per robot, so its former
  `state_projection` / `action_projection` become the deployment robot's entry
  under `state_projections.default` / `action_projections.default`;
* the action head gains a projection pair per extra robot;
* unfreezing adds an EMA copy of the visual encoder.

Renaming the first is exact -- same tensors, new address. The rest are genuinely
new parameters and start from their own initialisation, except the EMA copy, which
must start *equal to* the online encoder: seeding it randomly would have the world
model regress against noise for as long as the momentum takes to wash it out.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

# Modules that exist only under the new configuration. Missing entries for these
# are expected on a migrated checkpoint; anything else missing is a real mismatch.
NEW_MODULE_PREFIXES = (
    "fast_lewm_prefix_encoder.state_projections.",
    "fast_lewm_prefix_encoder.action_projections.",
    "action_head.embodiment_adapters.",
    "vision_target_encoder.",
)

# Old single-robot name -> new address under the deployment robot's key.
PREFIX_ENCODER_RENAMES = (
    ("fast_lewm_prefix_encoder.state_projection.", "fast_lewm_prefix_encoder.state_projections.default."),
    ("fast_lewm_prefix_encoder.action_projection.", "fast_lewm_prefix_encoder.action_projections.default."),
)


def migrate_state_dict(flat_model: "OrderedDict[str, object]") -> tuple["OrderedDict[str, object]", list[str]]:
    """Rename single-robot keys to their cross-embodiment addresses.

    Returns the migrated state dict and a list of the renames applied, so the
    caller can report what happened rather than silently reshaping a checkpoint.
    """
    migrated: "OrderedDict[str, object]" = OrderedDict()
    applied: list[str] = []
    for key, value in flat_model.items():
        for old, new in PREFIX_ENCODER_RENAMES:
            if key.startswith(old):
                key = new + key[len(old) :]
                applied.append(key)
                break
        migrated[key] = value
    return migrated, applied


def seed_target_encoder(flat_model: "OrderedDict[str, object]", missing_keys: Iterable[str]) -> int:
    """Copy online-encoder weights into the EMA target's slots.

    The EMA target must start equal to the encoder it tracks. Left to its own
    initialisation it would supply random regression targets to the world model
    for as long as the momentum takes to converge on the online weights.
    """
    wanted = [key for key in missing_keys if key.startswith("vision_target_encoder.")]
    seeded = 0
    for key in wanted:
        source = "vision_backbone." + key[len("vision_target_encoder.") :]
        if source in flat_model:
            flat_model[key] = flat_model[source].clone()
            seeded += 1
    return seeded


# Decorations PEFT adds when it wraps a module in LoRA adapters.
_PEFT_PREFIX = "base_model.model."
_PEFT_BASE_LAYER = ".base_layer"


def adapt_full_parameter_llm_to_lora(
    checkpoint_llm: "OrderedDict[str, object]", model_keys: Iterable[str]
) -> tuple["OrderedDict[str, object]", int, int]:
    """Re-address a full-parameter language checkpoint for a LoRA-wrapped model.

    Training Qwen outright and then continuing with adapters is a legitimate move --
    the base weights are exactly what a LoRA run should start from -- but PEFT
    renames every module it wraps, so the two state dicts share no keys at all.

    The mapping is derived from the model's own key list rather than from PEFT's
    naming rules: for each parameter the model expects, its decorations are stripped
    to recover the name the checkpoint would have used. Adapter tensors have no
    counterpart and keep their initialisation, which is what a fresh LoRA run wants.

    Returns the re-addressed dict, the number of base tensors mapped, and the number
    of adapter tensors left untouched.
    """
    adapted: "OrderedDict[str, object]" = OrderedDict()
    mapped = adapters = 0
    for key in model_keys:
        if "lora_" in key:
            adapters += 1
            continue
        source = key.replace(_PEFT_PREFIX, "", 1).replace(_PEFT_BASE_LAYER, "", 1)
        if source in checkpoint_llm:
            adapted[key] = checkpoint_llm[source]
            mapped += 1
    return adapted, mapped, adapters


def drop_extra_embodiments(
    state: "OrderedDict[str, object]", model_keys: Iterable[str]
) -> tuple["OrderedDict[str, object]", list[str]]:
    """Keep only what a single-robot model can hold, renaming the deployment robot's
    entry back to its unsuffixed address.

    A co-trained checkpoint carries one projection pair per robot. Narrowing back to
    the deployment robot alone is a legitimate continuation -- the other robots'
    projections simply have nowhere to go -- but their keys would otherwise be
    rejected as unexpected, and the deployment robot's own weights sit under a name
    the single-robot module does not use.
    """
    expected = set(model_keys)
    kept: "OrderedDict[str, object]" = OrderedDict()
    dropped: list[str] = []
    for key, value in state.items():
        if key in expected:
            kept[key] = value
            continue
        # `state_projections.default.X` -> `state_projection.X`, likewise for actions.
        renamed = key.replace("state_projections.default.", "state_projection.").replace(
            "action_projections.default.", "action_projection."
        )
        renamed = renamed.replace("embodiment_adapters.default.", "")
        if renamed in expected:
            kept[renamed] = value
        else:
            dropped.append(key)
    return kept, dropped


def expand_pooler_queries(
    checkpoint_state: dict,
    expected_state: dict,
    *,
    noise_std: float = 0.02,
) -> tuple[dict, list[str]]:
    """Carry a trained pooler onto one with more queries.

    Only ``query`` changes shape when the pooled latent grows from one token per
    timestep to several; the key, value and output projections -- which is where
    the pooler's learning actually sits -- are identical and load unchanged. The
    trained queries are tiled across the new slots and jittered, so every new
    query starts from what the old one had learned to look at and can diverge from
    there, rather than starting from noise.
    """
    import torch

    migrated, rebuilt = dict(checkpoint_state), []
    for key, expected in expected_state.items():
        saved = checkpoint_state.get(key)
        if saved is None or saved.shape == expected.shape:
            continue
        if key.endswith("query") and saved.ndim == expected.ndim == 3 and saved.shape[0] == expected.shape[0]:
            repeats = -(-expected.shape[1] // saved.shape[1])
            tiled = saved.repeat(1, repeats, 1)[:, : expected.shape[1]].clone()
            migrated[key] = tiled + noise_std * torch.randn_like(tiled)
        else:
            migrated[key] = expected.clone()
        rebuilt.append(key)
    return migrated, rebuilt
