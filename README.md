# PolyWAM — Training Source

Anonymous review snapshot of the training implementation. This repository
contains no evaluation launchers, rollout videos, datasets, checkpoints,
experiment logs, credentials, or original Git history.

## Implementation map

| Component | Source |
|---|---|
| Causal action-prefix encoder and window/horizon dynamics Transformer | `prismatic/models/fast_lewm.py` |
| Motion-residual and temporal objectives | `prismatic/models/robotwin_latent_ar.py` (`motion_residual_losses`) |
| Policy/world-head training integration | `prismatic/models/vlms/prismatic.py` (`_forward_fast_lewm`) |
| Flow-matching action head | `prismatic/models/flow_gr00t_action_head.py` |
| Distributed training entry point | `prismatic/training/train.py` |
| FSDP and optimization loop | `prismatic/training/strategies/` |
| RoboTwin sampling, paired targets and appearance augmentation | `prismatic/vla/datasets/robotwin_paper.py` |
| LIBERO RLDS loading | `prismatic/vla/datasets/rlds/` |
| Model and training configuration classes | `prismatic/conf/vla.py` |

Existing internal module names are retained to avoid changing import paths and
checkpoint keys. Shared model classes include their original methods; this
release does not include an evaluation application. Compatibility modules are
retained where imported by the training package.

The dense window-Transformer branch is selected with
`fast_lewm_head_type="window_transformer"`. Other configuration classes and
experimental branches in the shared source are not automatically the settings
used for the paper. In particular, action conditioning, prediction type,
trainable parameters and attention masks must be specified for the intended
experiment; LIBERO and RoboTwin presets are not interchangeable.

## Installation and checks

Use a separate Python 3.10+ environment with a PyTorch/CUDA build suitable for
your GPUs. The dependency declarations are inherited from the training project,
not a newly validated lock file:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The smoke tests exercise the core dynamics module on CPU, including causality,
output geometry, and gradient propagation. They do not constitute an end-to-end
distributed training or benchmark reproduction test.

## Training entry

`scripts/train_robotwin.sh` is an example single-node launcher for the dense
world-head preset. Supply local data, pretrained components, and normalization
statistics yourself. The base VLM directory must have `config.json` and a
`checkpoints/` subdirectory compatible with the upstream JEPA-WAM loader.

```bash
DATA_ROOT=/path/to/robotwin \
ACTION_STATS=/path/to/action_statistics.json \
BASE_VLM=/path/to/pretrained_vlm \
QWEN_PATH=/path/to/Qwen2.5-0.5B \
VJEPA_CHECKPOINT=/path/to/vjepa.pt \
NPROC_PER_NODE=8 bash scripts/train_robotwin.sh
```

This launcher demonstrates the required arguments; it is not a claim that its
defaults reproduce every reported checkpoint. Configure the training budget,
learning-rate schedule, task selection and initialization for the desired run.
Additional trainer arguments can be appended to the command.

For LIBERO the same Python training entry uses the
`jepawam-qwen25-vjepa-384px+0_5b+libero-fast-lewm-transformer` configuration and
requires the corresponding RLDS dataset and compatible released components.

## Anonymization and provenance

This is a source snapshot, not a copy of the original repository history.
Machine-specific paths were replaced by local-resource placeholders. No model
mathematics was intentionally changed during export. `SOURCE_MANIFEST.json`
records hashes of the exported source files. Third-party copyright and license
notices are retained as required; they identify upstream authors, not the
anonymous submitters.

See `THIRD_PARTY_NOTICES.md` for attribution. External datasets and pretrained
weights remain subject to their own licenses and access conditions.
