#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${ACTION_STATS:?Set ACTION_STATS}"
: "${BASE_VLM:?Set BASE_VLM}"
: "${QWEN_PATH:?Set QWEN_PATH}"
: "${VJEPA_CHECKPOINT:?Set VJEPA_CHECKPOINT}"
exec python -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --module prismatic.training.train \
  --vla.type jepawam-qwen25-vjepa-384px+0_5b+robotwin-paper20-clean-fast-lewm-transformer \
  --vla.expected_world_size "${NPROC_PER_NODE:-8}" \
  --vla.base_vlm "$BASE_VLM" \
  --llm_checkpoint_path "$QWEN_PATH" \
  --vla.vjepa_checkpoint_path "$VJEPA_CHECKPOINT" \
  --data_root_dir "$DATA_ROOT" \
  --vla.normalization_stats_path "$ACTION_STATS" \
  --vla.data_mix "${DATA_MIX:-robotwin_paper20_group1_clean}" \
  --vla.global_batch_size "${GLOBAL_BATCH_SIZE:-128}" \
  --vla.per_device_batch_size "${MICRO_BATCH_SIZE:-4}" \
  --vla.learning_rate "${LEARNING_RATE:-2e-5}" \
  --vla.min_learning_rate "${MIN_LEARNING_RATE:-1e-5}" \
  --vla.max_steps "${MAX_STEPS:-60000}" \
  --lr_decay_end_step "${LR_DECAY_END_STEP:-10000}" \
  --run_root_dir "${OUTPUT_DIR:-runs}" \
  --trackers '[jsonl]' "$@"
