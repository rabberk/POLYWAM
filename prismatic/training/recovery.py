"""Pure helpers for safe RoboTwin checkpoint recovery."""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping

import torch
from torch.optim.lr_scheduler import LambdaLR


RECOVERY_SCHEDULE_METADATA_KEYS = frozenset(
    {"max_steps", "learning_rate", "min_learning_rate", "warmup_ratio"}
)


def resume_micro_batches_within_checkpoint_epoch(
    *,
    global_step: int,
    checkpoint_epoch: int,
    optimizer_steps_per_epoch: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Return the saved epoch and only the micro-batch tail to replay within it."""
    if global_step < 0 or checkpoint_epoch < 0:
        raise ValueError("resume step and epoch must be non-negative")
    if optimizer_steps_per_epoch <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("resume epoch and accumulation sizes must be positive")

    epoch_start_step = checkpoint_epoch * optimizer_steps_per_epoch
    next_epoch_start_step = epoch_start_step + optimizer_steps_per_epoch
    if epoch_start_step > global_step:
        raise ValueError(
            f"checkpoint epoch {checkpoint_epoch} starts after global step {global_step}"
        )
    if global_step >= next_epoch_start_step:
        raise ValueError(
            f"checkpoint epoch {checkpoint_epoch} is stale for global step {global_step}"
        )
    return checkpoint_epoch, (global_step - epoch_start_step) * gradient_accumulation_steps


class SustainedLossGuard:
    """Trip on non-finite values or a sustained moving-average loss increase."""

    def __init__(self, *, threshold: float, window_size: int, patience: int) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("loss guard threshold must be finite and positive")
        if window_size <= 0 or patience <= 0:
            raise ValueError("loss guard window_size and patience must be positive")
        self.threshold = float(threshold)
        self.window = deque(maxlen=int(window_size))
        self.patience = int(patience)
        self.consecutive_bad_windows = 0

    @property
    def moving_average(self) -> float | None:
        if len(self.window) < self.window.maxlen:
            return None
        return sum(self.window) / len(self.window)

    def observe(self, value: float) -> bool:
        value = float(value)
        if not math.isfinite(value):
            return True
        self.window.append(value)
        average = self.moving_average
        if average is None or average <= self.threshold:
            self.consecutive_bad_windows = 0
            return False
        self.consecutive_bad_windows += 1
        return self.consecutive_bad_windows >= self.patience


def validate_resume_metadata(
    expected: Mapping[str, object],
    actual: Mapping[str, object] | None,
    *,
    allow_schedule_overrides: bool,
    allow_episode_selection_override: bool = False,
    allow_global_batch_size_override: bool = False,
) -> None:
    """Validate checkpoint provenance, optionally ignoring schedule-only fields."""
    if actual is None:
        raise ValueError("checkpoint provenance mismatch: checkpoint metadata is missing")

    keys = set(expected) | set(actual)
    if allow_schedule_overrides:
        keys.difference_update(RECOVERY_SCHEDULE_METADATA_KEYS)
    if allow_episode_selection_override:
        keys.discard("episode_selection")
    if allow_global_batch_size_override:
        keys.discard("global_batch_size")
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in sorted(keys)
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise ValueError(f"checkpoint provenance mismatch: {mismatches}")


def reset_optimizer_schedule_for_recovery(
    optimizer: torch.optim.Optimizer,
    *,
    start_lr: float,
    min_lr: float,
    recovery_steps: int,
) -> LambdaLR:
    """Reset optimizer group LRs and return a relative cosine recovery schedule."""
    if start_lr <= 0:
        raise ValueError("recovery start_lr must be positive")
    if min_lr < 0 or min_lr > start_lr:
        raise ValueError("recovery min_lr must lie in [0, start_lr]")
    if recovery_steps <= 0:
        raise ValueError("recovery_steps must be positive")

    floor_ratio = min_lr / start_lr
    for group in optimizer.param_groups:
        group["lr"] = start_lr
        group["initial_lr"] = start_lr
        group["base_lr"] = start_lr
        group["min_lr"] = min_lr

    def lr_lambda(step: int) -> float:
        progress = min(max(float(step) / float(recovery_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return LambdaLR(optimizer, [lr_lambda for _ in optimizer.param_groups])
