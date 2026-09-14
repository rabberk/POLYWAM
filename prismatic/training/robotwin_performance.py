"""Analytic FLOPs and sampled CUDA timing for RoboTwin B300 training."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping


B300_BF16_PEAK_FLOPS = 2.25e15

# Fixed public geometry. ``qwen_sequence_length`` is supplied from each padded
# batch because instruction length varies.
ROBOTWIN_FLOP_CONFIG: dict[str, int] = {
    "qwen_hidden_size": 896,
    "qwen_intermediate_size": 4_864,
    "qwen_num_layers": 24,
    "qwen_num_attention_heads": 14,
    "qwen_num_key_value_heads": 2,
    "qwen_head_dim": 64,
    "qwen_sequence_length": 0,
    "vjepa_hidden_size": 1_024,
    "vjepa_intermediate_size": 4_096,
    "vjepa_num_layers": 24,
    "vjepa_num_attention_heads": 16,
    "vjepa_head_dim": 64,
    # Action observation plus current pair: three views each. Future input is
    # three ten-frame views, or five 576-token tubelets per view.
    "vjepa_short_sequence_length": 576,
    "vjepa_short_sequences_per_sample": 6,
    "vjepa_long_sequence_length": 2_880,
    "vjepa_long_sequences_per_sample": 3,
    "action_hidden_size": 1_536,
    "action_intermediate_size": 6_144,
    "action_num_layers": 16,
    "action_num_attention_heads": 32,
    "action_num_key_value_heads": 32,
    "action_head_dim": 48,
    # One proprio token, 32 learned future tokens, and 50 action tokens.
    "action_sequence_length": 83,
}


class OptimizerStepTimingWindow:
    """Keep one phase timer and wall-clock origin for an accumulation window."""

    def __init__(self, timer_factory: Callable[[], object], *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._timer_factory = timer_factory
        self._clock = clock
        self._timer = None
        self._started_at = None

    def acquire(self) -> tuple[object, float]:
        if self._timer is None:
            self._timer = self._timer_factory()
            self._started_at = self._clock()
        return self._timer, self._started_at

    def reset(self) -> None:
        self._timer = None
        self._started_at = None


def _required_int(config: Mapping[str, int], key: str, *, allow_zero: bool = False) -> int:
    try:
        value = int(config[key])
    except KeyError as error:
        raise KeyError(f"missing RoboTwin FLOP field {key!r}") from error
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{key} must be {qualifier}, got {value}")
    return value


def _projection_parameters(hidden: int, heads: int, kv_heads: int, head_dim: int) -> int:
    query = heads * head_dim
    key_value = kv_heads * head_dim
    return hidden * (query + key_value + key_value + query)


def _training_transformer_flops(config: Mapping[str, int], prefix: str) -> int:
    hidden = _required_int(config, f"{prefix}_hidden_size")
    intermediate = _required_int(config, f"{prefix}_intermediate_size")
    layers = _required_int(config, f"{prefix}_num_layers")
    heads = _required_int(config, f"{prefix}_num_attention_heads")
    kv_heads = _required_int(config, f"{prefix}_num_key_value_heads")
    head_dim = _required_int(config, f"{prefix}_head_dim")
    sequence = _required_int(config, f"{prefix}_sequence_length")
    linear_parameters = _projection_parameters(hidden, heads, kv_heads, head_dim)
    linear_parameters += 3 * hidden * intermediate
    linear = 6 * layers * linear_parameters * sequence
    attention = 12 * layers * heads * head_dim * sequence * sequence
    return linear + attention


def _frozen_vjepa_flops(config: Mapping[str, int]) -> int:
    hidden = _required_int(config, "vjepa_hidden_size")
    intermediate = _required_int(config, "vjepa_intermediate_size")
    layers = _required_int(config, "vjepa_num_layers")
    heads = _required_int(config, "vjepa_num_attention_heads")
    head_dim = _required_int(config, "vjepa_head_dim")
    short_length = _required_int(config, "vjepa_short_sequence_length", allow_zero=True)
    short_count = _required_int(config, "vjepa_short_sequences_per_sample", allow_zero=True)
    long_length = _required_int(config, "vjepa_long_sequence_length", allow_zero=True)
    long_count = _required_int(config, "vjepa_long_sequences_per_sample", allow_zero=True)
    tokens = short_count * short_length + long_count * long_length
    squared_tokens = short_count * short_length**2 + long_count * long_length**2
    linear_parameters = 4 * hidden * hidden + 3 * hidden * intermediate
    # V-JEPA and its projector are frozen, so only forward FLOPs are counted.
    linear = 2 * layers * linear_parameters * tokens
    attention = 4 * layers * heads * head_dim * squared_tokens
    return linear + attention


def estimate_robotwin_training_flops(config: Mapping[str, int], batch_size: int) -> int:
    """Estimate algorithmic forward/backward FLOPs for one optimizer batch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    per_sample = _training_transformer_flops(config, "qwen")
    per_sample += _frozen_vjepa_flops(config)
    per_sample += _training_transformer_flops(config, "action")
    return int(batch_size) * per_sample


def estimated_mfu(
    flops: int | float,
    seconds: float,
    world_size: int,
    peak_flops: float = B300_BF16_PEAK_FLOPS,
) -> float:
    """Return estimated utilization against the aggregate device peak."""
    if flops < 0:
        raise ValueError("flops must be non-negative")
    if seconds <= 0 or world_size <= 0 or peak_flops <= 0:
        raise ValueError("seconds, world_size, and peak_flops must be positive")
    return float(flops) / (float(seconds) * int(world_size) * float(peak_flops))


def should_sample_performance(global_step: int, interval: int) -> bool:
    return interval > 0 and global_step > 0 and global_step % interval == 0


class CudaPhaseTimer:
    """Record CUDA phase events and synchronize once, only on sampled steps."""

    def __init__(
        self,
        enabled: bool,
        phases: tuple[str, ...],
        *,
        event_factory=None,
        synchronize=None,
        reset_peak_memory: Callable[[], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.phases = phases
        self._events = {phase: [] for phase in phases}
        self._cpu_elapsed = {phase: 0.0 for phase in phases}
        self._active = {}
        if self.enabled and (event_factory is None or synchronize is None):
            import torch

            event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
            synchronize = synchronize or torch.cuda.synchronize
        self._event_factory = event_factory
        self._synchronize = synchronize
        if self.enabled and reset_peak_memory is not None:
            reset_peak_memory()

    def _check_phase(self, phase: str) -> None:
        if phase not in self._events:
            raise KeyError(f"unknown CUDA timing phase: {phase}")

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        self._check_phase(phase)
        if phase in self._active:
            raise RuntimeError(f"CUDA phase {phase!r} was started twice")
        event = self._event_factory()
        event.record()
        self._active[phase] = event

    def stop(self, phase: str) -> None:
        if not self.enabled:
            return
        self._check_phase(phase)
        if phase not in self._active:
            raise RuntimeError(f"CUDA phase {phase!r} was stopped before it started")
        event = self._event_factory()
        event.record()
        self._events[phase].append((self._active.pop(phase), event))

    def add_cpu_time(self, phase: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._check_phase(phase)
        self._cpu_elapsed[phase] += float(seconds)

    def finish(self) -> dict[str, float]:
        elapsed = dict(self._cpu_elapsed)
        if not self.enabled:
            return elapsed
        if self._active:
            raise RuntimeError(f"unfinished CUDA timing phases: {sorted(self._active)}")
        self._synchronize()
        for phase, intervals in self._events.items():
            elapsed[phase] += sum(start.elapsed_time(end) for start, end in intervals) / 1_000.0
        return elapsed
