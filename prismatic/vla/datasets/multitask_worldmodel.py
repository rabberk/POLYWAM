"""AgiBot and Galaxea episodes in the RoboTwin five-tubelet layout, for world-model co-training.

Both ship as *one LeRobot dataset per task* rather than a single root, so episodes
are addressed by (task, index) and each task's metadata is validated against the
embodiment spec before its episodes are used -- a few tasks in each release carry
a different action layout, and silently mixing those in would misalign the shared
action projection.

Why these two:

* **AgiBot** is dual-arm with head/left-wrist/right-wrist cameras, which is exactly
  RoboTwin's three view slots. Nothing about the slot mapping has to be learned.
* **Galaxea** is also dual-arm with those three slots (plus a second head camera we
  drop), but adds a moving torso and base. Its base commands are non-zero in ~40% of
  tasks and its torso in ~55%, so both are part of the action vector: leaving them
  out would move the cameras for reasons the dynamics head cannot see, turning its
  own ego-motion into unpredictable noise.

Horizons follow wall-clock time, as for DROID, so AgiBot at 30 fps uses a 30-step
chunk and Galaxea at 15 fps uses 15 -- both spanning the same 1.0 s as RoboTwin's 50.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, NUM_TOKENS
from prismatic.vla.datasets.droid_worldmodel import horizon_schedule
from prismatic.vla.datasets.robotwin_paper import (
    apply_weak_photometric_augmentation,
    iter_sharded_episodes,
    normalize_q01_q99,
    sample_weak_photometric_params,
    stack_three_view_tubelets,
)


@dataclass(frozen=True)
class EmbodimentSpec:
    """Everything that differs between one robot's LeRobot release and another's."""

    name: str
    # Three camera keys in RoboTwin slot order: overhead, left wrist, right wrist.
    camera_keys: tuple[str, ...]
    state_columns: tuple[str, ...]
    action_columns: tuple[str, ...]
    proprio_dim: int
    action_dim: int
    fps: float
    # Galaxea's task strings are "<Chinese>@<English>"; keep the half we condition on.
    instruction_separator: str | None = None

    def __post_init__(self) -> None:
        if len(self.camera_keys) != 3:
            raise ValueError(f"{self.name}: expected exactly three camera slots")


AGIBOT_SPEC = EmbodimentSpec(
    name="agibot_qpos16",
    camera_keys=(
        "observation.images.head_color",
        "observation.images.hand_left_color",
        "observation.images.hand_right_color",
    ),
    state_columns=("observation.state",),
    action_columns=("action",),
    proprio_dim=16,
    action_dim=16,
    fps=30.0,
)

GALAXEA_SPEC = EmbodimentSpec(
    name="galaxea_qpos26",
    camera_keys=(
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ),
    state_columns=(
        "observation.state.left_arm",
        "observation.state.right_arm",
        "observation.state.left_gripper",
        "observation.state.right_gripper",
        "observation.state.torso",
        "observation.state.chassis",
    ),
    action_columns=(
        "action.left_arm",
        "action.right_arm",
        "action.left_gripper",
        "action.right_gripper",
        "action.torso.velocities",
        "action.chassis.velocities",
    ),
    proprio_dim=21,
    action_dim=26,
    fps=15.0,
    instruction_separator="@",
)

SPECS = {spec.name: spec for spec in (AGIBOT_SPEC, GALAXEA_SPEC)}


@dataclass
class MultiTaskWorldModelDatasetConfig:
    root: Path
    stats_path: Path
    spec_name: str
    seed: int = 7
    num_workers: int = 4
    prefetch_factor: int = 2
    max_decode_retries: int = 8
    enable_photometric_augmentation: bool = False
    photometric_augmentation_probability: float = 0.4
    photometric_augmentation_strength: float = 0.1
    max_tasks: int | None = None
    max_episodes_per_task: int | None = None
    # (recorded root, local root) pairs used to repair symlinks written on another host.
    path_remaps: tuple[tuple[str, str], ...] = (("/path/to/local_resource", "/path/to/local_resource"),)
    # Stat every clip during discovery. Off by default: it costs one stat per episode
    # per rank and, on these releases, rejects nothing.
    verify_videos: bool = False


@dataclass
class _Task:
    root: Path
    chunk_size: int
    num_episodes: int
    data_template: str
    video_template: str
    instructions: dict[int, str] = field(default_factory=dict)


def embodiment_spec_for(spec_name: str, root: str | Path) -> tuple[str, tuple[int, int, int]]:
    """(name, (d_action, d_proprio, action_horizon)) for the cross-embodiment registry."""
    spec = SPECS[spec_name]
    _, chunk = horizon_schedule(spec.fps)
    return spec.name, (spec.action_dim, spec.proprio_dim, chunk)


class MultiTaskWorldModelDataset(IterableDataset):
    """Stream per-task LeRobot releases shaped like RoboTwin's world-model batches."""

    def __init__(
        self,
        config: MultiTaskWorldModelDatasetConfig,
        image_transform,
        tokenizer,
        prompt_builder_fn,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.spec = SPECS[config.spec_name]
        self.root = Path(config.root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.future_offsets, self.action_chunk = horizon_schedule(self.spec.fps)
        self.stats = load_statistics(config.stats_path, self.spec)

        self.tasks, self.index = self._discover()
        if not self.index:
            raise ValueError(f"no usable {self.spec.name} episodes under {self.root}")

        self.dataset_statistics = {self.spec.name: self.stats}
        self.dataloader_num_workers = config.num_workers
        self.dataloader_prefetch_factor = config.prefetch_factor
        self.dataloader_pin_memory = True

    # ------------------------------------------------------------------ discovery

    def _discover(self) -> tuple[list[_Task], list[tuple[int, int]]]:
        """Index every episode of every task whose layout matches the spec."""
        task_dirs = sorted(p for p in self.root.iterdir() if (p / "meta" / "info.json").exists())
        if self.config.max_tasks is not None:
            task_dirs = task_dirs[: self.config.max_tasks]

        tasks: list[_Task] = []
        index: list[tuple[int, int]] = []
        for task_dir in task_dirs:
            task = self._load_task(task_dir)
            if task is None:
                continue
            count = task.num_episodes
            if self.config.max_episodes_per_task is not None:
                count = min(count, self.config.max_episodes_per_task)
            if count <= 0:
                continue
            task_id = len(tasks)
            # No per-episode stat here. Every clip in these releases resolves either
            # directly or through the symlink remap below, so a pre-scan rejects
            # nothing while costing one stat per episode *per rank* -- 160k on AgiBot,
            # against a shared mount, which stalls startup long before the first step.
            # A genuinely missing clip is handled by the decode retry in __iter__.
            if self.config.verify_videos:
                episodes = [
                    episode
                    for episode in range(count)
                    if self._video_path(task, episode, self.spec.camera_keys[0]) is not None
                ]
            else:
                episodes = range(count)
            tasks.append(task)
            index.extend((task_id, episode) for episode in episodes)
        return tasks, index

    def _video_relpath(self, task: _Task, episode_index: int, camera_key: str) -> str:
        return task.video_template.format(
            episode_chunk=episode_index // task.chunk_size,
            episode_index=episode_index,
            video_key=camera_key,
        )

    def _video_path(self, task: _Task, episode_index: int, camera_key: str) -> Path | None:
        """Resolve one clip, repairing links recorded against another host's root.

        The AgiBot mirror re-indexes episodes through symlinks whose targets are
        absolute paths on the machine that built it. The files are all present here
        under the same relative structure, so swapping the leading mount back makes
        them readable; without this the whole release looks empty.
        """
        path = task.root / self._video_relpath(task, episode_index, camera_key)
        if path.exists():
            return path
        if not path.is_symlink():
            return None
        target = os.readlink(path)
        for source_root, local_root in self.config.path_remaps:
            if target.startswith(source_root):
                candidate = Path(local_root + target[len(source_root) :])
                if candidate.exists():
                    return candidate
        return None

    def _load_task(self, task_dir: Path) -> _Task | None:
        """Reject a task whose frame rate or column widths differ from the spec.

        Both releases contain a handful of tasks recorded with a different action
        layout. Concatenating those into the same per-robot projection would shift
        every dimension, so they are dropped rather than reshaped.
        """
        try:
            info = json.loads((task_dir / "meta" / "info.json").read_text())
        except Exception:
            return None
        if abs(float(info.get("fps", 0)) - self.spec.fps) > 1e-6:
            return None

        features = info.get("features", {})
        for key in self.spec.camera_keys:
            if key not in features:
                return None
        if self._column_width(features, self.spec.state_columns) != self.spec.proprio_dim:
            return None
        if self._column_width(features, self.spec.action_columns) != self.spec.action_dim:
            return None

        instructions = {}
        tasks_file = task_dir / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            for line in tasks_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                instructions[int(record["task_index"])] = str(record.get("task", ""))

        return _Task(
            root=task_dir,
            chunk_size=int(info.get("chunks_size", 1000)),
            num_episodes=int(info.get("total_episodes", 0)),
            data_template=info.get(
                "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            video_template=info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            ),
            instructions=instructions,
        )

    @staticmethod
    def _column_width(features: dict, columns: Sequence[str]) -> int | None:
        total = 0
        for column in columns:
            if column not in features:
                return None
            shape = features[column].get("shape") or [1]
            width = 1
            for dim in shape:
                width *= int(dim)
            total += width
        return total

    # ------------------------------------------------------------------- loading

    @staticmethod
    def _decode_frames(path: Path, wanted: Sequence[int]) -> dict[int, torch.Tensor]:
        """Decode only the frames a sample needs, plus the seek-to-keyframe run-up.

        A sample uses seven frames out of a clip that can run well past a thousand,
        so decoding the whole clip does two orders of magnitude of useless work --
        enough to make the co-training streams, not the GPU, set the step time.
        """
        import av

        first, last = min(wanted), max(wanted)
        needed = set(wanted)
        frames: dict[int, torch.Tensor] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            rate = float(stream.average_rate or 0) or None
            time_base = float(stream.time_base) if stream.time_base else None
            if rate and time_base:
                # Land on the keyframe at or before the first frame we want; decoding
                # has to run from a keyframe, so seeking further forward would leave
                # the decoder unable to reconstruct the frame.
                start = int(stream.start_time or 0)
                try:
                    container.seek(start + int(first / rate / time_base), stream=stream, backward=True)
                except Exception:
                    container.seek(0)
            index = 0
            for frame in container.decode(stream):
                if rate and time_base and frame.pts is not None:
                    index = int(round((float(frame.pts) * time_base - (stream.start_time or 0) * time_base) * rate))
                if index in needed:
                    frames[index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                    if len(frames) == len(needed):
                        break
                if index > last:
                    break
                if not (rate and time_base):
                    index += 1
        if not frames:
            raise RuntimeError(f"decoded no frames from {path}")
        # A clip a little shorter than its table, or an index the seek overshot,
        # falls back to the nearest frame that did decode rather than failing.
        available = sorted(frames)
        for target in wanted:
            if target not in frames:
                frames[target] = frames[min(available, key=lambda i: abs(i - target))]
        return frames

    @staticmethod
    def _column_matrix(table, columns: Sequence[str]) -> np.ndarray:
        parts = []
        for column in columns:
            values = np.asarray(table[column].to_pylist(), dtype=np.float32)
            parts.append(values.reshape(len(values), -1))
        return np.concatenate(parts, axis=1)

    def _instruction(self, task: _Task, task_index: int) -> str:
        text = task.instructions.get(int(task_index), "")
        if self.spec.instruction_separator and self.spec.instruction_separator in text:
            text = text.split(self.spec.instruction_separator)[-1]
        return text.strip() or "complete the task"

    def _require_video(self, task: _Task, episode_index: int, camera_key: str) -> Path:
        path = self._video_path(task, episode_index, camera_key)
        if path is None:
            raise FileNotFoundError(f"{camera_key} clip missing for episode {episode_index}")
        return path

    def _load_episode(self, task_id: int, episode_index: int, anchor_rng: random.Random):
        import pyarrow.parquet as pq

        task = self.tasks[task_id]
        chunk = episode_index // task.chunk_size
        columns = list(self.spec.state_columns + self.spec.action_columns + ("task_index",))
        table = pq.read_table(
            task.root / task.data_template.format(episode_chunk=chunk, episode_index=episode_index),
            columns=columns,
        )
        states = self._column_matrix(table, self.spec.state_columns)
        actions = self._column_matrix(table, self.spec.action_columns)
        instruction = self._instruction(task, table["task_index"].to_pylist()[0])
        length = min(len(states), len(actions))
        if length <= 0 or states.shape[1] != self.spec.proprio_dim or actions.shape[1] != self.spec.action_dim:
            raise ValueError(f"invalid {self.spec.name} episode {task_id}/{episode_index}")

        # The anchor is drawn before decoding so only its own frames are read.
        anchor = anchor_rng.randrange(length)
        current_pair, future, _ = self._indices(anchor, length)
        wanted = sorted(set(current_pair) | set(future))
        videos = [
            self._decode_frames(self._require_video(task, episode_index, key), wanted)
            for key in self.spec.camera_keys
        ]
        return states, actions, instruction, videos, anchor

    @lru_cache(maxsize=512)
    def _tokenize_instruction(self, instruction: str) -> torch.Tensor:
        prompt = self.prompt_builder_fn("openvla")
        prompt.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        prompt.add_turn("gpt", "")
        input_ids = self.tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids
        input_ids.extend([ACTION_TOKEN_BEGIN_IDX] * NUM_TOKENS)
        return torch.tensor(input_ids, dtype=torch.long)

    def _indices(self, anchor: int, episode_length: int):
        clamp = lambda index: min(max(index, 0), episode_length - 1)
        return (
            (clamp(anchor - 1), anchor),
            tuple(clamp(anchor + offset) for offset in self.future_offsets),
            tuple(clamp(anchor + step) for step in range(self.action_chunk)),
        )

    def _make_sample(self, anchor, states, actions, instruction, videos, rng: random.Random) -> dict:
        augmentation = None
        if self.config.enable_photometric_augmentation:
            augmentation = sample_weak_photometric_params(
                rng,
                probability=self.config.photometric_augmentation_probability,
                strength=self.config.photometric_augmentation_strength,
            )

        def transform_frame(frame: torch.Tensor) -> torch.Tensor:
            image = Image.fromarray(frame.numpy()).convert("RGB")
            image = apply_weak_photometric_augmentation(image, augmentation)
            return self.image_transform(image)

        current_pair, future, action_indices = self._indices(anchor, len(states))
        selected = list(current_pair) + list(future)
        transformed_views = [
            torch.stack([transform_frame(frames[index]) for index in selected]) for frames in videos
        ]
        output = stack_three_view_tubelets(
            transformed_views,
            current_pair=(0, 1),
            future_indices=tuple(range(2, 2 + len(future))),
        )
        output.update(
            input_ids=self._tokenize_instruction(instruction),
            actions=normalize_q01_q99(
                actions[list(action_indices)], self.stats["action"]["q01"], self.stats["action"]["q99"]
            ),
            proprio=normalize_q01_q99(
                states[anchor], self.stats["proprio"]["q01"], self.stats["proprio"]["q99"]
            ),
            dataset_name=self.spec.name,
        )
        return output

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        epoch = 0
        while True:
            positions = list(
                iter_sharded_episodes(
                    len(self.index),
                    self.rank,
                    self.world_size,
                    worker_id,
                    num_workers,
                    seed=self.config.seed,
                    epoch=epoch,
                )
            )
            rng = random.Random(self.config.seed + epoch * 1_000_003 + self.rank * 10_007 + worker_id)
            for position_index, position in enumerate(positions):
                candidates = positions[position_index : position_index + self.config.max_decode_retries]
                episode = None
                for candidate in candidates:
                    try:
                        episode = self._load_episode(*self.index[candidate], rng)
                        break
                    except Exception:
                        continue
                if episode is None:
                    continue
                states, actions, instruction, videos, anchor = episode
                yield self._make_sample(anchor, states, actions, instruction, videos, rng)
            epoch += 1


def load_statistics(path: str | Path, spec: EmbodimentSpec) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(Path(path).read_text())
    stats = payload.get(spec.name, payload)
    for name, width in (("action", spec.action_dim), ("proprio", spec.proprio_dim)):
        if name not in stats:
            raise ValueError(f"{spec.name} statistics are missing `{name}`")
        for bound in ("q01", "q99"):
            values = np.asarray(stats[name][bound], dtype=np.float32)
            if values.shape != (width,):
                raise ValueError(f"{spec.name} {name}.{bound} must be {width}-D, got {values.shape}")
            stats[name][bound] = values
    return stats
