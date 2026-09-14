"""DROID episodes streamed in the RoboTwin five-tubelet layout, for world-model co-training.

Both the action and world-model losses are computed on this stream. DROID is a
single-arm Franka with an 8-D joint-space action, which the deployment robot's
14-D action head cannot regress directly -- so the head carries a per-robot pair of
projections into and out of its shared trunk, and the dynamics head likewise sees
only the fixed-width prefix produced by a per-robot projection (see
`MultiEmbodimentActionPrefixEncoder`). Everything between those ends is shared.

Two alignment decisions matter here:

* **Time, not steps.** RoboTwin runs at 50 fps and uses a 50-step chunk with
  horizons at 10/20/30/40/50 steps (0.2 s ... 1.0 s). DROID runs at 15 fps, so
  matching those *durations* means a 15-step chunk with horizons at 3/6/9/12/15.
  Matching step counts instead would train one shared head on two timescales.
* **Camera slots are positional.** DROID's (exterior_1, exterior_2, wrist_left)
  fill the same three view slots as RoboTwin's (head, left_wrist, right_wrist).
  The semantics differ -- DROID is single-arm and has one wrist camera -- so the
  head must learn view-slot semantics that generalise, not memorise them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, NUM_TOKENS
from prismatic.vla.datasets.robotwin_paper import (
    apply_weak_photometric_augmentation,
    iter_sharded_episodes,
    normalize_q01_q99,
    sample_weak_photometric_params,
    stack_three_view_tubelets,
)

DATASET_NAME = "droid_qpos8"

# Positional mapping onto the three RoboTwin view slots.
CAMERA_KEYS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
)

STATE_COLUMNS = ("observation.state.joint_position", "observation.state.gripper_position")
ACTION_COLUMNS = ("action.joint_position", "action.gripper_position")
ACTION_DIM = 8

# RoboTwin's schedule expressed in seconds, so any source frame rate lands on the
# same horizons: 0.2 / 0.4 / 0.6 / 0.8 / 1.0 s, with a 1.0 s action chunk.
HORIZON_SECONDS = (0.2, 0.4, 0.6, 0.8, 1.0)
ACTION_CHUNK_SECONDS = 1.0


def horizon_schedule(fps: float) -> tuple[tuple[int, ...], int]:
    """Return (future frame offsets, action chunk length) for a given frame rate."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    offsets = tuple(max(1, round(seconds * fps)) for seconds in HORIZON_SECONDS)
    chunk = max(len(offsets), round(ACTION_CHUNK_SECONDS * fps))
    # The prefix encoder splits the chunk into one block per horizon.
    chunk -= chunk % len(offsets)
    return offsets, chunk


def droid_embodiment_spec(root: str | Path) -> tuple[str, tuple[int, int, int]]:
    """(name, (d_action, d_proprio, action_horizon)) read from the dataset itself.

    The horizon follows the source frame rate so it covers the same wall-clock
    span as RoboTwin's chunk; hard-coding it would silently mis-align the two
    streams if a mirror were re-encoded at another rate.
    """
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    _, chunk = horizon_schedule(float(info.get("fps", 15)))
    return DATASET_NAME, (ACTION_DIM, ACTION_DIM, chunk)


@dataclass
class DroidWorldModelDatasetConfig:
    root: Path
    stats_path: Path
    seed: int = 7
    num_workers: int = 4
    prefetch_factor: int = 2
    max_decode_retries: int = 8
    enable_photometric_augmentation: bool = False
    photometric_augmentation_probability: float = 0.4
    photometric_augmentation_strength: float = 0.1
    max_episodes: int | None = None


def load_droid_statistics(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(Path(path).read_text())
    stats = payload.get(DATASET_NAME, payload)
    for name in ("action", "proprio"):
        if name not in stats:
            raise ValueError(f"DROID statistics are missing `{name}`")
        for field in ("q01", "q99"):
            values = np.asarray(stats[name][field], dtype=np.float32)
            if values.shape != (ACTION_DIM,):
                raise ValueError(f"{name}.{field} must be {ACTION_DIM}-D, got {values.shape}")
            stats[name][field] = values
    return stats


class DroidWorldModelDataset(IterableDataset):
    """Stream DROID episodes shaped like RoboTwin's current-pair/future-frame batches."""

    def __init__(
        self,
        config: DroidWorldModelDatasetConfig,
        image_transform,
        tokenizer,
        prompt_builder_fn,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.root = Path(config.root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self._validate_metadata()
        self.fps = float(self.info.get("fps", 15))
        self.future_offsets, self.action_chunk = horizon_schedule(self.fps)
        self.stats = load_droid_statistics(config.stats_path)

        # info.json describes the full DROID release; a local mirror often holds a
        # prefix of it, so trust the episodes actually present on disk.
        self.episode_ids = self._discover_episodes()
        if not self.episode_ids:
            raise ValueError(f"no DROID episodes found under {self.root}")
        self.dataset_statistics = {DATASET_NAME: self.stats}
        self.dataloader_num_workers = config.num_workers
        self.dataloader_prefetch_factor = config.prefetch_factor
        self.dataloader_pin_memory = True

    def _validate_metadata(self) -> None:
        features = self.info.get("features", {})
        missing = [key for key in CAMERA_KEYS + STATE_COLUMNS + ACTION_COLUMNS if key not in features]
        if missing:
            raise ValueError(f"DROID metadata is missing required features: {missing}")

    def _discover_episodes(self) -> tuple[int, ...]:
        chunk_size = int(self.info.get("chunks_size", 1000))
        found = []
        for path in sorted((self.root / "data").glob("chunk-*/episode_*.parquet")):
            index = int(path.stem.split("_")[-1])
            if (self.root / self._video_relpath(index, CAMERA_KEYS[0], chunk_size)).exists():
                found.append(index)
        if self.config.max_episodes is not None:
            found = found[: self.config.max_episodes]
        return tuple(found)

    def _video_relpath(self, episode_index: int, camera_key: str, chunk_size: int) -> str:
        template = self.info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        return template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=camera_key,
        )

    def _episode_path(self, episode_index: int) -> Path:
        template = self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        chunk_size = int(self.info.get("chunks_size", 1000))
        return self.root / template.format(episode_chunk=episode_index // chunk_size, episode_index=episode_index)

    @staticmethod
    def _decode_video(path: Path) -> torch.Tensor:
        import av

        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
        if not frames:
            raise RuntimeError(f"decoded no frames from {path}")
        return torch.stack(frames)

    @staticmethod
    def _column_matrix(table, columns: Sequence[str]) -> np.ndarray:
        """Stack scalar and vector columns into one [T, D] float array."""
        parts = []
        for column in columns:
            values = np.asarray(table[column].to_pylist(), dtype=np.float32)
            parts.append(values.reshape(len(values), -1))
        return np.concatenate(parts, axis=1)

    def _load_episode(self, episode_index: int):
        import pyarrow.parquet as pq

        chunk_size = int(self.info.get("chunks_size", 1000))
        table = pq.read_table(
            self._episode_path(episode_index),
            columns=list(STATE_COLUMNS + ACTION_COLUMNS + ("language_instruction",)),
        )
        states = self._column_matrix(table, STATE_COLUMNS)
        actions = self._column_matrix(table, ACTION_COLUMNS)
        instructions = table["language_instruction"].to_pylist()
        videos = [
            self._decode_video(self.root / self._video_relpath(episode_index, key, chunk_size))
            for key in CAMERA_KEYS
        ]
        length = min(len(states), len(actions), len(instructions), *(len(video) for video in videos))
        if length <= 0 or states.shape[1] != ACTION_DIM or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"invalid DROID episode {episode_index}")
        return (
            states[:length],
            actions[:length],
            instructions[:length],
            [video[:length] for video in videos],
        )

    @lru_cache(maxsize=512)
    def _tokenize_instruction(self, instruction: str) -> torch.Tensor:
        prompt = self.prompt_builder_fn("openvla")
        prompt.add_turn("human", f"What action should the robot take to {instruction.strip().lower()}?")
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

    def _make_sample(self, anchor: int, states, actions, instructions, videos, rng: random.Random) -> dict:
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
            torch.stack([transform_frame(video[index]) for index in selected]) for video in videos
        ]
        output = stack_three_view_tubelets(
            transformed_views,
            current_pair=(0, 1),
            future_indices=tuple(range(2, 2 + len(future))),
        )
        output.update(
            input_ids=self._tokenize_instruction(str(instructions[anchor])),
            actions=normalize_q01_q99(
                actions[list(action_indices)], self.stats["action"]["q01"], self.stats["action"]["q99"]
            ),
            proprio=normalize_q01_q99(
                states[anchor], self.stats["proprio"]["q01"], self.stats["proprio"]["q99"]
            ),
            dataset_name=DATASET_NAME,
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
                    len(self.episode_ids),
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
                for candidate_position in candidates:
                    try:
                        episode = self._load_episode(self.episode_ids[candidate_position])
                        break
                    except Exception:
                        continue
                if episode is None:
                    continue
                states, actions, instructions, videos = episode
                anchor = rng.randrange(len(states))
                yield self._make_sample(anchor, states, actions, instructions, videos, rng)
            epoch += 1
