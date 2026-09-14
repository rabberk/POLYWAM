"""Paper-faithful RoboTwin Clean-20 adapter for JEPA-WAM."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import IterableDataset, get_worker_info

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, NUM_TOKENS


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DATASET_NAME = "robotwin_qpos14_fastwam"
PAPER20_GROUP1_TASK_BLOCKS = (
    0, 1, 4, 5, 6, 7, 9, 11, 22, 23,
)
PAPER20_TASK_BLOCKS = PAPER20_GROUP1_TASK_BLOCKS + (
    24, 26, 29, 32, 36, 37, 42, 43, 46, 47,
)
PAPER20_GROUP2_TASK_BLOCKS = PAPER20_TASK_BLOCKS[len(PAPER20_GROUP1_TASK_BLOCKS) :]


@dataclass(frozen=True)
class TemporalIndices:
    current: int
    future: int
    action: tuple[int, ...]


@dataclass(frozen=True)
class FiveTubeletTemporalIndices:
    current_pair: tuple[int, int]
    future: tuple[int, ...]
    action: tuple[int, ...]


@dataclass(frozen=True)
class WeakPhotometricParams:
    """Geometry-preserving appearance perturbations shared by a temporal sample."""

    brightness: float
    contrast: float
    saturation: float
    gamma: float
    blur_radius: float
    noise_std: float
    noise_seed: int


def sample_weak_photometric_params(
    rng: random.Random,
    *,
    probability: float,
    strength: float,
) -> WeakPhotometricParams | None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("photometric augmentation probability must be in [0, 1]")
    if not 0.0 <= strength <= 0.5:
        raise ValueError("photometric augmentation strength must be in [0, 0.5]")
    if rng.random() >= probability:
        return None

    factor = lambda scale=1.0: 1.0 + rng.uniform(-strength, strength) * scale
    return WeakPhotometricParams(
        brightness=factor(),
        contrast=factor(),
        saturation=factor(),
        gamma=factor(0.5),
        blur_radius=rng.uniform(0.1, 0.5) if rng.random() < 0.15 else 0.0,
        noise_std=rng.uniform(0.5, max(0.5, strength * 12.0)) if rng.random() < 0.25 else 0.0,
        noise_seed=rng.randrange(2**31),
    )


def apply_weak_photometric_augmentation(
    image: Image.Image,
    params: WeakPhotometricParams | None,
) -> Image.Image:
    """Apply appearance-only augmentation without changing image geometry."""
    image = image.convert("RGB")
    if params is None:
        return image
    output = ImageEnhance.Brightness(image).enhance(params.brightness)
    output = ImageEnhance.Contrast(output).enhance(params.contrast)
    output = ImageEnhance.Color(output).enhance(params.saturation)
    if params.gamma != 1.0:
        gamma_lut = [min(255, max(0, round(255.0 * (value / 255.0) ** params.gamma))) for value in range(256)]
        output = output.point(gamma_lut * 3)
    if params.blur_radius > 0.0:
        output = output.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))
    if params.noise_std > 0.0:
        pixels = np.asarray(output, dtype=np.float32)
        noise = np.random.default_rng(params.noise_seed).normal(0.0, params.noise_std, pixels.shape)
        output = Image.fromarray(np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8))
    return output


def resolve_robotwin_root(path: str | Path) -> Path:
    root = Path(path)
    candidates = (root, root / "robotwin2.0")
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"missing RoboTwin metadata below {root}; expected meta/info.json or robotwin2.0/meta/info.json"
    )


def select_paper20_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    if total_episodes <= 0 or total_episodes % block_size:
        raise ValueError(f"RoboTwin requires complete {block_size}-episode task blocks")
    if not 0 < clean_per_block <= block_size:
        raise ValueError("clean_per_block must fit inside each task block")
    if max(PAPER20_TASK_BLOCKS) >= total_episodes // block_size:
        raise ValueError("dataset does not contain every JEPA-WAM RoboTwin task")
    return tuple(
        task_block * block_size + offset
        for task_block in PAPER20_TASK_BLOCKS
        for offset in range(clean_per_block)
    )


def select_paper20_group1_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    if total_episodes <= 0 or total_episodes % block_size:
        raise ValueError(f"RoboTwin requires complete {block_size}-episode task blocks")
    if not 0 < clean_per_block <= block_size:
        raise ValueError("clean_per_block must fit inside each task block")
    if max(PAPER20_GROUP1_TASK_BLOCKS) >= total_episodes // block_size:
        raise ValueError("dataset does not contain every first-group JEPA-WAM RoboTwin task")
    return tuple(
        task_block * block_size + offset
        for task_block in PAPER20_GROUP1_TASK_BLOCKS
        for offset in range(clean_per_block)
    )


def select_paper20_group2_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    if total_episodes <= 0 or total_episodes % block_size:
        raise ValueError(f"RoboTwin requires complete {block_size}-episode task blocks")
    if not 0 < clean_per_block <= block_size:
        raise ValueError("clean_per_block must fit inside each task block")
    if max(PAPER20_GROUP2_TASK_BLOCKS) >= total_episodes // block_size:
        raise ValueError("dataset does not contain every second-group JEPA-WAM RoboTwin task")
    return tuple(
        task_block * block_size + offset
        for task_block in PAPER20_GROUP2_TASK_BLOCKS
        for offset in range(clean_per_block)
    )


def build_temporal_indices(anchor: int, episode_length: int, offset: int = 50, horizon: int = 50) -> TemporalIndices:
    if episode_length <= 0 or not 0 <= anchor < episode_length:
        raise ValueError("anchor must index a non-empty episode")
    if offset <= 0 or horizon <= 0:
        raise ValueError("offset and horizon must be positive")
    last = episode_length - 1
    return TemporalIndices(
        current=anchor,
        future=min(anchor + offset, last),
        action=tuple(min(anchor + step, last) for step in range(horizon)),
    )


def build_five_tubelet_indices(anchor: int, episode_length: int) -> FiveTubeletTemporalIndices:
    """Build five JEPA-WAM targets that share the current-frame anchor."""
    if episode_length <= 0 or not 0 <= anchor < episode_length:
        raise ValueError("anchor must index a non-empty episode")
    clamp = lambda index: min(max(index, 0), episode_length - 1)
    future_offsets = (10, 20, 30, 40, 50)
    return FiveTubeletTemporalIndices(
        current_pair=(clamp(anchor - 1), anchor),
        future=tuple(clamp(anchor + offset) for offset in future_offsets),
        action=tuple(clamp(anchor + step) for step in range(50)),
    )


def normalize_q01_q99(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if values.shape[-1] != low.shape[-1] or low.shape != high.shape:
        raise ValueError("value and q01/q99 dimensions must match")
    if np.any(high < low):
        raise ValueError("q99 must be greater than or equal to q01")
    span = high - low
    normalized = np.where(np.abs(span) > 1e-8, 2.0 * (values - low) / np.maximum(span, 1e-8) - 1.0, 0.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def iter_sharded_episodes(
    total_episodes: int,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    *,
    seed: int,
    epoch: int,
) -> Iterator[int]:
    if total_episodes < 0 or world_size <= 0 or num_workers <= 0:
        raise ValueError("invalid shard dimensions")
    shard_count = world_size * num_workers
    shard_id = rank * num_workers + worker_id
    if not 0 <= shard_id < shard_count:
        raise ValueError("rank or worker id is out of range")
    order = list(range(total_episodes))
    random.Random(seed + epoch).shuffle(order)
    yield from order[shard_id::shard_count]


def stack_three_view_sample(
    transformed_views: Sequence[torch.Tensor],
    current_index: int,
    future_index: int,
) -> dict[str, torch.Tensor]:
    if len(transformed_views) != 3:
        raise ValueError("RoboTwin paper training requires exactly three camera views")
    for view in transformed_views:
        if view.ndim != 4:
            raise ValueError("each transformed view must have shape [T,C,H,W]")
    primary = transformed_views[0]
    wrists = transformed_views[1:]
    return {
        "pixel_values": primary[current_index],
        "pixel_values_wrist": torch.stack([view[current_index] for view in wrists]),
        "pair_pixel_values": primary[[current_index, future_index]],
        "pair_pixel_values_wrist": torch.stack(
            [view[[current_index, future_index]] for view in wrists]
        ),
    }


def stack_three_view_tubelets(
    transformed_views: Sequence[torch.Tensor],
    current_pair: tuple[int, int],
    future_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Package three camera streams as one observed pair and five future anchors."""
    if len(transformed_views) != 3:
        raise ValueError("RoboTwin five-tubelet training requires exactly three camera views")
    if len(current_pair) != 2 or len(future_indices) != 5:
        raise ValueError("fixed-anchor training requires 2 current and 5 future frames")
    for view in transformed_views:
        if view.ndim != 4:
            raise ValueError("each transformed view must have shape [T,C,H,W]")
    current = torch.stack([view[list(current_pair)] for view in transformed_views])
    future = torch.stack([view[list(future_indices)] for view in transformed_views])
    return {
        "pixel_values": current[0, 1],
        "pixel_values_wrist": current[1:, 1],
        "current_frame_pairs": current,
        "future_frames": future,
    }


def _merge_qpos_fields(arm: Sequence[float], gripper: Sequence[float]) -> np.ndarray:
    arm = np.asarray(arm, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    if arm.shape != (12,) or gripper.shape != (2,):
        raise ValueError("split qpos statistics must contain 12 arm and 2 gripper values")
    return np.concatenate((arm[:6], gripper[:1], arm[6:], gripper[1:])).astype(np.float32)


def load_robotwin_statistics(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(Path(path).read_text())
    if DATASET_NAME in payload:
        stats = payload[DATASET_NAME]
    elif "norm_stats" in payload:
        split = payload["norm_stats"]

        def merge(prefix: str, field: str) -> np.ndarray:
            return _merge_qpos_fields(
                split[f"{prefix}.arm.position"][field],
                split[f"{prefix}.effector.position"][field],
            )

        stats = {
            "action": {field: merge("action", field) for field in ("mean", "std", "min", "max", "q01", "q99")},
            "proprio": {
                field: merge("observation.state", field)
                for field in ("mean", "std", "min", "max", "q01", "q99")
            },
        }
    else:
        raise ValueError("unrecognized RoboTwin normalization file")
    for group in ("action", "proprio"):
        for field, values in stats[group].items():
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (14,):
                raise ValueError(f"{group}.{field} must be 14D")
            stats[group][field] = array
    return stats


@dataclass(frozen=True)
class RoboTwinPaperDatasetConfig:
    root: Path
    stats_path: Path
    seed: int = 7
    num_workers: int = 4
    prefetch_factor: int = 2
    max_decode_retries: int = 8
    enable_five_tubelet_ar: bool = False
    enable_photometric_augmentation: bool = False
    photometric_augmentation_probability: float = 0.4
    photometric_augmentation_strength: float = 0.1
    episode_selection: str = "robotwin_paper20_clean"


class RoboTwinPaperDataset(IterableDataset):
    """Stream the 1,000 Clean demonstrations used by the paper."""

    def __init__(
        self,
        config: RoboTwinPaperDatasetConfig,
        image_transform,
        tokenizer,
        prompt_builder_fn,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.root = resolve_robotwin_root(config.root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self._validate_metadata()
        self.stats = load_robotwin_statistics(config.stats_path)
        self.dataset_statistics = {DATASET_NAME: self.stats}
        self.dataloader_num_workers = int(config.num_workers)
        self.dataloader_prefetch_factor = int(config.prefetch_factor)
        self.dataloader_pin_memory = True
        if config.episode_selection == "robotwin_paper20_clean":
            self.episode_ids = select_paper20_clean_episode_ids(int(self.info["total_episodes"]))
        elif config.episode_selection == "robotwin_paper20_group1_clean":
            self.episode_ids = select_paper20_group1_clean_episode_ids(int(self.info["total_episodes"]))
        elif config.episode_selection == "robotwin_paper20_group2_clean":
            self.episode_ids = select_paper20_group2_clean_episode_ids(int(self.info["total_episodes"]))
        else:
            raise ValueError(f"unsupported RoboTwin paper episode selection: {config.episode_selection}")
        self._task_rows = [json.loads(line) for line in (self.root / "meta" / "tasks.jsonl").read_text().splitlines()]
        episode_rows = [json.loads(line) for line in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()]
        self._episode_lengths = np.asarray([int(row["length"]) for row in episode_rows], dtype=np.int64)
        self.global_dataset_length = int(self._episode_lengths[list(self.episode_ids)].sum())
        self.dataset_length = self.global_dataset_length

    def _validate_metadata(self) -> None:
        if self.info.get("codebase_version") != "v2.1" or int(self.info.get("fps", -1)) != 50:
            raise ValueError("RoboTwin paper adapter requires LeRobot v2.1 at 50 FPS")
        features = self.info.get("features", {})
        if features.get("observation.state", {}).get("shape") != [14]:
            raise ValueError("RoboTwin observation.state must be 14D")
        if features.get("action", {}).get("shape") != [14]:
            raise ValueError("RoboTwin action must be 14D")
        if any(key not in features for key in CAMERA_KEYS):
            raise ValueError("RoboTwin metadata is missing one of the three paper camera views")

    def __len__(self) -> int:
        return self.dataset_length

    def _episode_path(self, episode_index: int) -> Path:
        template = self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        chunk_size = int(self.info.get("chunks_size", 1000))
        return self.root / template.format(episode_chunk=episode_index // chunk_size, episode_index=episode_index)

    def _video_path(self, episode_index: int, camera_key: str) -> Path:
        template = self.info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        chunk_size = int(self.info.get("chunks_size", 1000))
        return self.root / template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=camera_key,
        )

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

    def _load_episode(self, episode_index: int):
        import pyarrow.parquet as pq

        table = pq.read_table(
            self._episode_path(episode_index), columns=["observation.state", "action", "task_index"]
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task_indices = table["task_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        videos = [self._decode_video(self._video_path(episode_index, key)) for key in CAMERA_KEYS]
        length = min(len(states), len(actions), len(task_indices), *(len(video) for video in videos))
        if length <= 0 or states.shape[1:] != (14,) or actions.shape[1:] != (14,):
            raise ValueError(f"invalid qpos14 episode {episode_index}")
        return states[:length], actions[:length], task_indices[:length], [video[:length] for video in videos]

    @lru_cache(maxsize=256)
    def _tokenize_task(self, task_index: int) -> torch.Tensor:
        row = self._task_rows[task_index]
        instruction = str(row["task"]).strip().lower()
        prompt = self.prompt_builder_fn("openvla")
        prompt.add_turn("human", f"What action should the robot take to {instruction}?")
        prompt.add_turn("gpt", "")
        input_ids = self.tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids
        input_ids.extend([ACTION_TOKEN_BEGIN_IDX] * NUM_TOKENS)
        return torch.tensor(input_ids, dtype=torch.long)

    def _make_sample(self, anchor: int, states, actions, task_indices, videos, rng: random.Random) -> dict:
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

        if self.config.enable_five_tubelet_ar:
            indices = build_five_tubelet_indices(anchor, len(states))
            selected = list(indices.current_pair) + list(indices.future)
            transformed_views = [
                torch.stack(
                    [transform_frame(video[index]) for index in selected]
                )
                for video in videos
            ]
            output = stack_three_view_tubelets(
                transformed_views,
                current_pair=(0, 1),
                future_indices=tuple(range(2, 7)),
            )
            action_indices = indices.action
        else:
            indices = build_temporal_indices(anchor, len(states))
            transformed_views = []
            for video in videos:
                transformed_views.append(
                    torch.stack(
                        [
                            transform_frame(video[indices.current]),
                            transform_frame(video[indices.future]),
                        ]
                    )
                )
            output = stack_three_view_sample(transformed_views, 0, 1)
            action_indices = indices.action
        output.update(
            input_ids=self._tokenize_task(int(task_indices[anchor])),
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
                    len(self.episode_ids), self.rank, self.world_size, worker_id, num_workers,
                    seed=self.config.seed, epoch=epoch,
                )
            )
            rng = random.Random(self.config.seed + epoch * 1_000_003 + self.rank * 10_007 + worker_id)
            for position_index, position in enumerate(positions):
                candidates = positions[position_index:position_index + self.config.max_decode_retries]
                for candidate_position in candidates:
                    try:
                        episode = self._load_episode(self.episode_ids[candidate_position])
                        break
                    except Exception:
                        episode = None
                if episode is None:
                    continue
                states, actions, task_indices, videos = episode
                anchors = list(range(len(states)))
                rng.shuffle(anchors)
                for anchor in anchors:
                    yield self._make_sample(anchor, states, actions, task_indices, videos, rng)
            epoch += 1
