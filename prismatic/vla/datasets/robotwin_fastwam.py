"""Streaming LeRobot v2.1 adapter for FastWAM RoboTwin qpos14 data."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info

from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, NUM_TOKENS
from prismatic.vla.datasets.ego_tar import preprocess_frames


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DATASET_NAME = "robotwin_qpos14_fastwam"

# Block order in anonymous/robotwin2.0-fastwam. These are the inferred first 10
# tasks in JEPA-WAM's two-group RoboTwin evaluation. Each 550-episode block
# stores 50 clean episodes followed by 500 randomized episodes.
PAPER20_GROUP1_TASK_BLOCKS = (
    0,   # adjust_bottle
    1,   # beat_block_hammer
    4,   # click_alarmclock
    5,   # click_bell
    6,   # dump_bin_bigbin
    7,   # grab_roller
    9,   # handover_mic
    11,  # lift_pot
    22,  # place_bread_basket
    23,  # place_bread_skillet
)

# All 20 tasks reported by JEPA-WAM, ordered as in the paper's result table.
PAPER20_TASK_BLOCKS = PAPER20_GROUP1_TASK_BLOCKS + (
    24,  # place_burger_fries
    26,  # place_cans_plasticbox
    29,  # place_empty_cup
    32,  # place_object_basket
    36,  # place_shoe
    37,  # press_stapler
    42,  # shake_bottle
    43,  # shake_bottle_horizontally
    46,  # stack_bowls_three
    47,  # stack_bowls_two
)


def select_robotwin_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    """Select the first 50 clean episodes from every 550-episode task block."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if block_size <= 0 or clean_per_block <= 0 or clean_per_block > block_size:
        raise ValueError("clean episode block dimensions are invalid")
    if total_episodes % block_size:
        raise ValueError(f"RoboTwin clean selection requires complete {block_size}-episode blocks")
    return tuple(
        block_start + offset
        for block_start in range(0, total_episodes, block_size)
        for offset in range(clean_per_block)
    )


def select_robotwin_paper20_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    """Select clean episodes for the 20 RoboTwin tasks reported by JEPA-WAM."""
    if total_episodes % block_size:
        raise ValueError(f"RoboTwin clean selection requires complete {block_size}-episode blocks")
    task_count = total_episodes // block_size
    if max(PAPER20_TASK_BLOCKS) >= task_count:
        raise ValueError("RoboTwin dataset does not contain every JEPA-WAM paper task block")
    return tuple(
        task_block * block_size + offset
        for task_block in PAPER20_TASK_BLOCKS
        for offset in range(clean_per_block)
    )


def select_robotwin_paper20_group1_clean_episode_ids(
    total_episodes: int,
    block_size: int = 550,
    clean_per_block: int = 50,
) -> tuple[int, ...]:
    """Select clean episodes for the inferred first JEPA-WAM task group."""
    if total_episodes % block_size:
        raise ValueError(f"RoboTwin clean selection requires complete {block_size}-episode blocks")
    task_count = total_episodes // block_size
    if max(PAPER20_GROUP1_TASK_BLOCKS) >= task_count:
        raise ValueError("RoboTwin dataset does not contain every JEPA-WAM Group 1 task block")
    return tuple(
        task_block * block_size + offset
        for task_block in PAPER20_GROUP1_TASK_BLOCKS
        for offset in range(clean_per_block)
    )


def validate_clean_statistics_provenance(
    path: str | Path,
    *,
    source_total_episodes: int,
    episode_count: int,
    frame_count: int,
) -> None:
    payload = json.loads(Path(path).read_text())
    expected = {
        "episode_selection": "robotwin_clean",
        "source_total_episodes": int(source_total_episodes),
        "episode_count": int(episode_count),
        "count": int(frame_count),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"clean normalization provenance mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True)
class TemporalIndices:
    current_pair: tuple[int, int]
    future: tuple[int, ...]
    action: tuple[int, ...]


@dataclass(frozen=True)
class FastWAMConfig:
    root: Path
    stats_path: Path
    image_size: int = 384
    episode_selection: str = "robotwin_clean"
    action_horizon: int = 50
    max_text_length: int = 128
    seed: int = 7
    max_decode_retries: int = 8
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 1
    dataloader_pin_memory: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "stats_path", Path(self.stats_path))
        if self.image_size not in {256, 384}:
            raise ValueError("JEPA-WAM RoboTwin inputs must be 256 or 384px")
        if self.episode_selection not in {
            "robotwin_clean",
            "robotwin_paper20_clean",
            "robotwin_paper20_group1_clean",
        }:
            raise ValueError(
                "JEPA-WAM RoboTwin post-training requires clean-only episode selection"
            )
        if self.action_horizon != 50:
            raise ValueError("RoboTwin action horizon must be 50")
        if self.max_decode_retries <= 0:
            raise ValueError("max_decode_retries must be positive")
        if self.dataloader_num_workers <= 0:
            raise ValueError("dataloader_num_workers must be positive")
        if self.dataloader_prefetch_factor <= 0:
            raise ValueError("dataloader_prefetch_factor must be positive")


def build_temporal_indices(anchor: int, episode_length: int) -> TemporalIndices:
    if episode_length <= 0 or not 0 <= anchor < episode_length:
        raise ValueError("anchor must index a non-empty episode")
    clamp = lambda index: min(max(index, 0), episode_length - 1)
    future_offsets = (9, 10, 19, 20, 29, 30, 39, 40, 49, 50)
    return TemporalIndices(
        current_pair=(clamp(anchor - 1), anchor),
        future=tuple(clamp(anchor + offset) for offset in future_offsets),
        action=tuple(clamp(anchor + offset) for offset in range(50)),
    )


def normalize_q01_q99(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if values.shape[-1] != low.shape[-1] or low.shape != high.shape:
        raise ValueError("value and q01/q99 dimensions must match")
    if np.any(high < low):
        raise ValueError("q99 must be greater than or equal to q01 in every dimension")
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
        raise ValueError("invalid episode or shard counts")
    shard_count = world_size * num_workers
    shard_id = rank * num_workers + worker_id
    if not 0 <= shard_id < shard_count:
        raise ValueError("rank or worker id is out of range")
    order = list(range(total_episodes))
    random.Random(seed + epoch).shuffle(order)
    yield from order[shard_id::shard_count]


def worker_resume_sample_offset(
    resume_micro_batches: int,
    batch_size: int,
    worker_id: int,
    num_workers: int,
) -> int:
    """Samples consumed from one worker under round-robin DataLoader dispatch."""
    if resume_micro_batches < 0 or batch_size <= 0 or num_workers <= 0:
        raise ValueError("invalid resume cursor")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker id is out of range")
    full_cycles, remainder = divmod(resume_micro_batches, num_workers)
    return (full_cycles + int(worker_id < remainder)) * batch_size


def validate_fastwam_metadata(root: str | Path) -> dict:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing FastWAM metadata: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"FastWAM must use LeRobot v2.1, got {info.get('codebase_version')!r}")
    if int(info.get("fps", -1)) != 50:
        raise ValueError(f"FastWAM must use 50 FPS, got {info.get('fps')!r}")
    features = info.get("features", {})
    if features.get("observation.state", {}).get("shape") != [14]:
        raise ValueError("FastWAM observation.state must be 14D")
    if features.get("action", {}).get("shape") != [14]:
        raise ValueError("FastWAM action must be 14D")
    missing_cameras = [key for key in CAMERA_KEYS if key not in features]
    if missing_cameras:
        raise ValueError(f"FastWAM camera contract is incomplete: {missing_cameras}")
    image_keys = tuple(key for key in features if key.startswith("observation.images."))
    if set(image_keys) != set(CAMERA_KEYS):
        raise ValueError(f"FastWAM must expose exactly the three supported cameras, got {image_keys}")
    if int(info.get("total_episodes", 0)) <= 0:
        raise ValueError("FastWAM must contain at least one episode")
    return info


def _merge_qpos_fields(arm: Sequence[float], gripper: Sequence[float]) -> np.ndarray:
    arm = np.asarray(arm, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    if arm.shape != (12,) or gripper.shape != (2,):
        raise ValueError("FastWAM split statistics must contain 12 arm and 2 gripper dimensions")
    return np.concatenate((arm[:6], gripper[:1], arm[6:], gripper[1:])).astype(np.float32)


def load_fastwam_statistics(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if "robotwin_qpos14_fastwam" in payload:
        stats = payload["robotwin_qpos14_fastwam"]
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
            "num_transitions": int(payload.get("count", 0)),
        }
    else:
        stats = payload
    for name in ("action", "proprio"):
        if name not in stats:
            raise ValueError(f"statistics are missing {name}")
        for field in ("q01", "q99"):
            values = np.asarray(stats[name][field], dtype=np.float32)
            if values.shape != (14,):
                raise ValueError(f"{name}.{field} must be 14D")
            stats[name][field] = values
        for field in ("mean", "std", "min", "max"):
            if field in stats[name]:
                stats[name][field] = np.asarray(stats[name][field], dtype=np.float32)
    return stats


class FastWAMCollator:
    def __init__(
        self,
        pad_token_id: int,
        model_max_length: int,
        compile_stable_length: int | None = None,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.model_max_length = int(model_max_length)
        self.compile_stable_length = (
            None if compile_stable_length is None else int(compile_stable_length)
        )
        if self.compile_stable_length is not None and not (
            0 < self.compile_stable_length <= self.model_max_length
        ):
            raise ValueError("compile_stable_length must be within model_max_length")

    def _pad(self, samples: Sequence[dict], key: str) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [sample[key] for sample in samples]
        if self.compile_stable_length is not None:
            longest = max(sequence.numel() for sequence in sequences)
            if longest > self.compile_stable_length:
                raise ValueError(
                    f"{key} length {longest} exceeds compile-stable length "
                    f"{self.compile_stable_length}"
                )
        result = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        target_length = self.compile_stable_length or min(result.shape[1], self.model_max_length)
        result = result[:, :target_length]
        if result.shape[1] < target_length:
            padding = result.new_full(
                (result.shape[0], target_length - result.shape[1]), self.pad_token_id
            )
            result = torch.cat((result, padding), dim=1)
        return result, result.ne(self.pad_token_id)

    def __call__(self, samples: Sequence[dict]) -> dict[str, torch.Tensor | list[str]]:
        if not samples:
            raise ValueError("cannot collate an empty FastWAM batch")
        action_ids, action_mask = self._pad(samples, "action_input_ids")
        world_ids, world_mask = self._pad(samples, "world_input_ids")
        return {
            "input_ids": action_ids,
            "attention_mask": action_mask,
            "world_input_ids": world_ids,
            "world_attention_mask": world_mask,
            "current_frame_pairs": torch.stack([sample["current_frame_pairs"] for sample in samples]),
            "future_frames": torch.stack([sample["future_frames"] for sample in samples]),
            "proprio": torch.from_numpy(np.stack([sample["proprio"] for sample in samples])).float(),
            "actions": torch.from_numpy(np.stack([sample["actions"] for sample in samples])).float(),
            "dataset_names": [sample["dataset_name"] for sample in samples],
        }


class FastWAMRoboTwinDataset(IterableDataset):
    """Decode one FastWAM episode per shard and emit infinite shuffled epochs."""

    def __init__(self, config: FastWAMConfig, tokenizer, prompt_builder_fn, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.info = validate_fastwam_metadata(config.root)
        self.stats = load_fastwam_statistics(config.stats_path)
        self.dataset_statistics = {DATASET_NAME: self.stats}
        self.dataloader_num_workers = config.dataloader_num_workers
        self.dataloader_prefetch_factor = config.dataloader_prefetch_factor
        self.dataloader_pin_memory = config.dataloader_pin_memory
        self.start_epoch = 0
        self.resume_micro_batches = 0
        self.resume_batch_size = 1
        self._task_offsets = self._index_jsonl(config.root / "meta" / "tasks.jsonl")
        self._episode_lengths = self._index_episode_lengths(config.root / "meta" / "episodes.jsonl")
        if len(self._episode_lengths) != int(self.info["total_episodes"]):
            raise ValueError("FastWAM episode metadata count does not match info.json")
        all_clean_episode_ids = select_robotwin_clean_episode_ids(int(self.info["total_episodes"]))
        if config.episode_selection == "robotwin_paper20_clean":
            self.episode_ids = select_robotwin_paper20_clean_episode_ids(
                int(self.info["total_episodes"])
            )
        elif config.episode_selection == "robotwin_paper20_group1_clean":
            self.episode_ids = select_robotwin_paper20_group1_clean_episode_ids(
                int(self.info["total_episodes"])
            )
        else:
            self.episode_ids = all_clean_episode_ids
        self.global_dataset_length = int(self._episode_lengths[list(self.episode_ids)].sum())
        self.dataset_length = self.global_dataset_length
        # The user explicitly keeps the existing all-clean normalization file
        # when narrowing training to the paper's 20-task subset.
        stats_frame_count = int(self._episode_lengths[list(all_clean_episode_ids)].sum())
        validate_clean_statistics_provenance(
            config.stats_path,
            source_total_episodes=int(self.info["total_episodes"]),
            episode_count=len(all_clean_episode_ids),
            frame_count=stats_frame_count,
        )

    @staticmethod
    def _index_jsonl(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"missing FastWAM task metadata: {path}")
        offsets = []
        offset = 0
        with path.open("rb") as handle:
            for line in handle:
                offsets.append(offset)
                offset += len(line)
        return np.asarray(offsets, dtype=np.int64)

    @staticmethod
    def _index_episode_lengths(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"missing FastWAM episode metadata: {path}")
        lengths = []
        with path.open("rb") as handle:
            for line in handle:
                try:
                    lengths.append(int(line.rsplit(b'"length":', 1)[1].split(b"}", 1)[0]))
                except (IndexError, ValueError) as error:
                    raise ValueError(f"invalid episode length metadata at row {len(lengths)}") from error
        values = np.asarray(lengths, dtype=np.int64)
        if values.size == 0 or np.any(values <= 0):
            raise ValueError("FastWAM episode lengths must be positive")
        return values

    def __len__(self) -> int:
        return self.dataset_length

    def _episode_path(self, episode_index: int) -> Path:
        template = self.info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        chunk_size = int(self.info.get("chunks_size", 1000))
        return self.config.root / template.format(
            episode_chunk=episode_index // chunk_size, episode_index=episode_index
        )

    def _video_path(self, episode_index: int, camera_key: str) -> Path:
        template = self.info.get(
            "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        chunk_size = int(self.info.get("chunks_size", 1000))
        return self.config.root / template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=camera_key,
        )

    @lru_cache(maxsize=4096)
    def _task_text(self, task_index: int) -> str:
        if not 0 <= task_index < len(self._task_offsets):
            raise ValueError(f"task index {task_index} is outside metadata")
        path = self.config.root / "meta" / "tasks.jsonl"
        with path.open("rb") as handle:
            handle.seek(int(self._task_offsets[task_index]))
            row = json.loads(handle.readline())
        if int(row["task_index"]) != task_index or not str(row["task"]).strip():
            raise ValueError(f"invalid task metadata at index {task_index}")
        return str(row["task"]).strip()

    @staticmethod
    def _decode_video(path: Path) -> torch.Tensor:
        import av

        decoded = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
        if not decoded:
            raise RuntimeError(f"decoded no frames from {path}")
        return torch.stack(decoded)

    def _load_episode(self, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[torch.Tensor]]:
        import pyarrow.parquet as pq

        table = pq.read_table(
            self._episode_path(episode_index), columns=["observation.state", "action", "task_index"]
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task_indices = table["task_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        if states.ndim != 2 or states.shape[1] != 14 or actions.shape != states.shape:
            raise ValueError(f"episode {episode_index} does not contain aligned qpos14 state/action")
        videos = [self._decode_video(self._video_path(episode_index, key)) for key in CAMERA_KEYS]
        length = min(len(states), *(video.shape[0] for video in videos))
        if length <= 0:
            raise RuntimeError(f"episode {episode_index} is empty")
        return states[:length], actions[:length], task_indices[:length], [video[:length] for video in videos]

    @lru_cache(maxsize=4096)
    def _tokenize_task(self, task_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        instruction = self._task_text(task_index)
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        prompt_builder.add_turn("gpt", "")
        action_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        action_ids.extend([ACTION_TOKEN_BEGIN_IDX] * NUM_TOKENS)
        world_prompt = f"Predict the future visual state: {instruction}"
        world_ids = self.tokenizer(
            world_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_text_length,
        ).input_ids
        return torch.tensor(action_ids, dtype=torch.long), torch.tensor(world_ids, dtype=torch.long)

    def _make_sample(
        self,
        anchor: int,
        states: np.ndarray,
        actions: np.ndarray,
        task_indices: np.ndarray,
        videos: list[torch.Tensor],
    ) -> dict:
        indices = build_temporal_indices(anchor, len(states))
        selected = list(indices.current_pair) + list(indices.future)
        processed_views = [preprocess_frames(video[selected], self.config.image_size) for video in videos]
        stacked = torch.stack(processed_views).to(torch.bfloat16)
        action_ids, world_ids = self._tokenize_task(int(task_indices[anchor]))
        return {
            "action_input_ids": action_ids,
            "world_input_ids": world_ids,
            "current_frame_pairs": stacked[:, :2],
            "future_frames": stacked[:, 2:],
            "proprio": normalize_q01_q99(
                states[anchor], self.stats["proprio"]["q01"], self.stats["proprio"]["q99"]
            ),
            "actions": normalize_q01_q99(
                actions[list(indices.action)], self.stats["action"]["q01"], self.stats["action"]["q99"]
            ),
            "dataset_name": DATASET_NAME,
        }

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        epoch = int(self.start_epoch)
        remaining_skip = worker_resume_sample_offset(
            int(self.resume_micro_batches),
            int(self.resume_batch_size),
            worker_id,
            num_workers,
        )
        while True:
            rng = random.Random(self.config.seed + epoch * 1_000_003 + self.rank * 10_007 + worker_id)
            episode_positions = iter_sharded_episodes(
                len(self.episode_ids),
                self.rank,
                self.world_size,
                worker_id,
                num_workers,
                seed=self.config.seed,
                epoch=epoch,
            )
            episodes = [self.episode_ids[position] for position in episode_positions]
            consumed_episodes: set[int] = set()
            for episode_position, episode_index in enumerate(episodes):
                if episode_index in consumed_episodes:
                    # A successful forward fallback already shuffled and yielded this
                    # episode, so advancing RNG here would consume its permutation twice.
                    continue
                candidate_window = episodes[
                    episode_position : episode_position + self.config.max_decode_retries
                ]
                for candidate in candidate_window:
                    if candidate in consumed_episodes:
                        continue
                    try:
                        states, actions, task_indices, videos = self._load_episode(candidate)
                        break
                    except Exception:
                        continue
                else:
                    # This corrupt slot has no viable forward fallback. Leave it out of
                    # this epoch instead of wrapping to and duplicating an earlier sample.
                    continue
                consumed_episodes.add(candidate)
                anchors = list(range(len(states)))
                rng.shuffle(anchors)
                if remaining_skip >= len(anchors):
                    # Count against the actual decoded/fallback stream. Source metadata
                    # can describe a corrupt episode whose fallback has another length.
                    remaining_skip -= len(anchors)
                    continue
                if remaining_skip:
                    anchors = anchors[remaining_skip:]
                    remaining_skip = 0
                for anchor in anchors:
                    yield self._make_sample(anchor, states, actions, task_indices, videos)
            epoch += 1
