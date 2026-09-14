"""Tar-byte-range Ego dataset for five full-resolution future tubelets."""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EgoTarDatasetConfig:
    parquet_path: str = "/path/to/local_resource"
    current_frames: int = 2
    future_frames: int = 10
    image_size: int = 384
    target_fps: float = 15.0
    max_text_length: int = 128
    seed: int = 7
    max_decode_retries: int = 8

    def __post_init__(self) -> None:
        if self.current_frames != 2 or self.future_frames != 10:
            raise ValueError("Ego five-tubelet training requires 2 current and 10 future frames")
        if self.image_size not in {256, 384}:
            raise ValueError("JEPA-WAM Ego inputs must be 256 or 384px")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")


def sample_frame_indices(
    num_frames: int,
    fps: float,
    config: EgoTarDatasetConfig,
    rng: random.Random,
    anchor_frame: Optional[int] = None,
) -> tuple[list[int], list[int], float]:
    """Sample distinct current/future frames, shrinking stride for short clips."""
    required_intervals = config.current_frames - 1 + config.future_frames
    if num_frames - 1 < required_intervals:
        raise RuntimeError(
            f"video too short: {num_frames} frames; need at least {required_intervals + 1} distinct frames"
        )

    stride = max(1, int(round(float(fps) / config.target_fps)))
    stride = min(stride, (num_frames - 1) // required_intervals)
    first_anchor = (config.current_frames - 1) * stride
    last_anchor = num_frames - 1 - config.future_frames * stride
    if anchor_frame is None:
        anchor = rng.randint(first_anchor, last_anchor)
    else:
        anchor = int(anchor_frame)
        if not first_anchor <= anchor <= last_anchor:
            raise ValueError(f"anchor_frame {anchor} is outside [{first_anchor}, {last_anchor}]")

    current = [anchor - (config.current_frames - 1 - index) * stride for index in range(config.current_frames)]
    future = [anchor + (index + 1) * stride for index in range(config.future_frames)]
    return current, future, anchor / max(float(fps), 1e-6)


def iter_sharded_indices(
    length: int,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    seed: int,
    epoch: int,
) -> Iterator[int]:
    """Yield one deterministic, disjoint rank-and-worker shard per epoch."""
    if length < 0 or world_size <= 0 or num_workers <= 0:
        raise ValueError("invalid length/world_size/num_workers")
    shard_count = world_size * num_workers
    shard_id = rank * num_workers + worker_id
    if not 0 <= shard_id < shard_count:
        raise ValueError("rank or worker id is out of range")
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    yield from order[shard_id::shard_count]


def parse_global_caption(raw: str) -> str:
    text = (raw or "").strip()
    if not text or text[0] not in "[{":
        return text
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if not isinstance(value, dict):
        return text
    global_text = value.get("global")
    if isinstance(global_text, str) and global_text.strip():
        return global_text.strip()
    events = value.get("events")
    if isinstance(events, list):
        parts = [event.get("content", "").strip() for event in events if isinstance(event, dict)]
        combined = " ".join(part for part in parts if part)
        if combined:
            return combined
    return text


def preprocess_frames(frames: torch.Tensor, image_size: int = 384) -> torch.Tensor:
    """Convert `[T,H,W,3]` or `[T,3,H,W]` uint8 frames to JEPA-normalized tensors."""
    if frames.ndim != 4:
        raise ValueError("frames must have four dimensions")
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    if frames.shape[1] != 3:
        raise ValueError("frames must contain RGB channels")
    height, width = frames.shape[-2:]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    result = frames[..., top : top + side, left : left + side].float().div_(255.0)
    result = F.interpolate(result, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    mean = result.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = result.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return result.sub_(mean).div_(std)


def collate_ego_world_model(samples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "current_frames": torch.stack([sample["current_frames"] for sample in samples]),
        "future_frames": torch.stack([sample["future_frames"] for sample in samples]),
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "attention_mask": torch.stack([sample["attention_mask"].bool() for sample in samples]),
    }


class EgoTarIterableDataset(IterableDataset):
    """Stream Ego clips from compact Arrow columns without duplicating a Python index."""

    def __init__(self, config: EgoTarDatasetConfig, tokenizer, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self._load_index()

    def _load_index(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(self.config.parquet_path, columns=["video_path", "offset", "size", "caption"])
        encoded = table["video_path"].dictionary_encode().combine_chunks()
        self.tar_paths = encoded.dictionary.to_pylist()
        self.tar_codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)
        self.offsets = table["offset"].to_numpy(zero_copy_only=False).astype(np.int64)
        self.sizes = table["size"].to_numpy(zero_copy_only=False).astype(np.int64)
        self.captions = table["caption"].cast(pa.large_string()).combine_chunks()

    def __len__(self) -> int:
        return len(self.offsets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _read_bytes(self, index: int) -> bytes:
        path = self.tar_paths[int(self.tar_codes[index])]
        with open(path, "rb") as handle:
            handle.seek(int(self.offsets[index]))
            return handle.read(int(self.sizes[index]))

    @staticmethod
    def _decode_selected(payload: bytes, requested: Sequence[int]) -> tuple[torch.Tensor, float]:
        import av

        with av.open(io.BytesIO(payload)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.thread_count = 1
            fps = float(stream.average_rate or 15.0)
            decoded = []
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in requested:
                    decoded.append((frame_index, torch.from_numpy(frame.to_ndarray(format="rgb24"))))
                if frame_index > requested[-1]:
                    break
        by_index = dict(decoded)
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise RuntimeError(f"decoder missed requested frames: {missing[:4]}")
        return torch.stack([by_index[index] for index in requested]), fps

    def _decode_item(self, index: int, rng: random.Random) -> dict[str, torch.Tensor]:
        import av

        payload = self._read_bytes(index)
        with av.open(io.BytesIO(payload)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or self.config.target_fps)
            num_frames = int(stream.frames or 0)
            if num_frames <= 1 and stream.duration:
                num_frames = int(float(stream.duration * stream.time_base) * fps)
        current_indices, future_indices, _ = sample_frame_indices(num_frames, fps, self.config, rng)
        frames, _ = self._decode_selected(payload, current_indices + future_indices)
        frames = preprocess_frames(frames, self.config.image_size)
        caption = parse_global_caption(self.captions[index].as_py())
        prompt = f"Predict the future visual state: {caption}" if caption else "Predict the future visual state."
        tokens = self.tokenizer(
            prompt,
            max_length=self.config.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "current_frames": frames[: self.config.current_frames],
            "future_frames": frames[self.config.current_frames :],
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0).bool(),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        indices = iter_sharded_indices(
            len(self), self.rank, self.world_size, worker_id, num_workers, self.config.seed, self.epoch
        )
        rng = random.Random(self.config.seed + self.epoch * 1_000_003 + self.rank * 10_007 + worker_id)
        for index in indices:
            last_error = None
            candidate = index
            for _ in range(self.config.max_decode_retries):
                try:
                    yield self._decode_item(candidate, rng)
                    break
                except Exception as error:  # corrupt clips are replaced by another real clip, never zero frames
                    last_error = error
                    candidate = rng.randrange(len(self))
            else:
                raise RuntimeError(f"Ego decode failed after retries; last error: {last_error!r}")
