"""TorchCodec dataset backed by the validated JSONL manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .sampling import TemporalSampler, TemporalSamplingConfig
from .labels import label_for


@dataclass(frozen=True)
class ManifestRecord:
    video_id: str
    category: str
    label: int
    video_path: str
    source_sequence: str
    num_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ManifestRecord(**json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid manifest record at {path}:{line_number}: {exc}") from exc
            if record.video_id in seen_ids:
                raise ValueError(f"Duplicate video_id: {record.video_id}")
            expected_label = label_for(record.category)
            if type(record.label) is not int or record.label != expected_label:
                raise ValueError(f"Wrong label for {record.video_id}: {record.category} requires {expected_label}, got {record.label}")
            seen_ids.add(record.video_id)
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records


class VideoManifestDataset(Dataset[dict[str, Any]]):
    """Decode fixed-length clips on CPU and retain all non-label metadata.

    Returned `video` shape is `[T,C,H,W]`. When an official processor is
    supplied, `pixel_values_videos` is also `[T,C,256,256]` and batches collate
    to the Hugging Face model input `[B,T,C,256,256]`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        dataset_root: str | Path | None = None,
        sampling: TemporalSamplingConfig | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 0,
        decoder_threads: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root is not None
            else self.manifest_path.parent.resolve()
        )
        self.records = read_manifest(self.manifest_path)
        self.sampler = TemporalSampler(sampling or TemporalSamplingConfig(), seed=seed)
        self.transform = transform
        self.decoder_threads = decoder_threads

        for record in self.records:
            relative = Path(record.video_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe video_path in manifest: {record.video_path}")
            resolved = (self.dataset_root / relative).resolve()
            if not resolved.is_relative_to(self.dataset_root):
                raise ValueError(f"video_path escapes dataset root: {record.video_path}")
            if not resolved.is_file():
                raise FileNotFoundError(resolved)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as exc:
            raise ImportError("TorchCodec is required to decode the generated MP4 dataset") from exc

        record = self.records[index]
        path = self.dataset_root / record.video_path
        decoder = VideoDecoder(
            path,
            dimension_order="NCHW",
            num_ffmpeg_threads=self.decoder_threads,
            device="cpu",
            seek_mode="exact",
        )
        metadata = decoder.metadata
        frame_count = len(decoder)
        source_fps = float(metadata.average_fps)
        if frame_count != record.num_frames:
            raise RuntimeError(
                f"Manifest/decode frame mismatch for {record.video_id}: "
                f"{record.num_frames} vs {frame_count}"
            )
        indices, valid_mask = self.sampler.sample_with_validity(
            num_frames=frame_count,
            source_fps=source_fps,
            sample_index=index,
        )
        frames = decoder.get_frames_at(indices=indices.tolist()).data
        if tuple(frames.shape[:2]) != (self.sampler.config.clip_length, 3):
            raise RuntimeError(f"Unexpected decoded tensor shape: {tuple(frames.shape)}")

        item: dict[str, Any] = {
            "video": frames,
            "frame_indices": torch.from_numpy(indices.copy()),
            "temporal_valid_mask": torch.from_numpy(valid_mask.copy()),
            "video_id": record.video_id,
            "category": record.category,
            "label": record.label,
            "video_path": record.video_path,
            "source_sequence": record.source_sequence,
            "source_fps": source_fps,
            "num_source_frames": frame_count,
        }
        if self.transform is not None:
            item["pixel_values_videos"] = self.transform(frames)
        return item


def collate_video_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate metadata safely while allowing source videos of different sizes."""

    if not items:
        raise ValueError("Cannot collate an empty batch")
    result: dict[str, Any] = {
        "video": [item["video"] for item in items],
        "frame_indices": torch.stack([item["frame_indices"] for item in items]),
        "temporal_valid_mask": torch.stack([item["temporal_valid_mask"] for item in items]),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }
    for key in (
        "video_id",
        "category",
        "video_path",
        "source_sequence",
        "source_fps",
        "num_source_frames",
    ):
        result[key] = [item[key] for item in items]
    if all("pixel_values_videos" in item for item in items):
        result["pixel_values_videos"] = torch.stack(
            [item["pixel_values_videos"] for item in items]
        )
    return result
