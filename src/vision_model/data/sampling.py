"""Temporal index sampling for V-JEPA 2 clips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ShortVideoPolicy = Literal["repeat_last", "loop", "error"]


@dataclass(frozen=True)
class TemporalSamplingConfig:
    clip_length: int = 64
    target_fps: float = 4.0
    training: bool = True
    short_video_policy: ShortVideoPolicy = "repeat_last"

    def __post_init__(self) -> None:
        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")


class TemporalSampler:
    """Sample a fixed clip at a target temporal rate.

    Training uses a random start when the video contains the full temporal
    window. Validation uses a centered start. Short videos follow an explicit
    policy and never cause implicit sample replacement.
    """

    def __init__(self, config: TemporalSamplingConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample(
        self,
        *,
        num_frames: int,
        source_fps: float,
        sample_index: int = 0,
    ) -> np.ndarray:
        indices, _ = self.sample_with_validity(
            num_frames=num_frames, source_fps=source_fps, sample_index=sample_index
        )
        return indices

    def sample_with_validity(
        self,
        *,
        num_frames: int,
        source_fps: float,
        sample_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sampled indices and a mask excluding padding/repeated tail positions."""
        if num_frames <= 0:
            raise ValueError("Cannot sample a video with no frames")
        if source_fps <= 0:
            raise ValueError(f"Invalid source FPS: {source_fps}")

        step = source_fps / self.config.target_fps
        offsets = np.arange(self.config.clip_length, dtype=np.float64) * step
        required_last = float(offsets[-1])
        max_start = (num_frames - 1) - required_last

        if max_start >= 0:
            if self.config.training:
                # Per-example/epoch seeding is reproducible across worker counts.
                seed = np.random.SeedSequence([self.seed, self.epoch, int(sample_index)])
                rng = np.random.default_rng(seed)
                start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
            else:
                start = max_start / 2.0
            indices = np.rint(start + offsets).astype(np.int64)
            return np.clip(indices, 0, num_frames - 1), np.ones(self.config.clip_length, dtype=np.bool_)

        policy = self.config.short_video_policy
        base = np.rint(offsets).astype(np.int64)
        if policy == "repeat_last":
            valid = base < num_frames
            return np.clip(base, 0, num_frames - 1), valid
        if policy == "loop":
            return np.mod(base, num_frames), np.ones(self.config.clip_length, dtype=np.bool_)
        if policy == "error":
            span_seconds = required_last / source_fps
            raise ValueError(
                f"Video has {num_frames} frames at {source_fps:g} FPS, but "
                f"{self.config.clip_length} frames at {self.config.target_fps:g} FPS "
                f"require a {span_seconds:g}-second span"
            )
        raise ValueError(f"Unknown short-video policy: {policy}")
