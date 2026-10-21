"""Exact selected-frame decoding remains usable without TorchCodec."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.video_dataset import VideoManifestDataset


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_real_mp4_decodes_sampled_indices_and_repeated_tail(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=32x24:rate=4", "-frames:v", "8", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(video_id="clean_one", category="clean", label=0,
        video_path="clip.mp4", source_sequence="one", num_frames=8, fps=4.,
        width=32, height=24, duration_seconds=2.)) + "\n")
    dataset = VideoManifestDataset(manifest, dataset_root=tmp_path,
        sampling=TemporalSamplingConfig(clip_length=12, target_fps=4., training=False,
                                        short_video_policy="repeat_last"))
    item = dataset[0]
    assert item["video"].shape == (12, 3, 24, 32)
    assert item["frame_indices"].tolist() == list(range(8)) + [7] * 4
    assert item["temporal_valid_mask"].sum() == 8
    assert not torch.equal(item["video"][0], item["video"][1])
    torch.testing.assert_close(item["video"][7], item["video"][11])
