"""Direct MP4-to-temporal-visualization entry point."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import numpy as np

from vision_model.training.trajectory import TrajectoryModel
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.interpretability.spatial_final_block import SpatialFinalBlockResult
from vision_model.interpretability.video_visualization import save_spatial_final_block_video
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import visualize_trajectory_video as entry
import render_temporal_attention_video as render_entry


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("checkpoint_format", [None, "trajectory_resumable_v1", "trajectory_lr_search_v1"])
def test_direct_video_entry_point_decodes_prefixes_and_writes_figures(tmp_path, monkeypatch, checkpoint_format):
    legacy = checkpoint_format is None
    video = tmp_path / "input.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=32x24:rate=4", "-frames:v", "8", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    encoder_dir = tmp_path / "hf"
    encoder_dir.mkdir()
    for name in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
        (encoder_dir / name).write_bytes(b"synthetic test weights")
    model = TrajectoryModel(16, 16, 8)
    with torch.no_grad():
        torch.nn.init.normal_(model.up.weight, std=.04)
    checkpoint = tmp_path / "checkpoint.pt"
    normalization = {"mean": (.485, .456, .406), "std": (.229, .224, .225)}
    payload = dict(model=model.state_dict(), width=16, hidden=16, embedding_dim=8,
                   cache_metadata=dict(temporal_context_policy="past_only_prefix_last_tubelet_v1",
                                       model_id="facebook/vjepa2-vitl-fpc64-256",
                                       checkpoint_sha256=(None if legacy else hashlib.sha256(
                                           (encoder_dir / "model.safetensors").read_bytes()).hexdigest()),
                                       sampling=dict(clip_length=8, target_fps=4., training=False,
                                                     short_video_policy="repeat_last"),
                                       normalization=normalization))
    if checkpoint_format is not None:
        payload["format"] = checkpoint_format
    if checkpoint_format == "trajectory_lr_search_v1":
        payload["train_source_ids"] = ["other"]
        payload["cache_metadata"].update(ids=["input", "other"],
            video_sha256=[hashlib.sha256(video.read_bytes()).hexdigest(), "other_hash"])
    torch.save(payload, checkpoint)
    calls = []

    class FakeEncoder(torch.nn.Module):
        architecture = SimpleNamespace(patch_size=16)

        def forward(self, clip):
            calls.append(clip.shape[2])
            value = clip[:, :, -2:].mean((1, 2, 3, 4))[:, None, None]
            return value.expand(1, 2, 16)

    class FakeLoaded:
        encoder = FakeEncoder()

        def freeze_encoder(self, value):
            assert value

    class FakeTransform:
        def __init__(self, path, *, preserve_field_of_view):
            assert path == str(encoder_dir)
            assert preserve_field_of_view

        @property
        def normalization(self):
            return normalization

        def __call__(self, frames):
            return frames.float() / 255

    monkeypatch.setattr(entry, "load_pretrained_encoder", lambda *_, **__: FakeLoaded())
    monkeypatch.setattr(entry, "OfficialVJEPA2Transform", FakeTransform)
    output = tmp_path / "visuals"
    entry.main(["--video", str(video), "--checkpoint", str(checkpoint),
                "--encoder-dir", str(encoder_dir), "--output", str(output),
                "--target", "state", "--timestep", "2", "--component", "0",
                "--temporal-only"])
    assert calls == [2, 4, 6, 8]
    metadata = json.loads((output / "attribution.json").read_text())
    assert metadata["frame_indices"] == [1, 3, 5, 7]
    assert metadata["timestamps"] == [.25, .75, 1.25, 1.75]
    assert metadata["target"]["timestep"] == 2
    assert metadata["relevance"][3] == 0
    assert metadata["encoder_weight_verification"] == (
        "model_id_only_hash_unrecorded" if legacy else "sha256_match")
    if checkpoint_format == "trajectory_lr_search_v1":
        assert metadata["source_matches_training_cache"] is False
    assert set(metadata["figures"]) == {"temporal_heatmap.png", "timeline.png",
                                       "causal_query_key_maps.png", "storyboard.png",
                                       "temporal_attention_video.mp4"}
    assert (output / "attention_maps.pt").is_file()
    assert all((output / name).stat().st_size > 0 for name in metadata["figures"])
    assert entry.probe_video(output / "temporal_attention_video.mp4") == (8, 4., 640, 520)
    existing = tmp_path / "existing_attribution"
    existing.mkdir()
    prior = dict(metadata)
    prior["figures"] = [name for name in metadata["figures"]
                        if name != "temporal_attention_video.mp4"]
    (existing / "attribution.json").write_text(json.dumps(prior))
    render_entry.main(["--attribution-dir", str(existing)])
    assert entry.probe_video(existing / "temporal_attention_video.mp4") == (8, 4., 640, 520)
    assert "temporal_attention_video.mp4" in json.loads(
        (existing / "attribution.json").read_text())["figures"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_direct_video_entry_point_writes_target_conditioned_spatial_overlay(tmp_path, monkeypatch):
    video = tmp_path / "small.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=16x16:rate=4", "-frames:v", "8", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    encoder_dir = tmp_path / "hf"
    encoder_dir.mkdir()
    for name in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
        (encoder_dir / name).write_bytes(b"synthetic test weights")
    encoder = VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=8, crop_size=16,
        patch_size=8, tubelet_size=2, hidden_size=96, num_hidden_layers=2,
        num_attention_heads=3, mlp_ratio=2.)).eval()
    model = TrajectoryModel(96, 16, 8).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.up.weight, std=.04)
    checkpoint = tmp_path / "checkpoint.pt"
    normalization = {"mean": (.485, .456, .406), "std": (.229, .224, .225)}
    torch.save(dict(format="trajectory_resumable_v1", model=model.state_dict(), width=96,
                    hidden=16, embedding_dim=8,
                    cache_metadata=dict(temporal_context_policy="past_only_prefix_last_tubelet_v1",
                                        model_id="facebook/vjepa2-vitl-fpc64-256",
                                        checkpoint_sha256=hashlib.sha256(
                                            (encoder_dir / "model.safetensors").read_bytes()).hexdigest(),
                                        sampling=dict(clip_length=8, target_fps=4., training=False,
                                                      short_video_policy="repeat_last"),
                                        normalization=normalization)), checkpoint)

    class FakeTransform:
        processor = SimpleNamespace(crop_size={"height": 16, "width": 16})

        def __init__(self, path, *, preserve_field_of_view):
            assert preserve_field_of_view

        @property
        def normalization(self):
            return normalization

        def __call__(self, frames):
            return frames.float() / 255

    class Loaded:
        def __init__(self):
            self.encoder = encoder

        def freeze_encoder(self, frozen):
            self.encoder.requires_grad_(not frozen)

    monkeypatch.setattr(entry, "OfficialVJEPA2Transform", FakeTransform)
    monkeypatch.setattr(entry, "load_pretrained_encoder", lambda *_, **__: Loaded())
    output = tmp_path / "spatial"
    entry.main(["--video", str(video), "--model", str(checkpoint),
                "--encoder-dir", str(encoder_dir), "--output", str(output),
                "--target", "state", "--timestep", "2", "--component", "0"])
    metadata = json.loads((output / "attribution.json").read_text())
    assert metadata["spatial_diagnostic"]["replay_score_error"] < 1e-5
    assert metadata["spatial_diagnostic"]["spatial_grid"] == [2, 2]
    assert "spatial_final_block_overlay.mp4" in metadata["figures"]
    assert entry.probe_video(output / "spatial_final_block_overlay.mp4") == (8, 4., 1280, 500)
    maps = torch.load(output / "spatial_final_block_maps.pt", weights_only=True)
    assert maps["positive_maps"].shape == (4, 2, 2)
    assert maps["positive_maps"][3].count_nonzero() == 0  # target at step 2
    assert metadata['spatial_rendering']['zero']=='exactly transparent'
    assert metadata['spatial_rendering']['scope']=='conditional_attention_key_gradient'


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_spatial_overlay_preserves_zero_pixels_and_uses_one_magnitude_scale(tmp_path):
    maps = torch.tensor([[[0., 1.], [0., 1.]],
                         [[0., .5], [0., .5]]])
    result = SpatialFinalBlockResult(maps, maps, maps, torch.ones(2), 0., 0., 1.)
    frames = {index: np.full((16, 16, 3), 128, dtype=np.uint8) for index in range(4)}
    video = save_spatial_final_block_video(result, tmp_path,
        frames_by_index=frames, sampled_frame_indices=list(frames),
        source_fps=4., playback_fps=4., checkpoint_image_size=16)
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                              "-frames:v", "4", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "pipe:1"], capture_output=True, check=True).stdout
    image = np.frombuffer(decoded, dtype=np.uint8).reshape(4,500,1280,3)
    low, high = image[0,200,790], image[0,200,1130]
    assert np.max(np.abs(low.astype(int)-128))<=5  # codec tolerance; no blue wash
    # The native-cell boundary is x=960 in this video. A second bilinear
    # resize after compositing would spread red into the zero cell at x=952.
    assert np.max(np.abs(image[0,200,952].astype(int)-128))<=5
    assert int(high[0]) > int(high[2]) + 70
    # Half the attribution in the later tubelet must have lower opacity; no
    # per-frame max normalization may turn it into another full-red cell.
    later=image[2,200,1130]
    assert int(high[0])-int(later[0])>15
