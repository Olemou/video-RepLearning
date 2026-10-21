#!/usr/bin/env python3
"""Explain a trained trajectory model from one MP4 with temporal and spatial views.

The frozen V-JEPA 2 encoder is run independently on every observed prefix.
This can take several minutes. Requires ffmpeg/ffprobe and the local encoder
weights used for the trained trajectory checkpoint.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.interpretability.temporal_attribution import (
    AttributionTarget, attribute_trajectory, perturbation_check)
from vision_model.interpretability.video_visualization import (
    save_temporal_figures, save_temporal_overlay_video, save_spatial_final_block_video)
from vision_model.interpretability.spatial_final_block import attribute_spatial_final_block
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> tuple[int, float, int, int]:
    """Read decoded frame count, average FPS, width, and height with FFprobe."""
    process = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
                              "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one primary video stream in {path}")
    stream = streams[0]
    try:
        count = int(stream["nb_read_frames"])
        fps = float(Fraction(stream["avg_frame_rate"]))
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Video needs a known frame count, FPS, width, and height") from exc
    if min(count, width, height) < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video frame count, FPS, or dimensions")
    return count, fps, width, height


def decode_selected_frames(path: Path, indices: list[int], *, frame_count: int,
                           width: int, height: int) -> dict[int, np.ndarray]:
    """Decode only requested source indices into CPU RGB arrays."""
    selected = sorted(set(int(index) for index in indices))
    if not selected or selected[0] < 0 or selected[-1] >= frame_count:
        raise ValueError("Sampled source frame index is outside the video")
    expression = "+".join(f"eq(n\\,{index})" for index in selected)
    process = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-an",
                              "-vf", f"select={expression}", "-vsync", "0",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                             capture_output=True, check=True)
    frame_bytes = width * height * 3
    if len(process.stdout) != len(selected) * frame_bytes:
        raise RuntimeError("FFmpeg returned a different number of selected frames")
    pixels = np.frombuffer(process.stdout, dtype=np.uint8).reshape(len(selected), height, width, 3)
    return {index: pixels[position].copy() for position, index in enumerate(selected)}


def extract_prefix_features(encoder, transform, frames: torch.Tensor, *, device: torch.device,
                            dtype: torch.dtype, capture_final_inputs: bool = False):
    """Match extract_causal_video_features.py: final tubelet spatial mean."""
    pixels = transform(frames)
    if pixels.ndim != 4 or pixels.shape[0] != frames.shape[0] or len(pixels) % 2:
        raise ValueError("Preprocessing must return an even number of video frames")
    patch = encoder.architecture.patch_size
    spatial = (pixels.shape[-2] // patch) * (pixels.shape[-1] // patch)
    features = []
    final_inputs = []
    handle = None
    if capture_final_inputs:
        def save_final_input(_module, inputs):
            final_inputs.append(inputs[0].detach().cpu().clone())
        handle = encoder.blocks[-1].register_forward_pre_hook(save_final_input)
    try:
        for end in range(1, len(pixels) // 2 + 1):
            clip = pixels[:2 * end].permute(1, 0, 2, 3)[None].to(device=device, dtype=dtype)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype,
                                                         enabled=device.type == "cuda"):
                tokens = encoder(clip)
                features.append(tokens[:, -spatial:].mean(1).float().cpu()[0])
    finally:
        if handle is not None:
            handle.remove()
    result = torch.stack(features)
    if capture_final_inputs:
        if len(final_inputs) != len(result):
            raise RuntimeError("Final-block capture omitted an observed prefix")
        return result, final_inputs
    return result


def _reference(args, checkpoint: dict, device: torch.device):
    if args.target != "semantic_similarity":
        return None
    if args.reference is not None:
        value = torch.load(args.reference, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value["embedding"]
        return torch.as_tensor(value, device=device).flatten()
    if args.prototype_class is None:
        raise ValueError("Semantic similarity needs --reference or --prototype-class")
    path = args.checkpoint.parent / "representations.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = saved["embedding"].reshape(len(saved["ids"]), 2, -1)[:, 0]
    selected = embeddings[saved["labels"] == args.prototype_class]
    if selected.numel() == 0:
        raise ValueError("Chosen class has no fitted-source representations")
    return torch.nn.functional.normalize(selected.mean(0), dim=0).to(device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True, help="Input MP4 or other FFmpeg-readable video")
    p.add_argument("--checkpoint", "--model", type=Path, required=True,
                   help="Trained trajectory checkpoint.pt")
    p.add_argument("--encoder-dir", type=Path, default=Path("/home/olemou/hf"),
                   help="Local facebook/vjepa2-vitl-fpc64-256 files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", choices=["state", "dynamics", "semantic_similarity",
                                      "future_prediction", "future_error"], default="state")
    p.add_argument("--timestep", type=int, help="Zero-based tubelet step; defaults to last valid step")
    p.add_argument("--component", type=int, default=0,
                   help="Explicit embedding/dynamics component for state or prediction targets")
    p.add_argument("--future-timestep", type=int)
    p.add_argument("--reference", type=Path, help="Saved reference embedding tensor")
    p.add_argument("--prototype-class", type=int, choices=[0, 1],
                   help="Fitted-source class prototype from representations.pt")
    p.add_argument("--method", choices=["positive_gradient", "attention_gradient"],
                   default="positive_gradient")
    p.add_argument("--normalization", choices=["none", "sum", "max"], default="sum")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--heads", type=int, nargs="+")
    p.add_argument("--max-attention-elements", type=int, default=8_000_000)
    p.add_argument("--storyboard-count", type=int, default=8)
    p.add_argument("--faithfulness", action="store_true")
    p.add_argument("--temporal-only", action="store_true",
                   help="Skip the frozen V-JEPA final-block spatial diagnostic")
    args = p.parse_args(argv)
    if not args.video.is_file():
        p.error(f"Video does not exist: {args.video}")
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in (None, "trajectory_resumable_v1", "trajectory_lr_search_v1") or not all(
            key in checkpoint for key in ("model", "width", "hidden", "embedding_dim", "cache_metadata")):
        raise ValueError("Expected a trained trajectory checkpoint with causal cache metadata")
    metadata = checkpoint.get("cache_metadata", {})
    if metadata.get("temporal_context_policy") != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Checkpoint lacks verified causal-prefix feature metadata")
    if metadata.get("model_id") != "facebook/vjepa2-vitl-fpc64-256":
        raise ValueError("This script requires the trained facebook/vjepa2-vitl-fpc64-256 encoder")
    for name in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
        if not (args.encoder_dir / name).is_file():
            raise FileNotFoundError(args.encoder_dir / name)
    encoder_sha = sha256_file(args.encoder_dir / "model.safetensors")
    expected_encoder_sha = metadata.get("checkpoint_sha256")
    if expected_encoder_sha is not None and expected_encoder_sha != encoder_sha:
        raise ValueError("Local encoder weights do not match the training feature cache")
    encoder_verification = "sha256_match" if expected_encoder_sha is not None else "model_id_only_hash_unrecorded"
    sampling = metadata.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("training") is not False or sampling.get("short_video_policy") != "repeat_last":
        raise ValueError("Training cache sampling policy is missing or incompatible")
    if sampling["clip_length"] % 2:
        raise ValueError("Two-frame tubelets require an even clip length")
    if args.target in ("future_prediction", "future_error") and args.timestep is None:
        p.error("Future targets require --timestep and --future-timestep")
    frame_count, fps, width, height = probe_video(args.video)
    sample_indices, sample_valid = TemporalSampler(TemporalSamplingConfig(**sampling)).sample_with_validity(
        num_frames=frame_count, source_fps=fps)
    length = int(sample_valid.sum()) // 2
    if length < 2:
        raise ValueError("Video has fewer than two valid observed tubelets")
    source_indices = sample_indices[:2 * length].tolist()
    frames_by_index = decode_selected_frames(args.video, source_indices, frame_count=frame_count,
                                              width=width, height=height)
    frames = torch.stack([torch.from_numpy(frames_by_index[index]).permute(2, 0, 1)
                          for index in source_indices])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    transform = OfficialVJEPA2Transform(str(args.encoder_dir), preserve_field_of_view=True)
    if metadata.get("normalization") != transform.normalization:
        raise ValueError("Local processor normalization differs from the training feature cache")
    loaded = load_pretrained_encoder(str(args.encoder_dir), local_files_only=True)
    loaded.freeze_encoder(True)
    encoder = loaded.encoder.to(device=device, dtype=dtype).eval()
    extracted = extract_prefix_features(encoder, transform, frames, device=device, dtype=dtype,
                                        capture_final_inputs=not args.temporal_only)
    features, final_inputs = (extracted, None) if args.temporal_only else extracted
    del loaded, frames
    if features.shape[-1] != checkpoint["width"]:
        raise ValueError("Extracted feature width differs from the trajectory checkpoint")
    model = TrajectoryModel(checkpoint["width"], checkpoint["hidden"],
                            checkpoint["embedding_dim"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    ends = torch.as_tensor(source_indices[1::2], dtype=torch.long)
    times = ends.float() / fps
    valid = torch.ones(length, dtype=torch.bool, device=device)
    step = (length - 1) if args.timestep is None and args.target != "semantic_similarity" else args.timestep
    reference = _reference(args, checkpoint, device)
    target = AttributionTarget(args.target, step, args.component, reference, args.future_timestep)
    result = attribute_trajectory(model, features.to(device), valid, times.to(device), target,
        video_id=args.video.stem, sample_id="identity", frame_indices=ends,
        checkpoint_id=sha256_file(args.checkpoint),
        method=args.method, normalization=args.normalization,
        layers=None if args.layers is None else tuple(args.layers),
        heads=None if args.heads is None else tuple(args.heads),
        include_maps=True, max_attention_elements=args.max_attention_elements)
    faithfulness = (perturbation_check(model, features.to(device), valid, times.to(device),
                                      target, result.relevance) if args.faithfulness else None)
    spatial = None if args.temporal_only else attribute_spatial_final_block(
        encoder, model, final_inputs, features, times, target)
    del encoder, final_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    figures = save_temporal_figures(result, args.output, frames_by_index=frames_by_index,
                                    storyboard_count=args.storyboard_count)
    figures.append(save_temporal_overlay_video(result, args.output, frames_by_index=frames_by_index,
                                               sampled_frame_indices=source_indices, source_fps=fps,
                                               playback_fps=sampling["target_fps"]))
    if spatial is not None:
        figures.append(save_spatial_final_block_video(spatial, args.output,
            frames_by_index=frames_by_index, sampled_frame_indices=source_indices,
            source_fps=fps, playback_fps=sampling["target_fps"],
            checkpoint_image_size=transform.processor.crop_size["height"] if hasattr(
                transform, "processor") else 256))
        torch.save(dict(positive_maps=spatial.positive_maps, signed_maps=spatial.signed_maps,
                        raw_attention_maps=spatial.raw_attention_maps),
                   args.output / "spatial_final_block_maps.pt")
    video_sha = sha256_file(args.video)
    payload = result.metadata()
    cached_hashes = metadata.get("video_sha256")
    if checkpoint.get("train_source_ids") is not None:
        training_ids = set(checkpoint["train_source_ids"])
        cached_hashes = [digest for video_id, digest in zip(
            metadata["ids"], metadata["video_sha256"]) if video_id in training_ids]
    payload.update(source_video=str(args.video.resolve()), video_sha256=video_sha,
                   source_matches_training_cache=(None if cached_hashes is None else
                                                  video_sha in cached_hashes),
                   encoder_sha256=encoder_sha, encoder_weight_verification=encoder_verification,
                   source_fps=fps, sampled_source_frame_indices=source_indices,
                   sampling=sampling, figures=[path.name for path in figures],
                   spatial_diagnostic=None if spatial is None else spatial.metadata(),
                   faithfulness=faithfulness,
                   note="Final-block spatial patch sensitivity and temporal relevance; neither is a validated spill mask")
    (args.output / "attribution.json").write_text(json.dumps(payload, indent=2))
    torch.save(dict(raw_attention=result.raw_attention, signed_gradients=result.signed_gradients,
                    relevance_maps=result.relevance_maps), args.output / "attention_maps.pt")
    print(json.dumps(dict(output=str(args.output), target_value=result.target_value,
                          valid_tubelets=length, figures=payload["figures"])))


if __name__ == "__main__":
    main()
