#!/usr/bin/env python3
"""Add an MP4 temporal heatmap strip to an existing attribution result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vision_model.interpretability.video_visualization import save_temporal_overlay_video
from visualize_trajectory_video import decode_selected_frames, probe_video, sha256_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attribution-dir", type=Path, required=True)
    args = p.parse_args(argv)
    metadata_path = args.attribution_dir / "attribution.json"
    metadata = json.loads(metadata_path.read_text())
    video = Path(metadata["source_video"])
    if sha256_file(video) != metadata["video_sha256"]:
        raise ValueError("Source video differs from the attributed video")
    count, fps, width, height = probe_video(video)
    if abs(fps - metadata["source_fps"]) > 1e-4:
        raise ValueError("Source FPS differs from attribution metadata")
    selected = metadata["sampled_source_frame_indices"]
    result = SimpleNamespace(frame_indices=torch.as_tensor(metadata["frame_indices"]),
                             valid_length=int(metadata["valid_length"]),
                             relevance=torch.as_tensor(metadata["relevance"]),
                             timestamps=torch.as_tensor(metadata["timestamps"]),
                             target=metadata["target"], target_value=metadata["target_value"])
    frames = decode_selected_frames(video, selected, frame_count=count, width=width, height=height)
    movie = save_temporal_overlay_video(result, args.attribution_dir,
        frames_by_index=frames, sampled_frame_indices=selected, source_fps=fps,
        playback_fps=metadata["sampling"]["target_fps"])
    metadata.setdefault("figures", []).append(movie.name)
    metadata["video_overlay_kind"] = "temporal_heatmap_strip_below_unchanged_frames"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(metadata_path)
    print(movie)


if __name__ == "__main__":
    main()
