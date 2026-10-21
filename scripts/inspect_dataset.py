#!/usr/bin/env python3
"""Print manifest and optional TorchCodec sample information."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model.data.video_dataset import VideoManifestDataset, read_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset" / "manifest.jsonl")
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args(argv)
    records = read_manifest(args.manifest)
    counts = Counter(record.category for record in records)
    frames = [record.num_frames for record in records]
    payload: dict[str, object] = {
        "videos": len(records),
        "categories": dict(sorted(counts.items())),
        "total_frames": sum(frames),
        "frame_count_min": min(frames),
        "frame_count_median": statistics.median(frames),
        "frame_count_max": max(frames),
        "resolutions": sorted({f"{record.width}x{record.height}" for record in records}),
        "fps_values": sorted({record.fps for record in records}),
    }
    if args.decode:
        sample = VideoManifestDataset(args.manifest)[0]
        payload["decoded_sample"] = {
            "video_id": sample["video_id"],
            "shape": list(sample["video"].shape),
            "frame_indices": sample["frame_indices"].tolist(),
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
