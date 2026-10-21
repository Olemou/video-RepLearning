#!/usr/bin/env python3
"""Prepare seven unique clean and seven unique spill videos for honest training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_model.data.video_dataset import VideoManifestDataset


CLEAN_IDS = (
    "clean_frames_20260227_181941", "clean_videoFrame87", "clean_videoFrame99",
    "clean_videoFrame101", "clean_videoFrame131", "clean_videoFrame134", "clean_videoFrame136",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/experiments/unique_7_clean_7_spill")
    args = parser.parse_args()
    source = [json.loads(line) for line in args.base_manifest.read_text().splitlines() if line.strip()]
    by_id = {row["video_id"]: row for row in source}
    rows = [by_id[video_id] for video_id in CLEAN_IDS] + [row for row in source if row["label"] == 1]
    if len(rows) != 14 or sum(row["label"] == 0 for row in rows) != 7 or sum(row["label"] == 1 for row in rows) != 7:
        raise ValueError("Expected seven clean and seven spill videos")
    if len({row["video_id"] for row in rows}) != 14:
        raise ValueError("Duplicate video IDs")
    hashes = {row["video_id"]: hashlib.sha256((args.dataset_root / row["video_path"]).read_bytes()).hexdigest()
              for row in rows}
    if len(set(hashes.values())) != 14:
        raise ValueError("This selection contains byte-identical videos")
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "video_training", "segmentation/annotations", "segmentation/output"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    VideoManifestDataset(manifest, dataset_root=args.dataset_root)
    provenance = {
        "name": "unique_7_clean_7_spill", "class_counts": {"clean_0": 7, "spill_1": 7},
        "label_map": {"clean": 0, "cold": 1, "warm": 1},
        "selection_policy": "Seven byte-unique clean videos and all seven byte-unique spill videos in the base manifest.",
        "excluded_exact_duplicates_from_legacy_8_8": [
            ["clean_videoFrame87", "clean_videoFrame90"],
            ["warm_scene_1_hot", "cold_scene_4"],
        ],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_root": str(args.dataset_root.resolve()), "video_sha256": hashes,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"experiment": str(args.output), "manifest": str(manifest),
                      "class_counts": provenance["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
