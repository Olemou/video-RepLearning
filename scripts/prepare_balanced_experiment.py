#!/usr/bin/env python3
"""Prepare the reproducible eight-clean/eight-spill experiment layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_model.data.video_dataset import VideoManifestDataset


CLEAN_IDS = (
    "clean_frames_20260227_181941", "clean_videoFrame87", "clean_videoFrame90",
    "clean_videoFrame99", "clean_videoFrame101", "clean_videoFrame131",
    "clean_videoFrame134", "clean_videoFrame136",
)


def probe(path: Path) -> dict:
    process = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(process.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    return {
        "num_frames": int(stream["nb_read_frames"]), "fps": numerator / denominator,
        "width": int(stream["width"]), "height": int(stream["height"]),
        "duration_seconds": float(stream["duration"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/experiments/balanced_8_clean_8_spill")
    args = parser.parse_args()
    source = [json.loads(line) for line in args.base_manifest.read_text().splitlines() if line.strip()]
    by_id = {row["video_id"]: row for row in source}
    clean = [by_id[video_id] for video_id in CLEAN_IDS]
    positive = [row for row in source if row["label"] == 1]
    extra_path = args.dataset_root / "cold/scene_4.mp4"
    if not extra_path.is_file():
        raise FileNotFoundError(f"Required eighth positive video is missing: {extra_path}")
    metadata = probe(extra_path)
    if metadata["num_frames"] <= 0 or metadata["duration_seconds"] <= 0:
        raise RuntimeError(f"Invalid eighth positive video: {extra_path}")
    positive.append({
        "category": "cold", **metadata, "label": 1,
        "source_sequence": "cold/scene_4", "video_id": "cold_scene_4",
        "video_path": "cold/scene_4.mp4",
    })
    rows = clean + positive
    if len(rows) != 16 or sum(row["label"] == 0 for row in rows) != 8 or sum(row["label"] == 1 for row in rows) != 8:
        raise RuntimeError("Expected exactly eight clean and eight spill videos")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate video IDs in balanced selection")
    hashes = {row["video_id"]: hashlib.sha256((args.dataset_root / row["video_path"]).read_bytes()).hexdigest()
              for row in rows}
    by_hash: dict[str, list[str]] = {}
    for video_id, digest in hashes.items():
        by_hash.setdefault(digest, []).append(video_id)
    duplicates = [ids for ids in by_hash.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(
            "Selected videos contain byte-identical duplicates: "
            + "; ".join(", ".join(ids) for ids in duplicates)
            + ". The 8/8 manifest cannot be presented as 16 independent videos."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "video_training", "segmentation/annotations", "segmentation/output"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    # Construction validates labels, paths, and manifest schema against the real dataset root.
    VideoManifestDataset(manifest, dataset_root=args.dataset_root)
    provenance = {
        "name": "balanced_8_clean_8_spill", "class_counts": {"clean_0": 8, "spill_1": 8},
        "label_map": {"clean": 0, "cold": 1, "warm": 1},
        "selection_policy": "Eight deterministic clean videos spanning the available list; all seven manifest positives plus distinct validated cold scene_4.",
        "manifest": str(manifest), "dataset_root": str(args.dataset_root.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "video_sha256": hashes,
        "scene_4_note": "scene_4.mp4 fully decodes as 50 H.264 frames at 10 FPS but was absent from the base manifest and source-image tree.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"experiment": str(args.output), "manifest": str(manifest),
                      "class_counts": provenance["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
