#!/usr/bin/env python3
"""Build a byte-unique, class-balanced manifest from a video dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_model.data.labels import LABEL_MAP
from vision_model.data.video_dataset import VideoManifestDataset, read_manifest


def balance(manifest: Path, dataset_root: Path, *, seed: int,
            per_class: int | None = None) -> tuple[list[dict], dict]:
    records = read_manifest(manifest)
    VideoManifestDataset(manifest, dataset_root=dataset_root)
    by_digest: dict[str, list] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256((dataset_root / record.video_path).read_bytes()).hexdigest()
        by_digest[digest].append(record)
    conflicts = [members for members in by_digest.values()
                 if len({record.label for record in members}) != 1]
    if conflicts:
        raise ValueError("Byte-identical videos have conflicting class labels")
    unique = [min(members, key=lambda record: record.video_id) for members in by_digest.values()]
    classes = {label: sorted((record for record in unique if record.label == label),
                             key=lambda record: record.video_id) for label in (0, 1)}
    if min(map(len, classes.values())) < 2:
        raise ValueError("Both classes need at least two distinct source videos")
    available = min(map(len, classes.values()))
    if per_class is not None and (per_class < 2 or per_class > available):
        raise ValueError(f"--per-class must be between 2 and {available} byte-unique videos")
    count = available if per_class is None else per_class
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        selected.extend(sorted(rng.sample(classes[label], count), key=lambda record: record.video_id))
    rows = [record.__dict__ for record in selected]
    metadata = {
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_root": str(dataset_root.resolve()), "seed": seed,
        "per_class_requested": per_class,
        "label_map": LABEL_MAP,
        "input_counts": dict(Counter(str(record.label) for record in records)),
        "unique_counts": {str(label): len(classes[label]) for label in (0, 1)},
        "selected_counts": {"0": count, "1": count},
        "removed_byte_duplicate_ids": sorted(record.video_id for members in by_digest.values()
                                             for record in members
                                             if record != min(members, key=lambda item: item.video_id)),
        "selected_video_sha256": {record.video_id: hashlib.sha256(
            (dataset_root / record.video_path).read_bytes()).hexdigest() for record in selected},
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--per-class", type=int,
                        help="Select exactly this many byte-unique videos from each class")
    args = parser.parse_args()
    rows, metadata = balance(args.manifest, args.dataset_root, seed=args.seed,
                             per_class=args.per_class)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_manifest = args.output_dir / "manifest.jsonl"
    output_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    VideoManifestDataset(output_manifest, dataset_root=args.dataset_root)
    metadata["balanced_manifest_sha256"] = hashlib.sha256(output_manifest.read_bytes()).hexdigest()
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"manifest": str(output_manifest), "counts": metadata["selected_counts"]}))


if __name__ == "__main__":
    main()
