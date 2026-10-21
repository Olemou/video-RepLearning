#!/usr/bin/env python3
"""Report mask and two-frame tubelet coverage for positive videos."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    videos = []
    for record in records:
        if record["label"] != 1:
            continue
        folder = args.annotations / record["video_id"]
        metadata_path = folder / "annotation.json"
        expected = json.loads(metadata_path.read_text())["frames"] if metadata_path.is_file() else []
        expected_slots = {int(row["slot"]) for row in expected}
        saved_files = sorted((folder / "masks").glob("slot_*_source_*.png"))
        saved_slots = {int(re.search(r"slot_(\d+)_", path.name).group(1)) for path in saved_files}
        complete_pairs = sum(slot in saved_slots and slot + 1 in saved_slots
                             for slot in range(0, 64, 2) if slot in expected_slots)
        expected_pairs = sum(slot in expected_slots and slot + 1 in expected_slots
                             for slot in range(0, 64, 2))
        videos.append({"video_id": record["video_id"], "saved_masks": len(saved_slots),
                       "expected_masks": len(expected_slots), "complete_pairs": complete_pairs,
                       "expected_pairs": expected_pairs, "complete": saved_slots == expected_slots})
    result = {
        "videos": videos,
        "saved_masks": sum(row["saved_masks"] for row in videos),
        "expected_masks": sum(row["expected_masks"] for row in videos),
        "minimum_training_ready": all(row["complete_pairs"] >= 1 for row in videos),
        "full_experiment_ready": all(row["complete"] for row in videos),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered)
    if args.require_complete and not result["full_experiment_ready"]:
        raise SystemExit(2)


if __name__ == "__main__": main()
