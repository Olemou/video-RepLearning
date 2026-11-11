"""Balanced manifests must count independent source videos, not duplicate files."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_balanced_videos import balance


def make_row(category, name):
    return dict(video_id=f"{category}_{name}", category=category,
                label=0 if category == "clean" else 1,
                video_path=f"{category}/{name}.mp4", source_sequence=name,
                num_frames=8, fps=4.0, width=16, height=16, duration_seconds=2.0)


def test_balancing_removes_byte_copies_and_records_selection(tmp_path):
    rows = [make_row("clean", name) for name in ("a", "b", "c", "copy")]
    rows += [make_row("cold", name) for name in ("d", "e")]
    for row in rows:
        path = tmp_path / row["video_path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"a" if row["video_id"] in {"clean_a", "clean_copy"}
                         else row["video_id"].encode())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected, metadata = balance(manifest, tmp_path, seed=239)
    assert [sum(row["label"] == label for row in selected) for label in (0, 1)] == [2, 2]
    assert metadata["unique_counts"] == {"0": 3, "1": 2}
    assert metadata["removed_byte_duplicate_ids"] == ["clean_copy"]
    assert len(set(metadata["selected_video_sha256"].values())) == 4
    assert selected == balance(manifest, tmp_path, seed=239)[0]
    requested, requested_metadata = balance(manifest, tmp_path, seed=239, per_class=2)
    assert [sum(row["label"] == label for row in requested) for label in (0, 1)] == [2, 2]
    assert requested_metadata["per_class_requested"] == 2
    with pytest.raises(ValueError, match="--per-class"):
        balance(manifest, tmp_path, seed=239, per_class=4)


def test_balancing_rejects_same_bytes_with_opposite_labels(tmp_path):
    rows = [make_row("clean", "a"), make_row("cold", "b")]
    for row in rows:
        path = tmp_path / row["video_path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"same")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="conflicting class labels"):
        balance(manifest, tmp_path, seed=239)
