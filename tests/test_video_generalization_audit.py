import json
from pathlib import Path


def test_deduplication_removes_exact_file_copies(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_video_generalization import deduplicated_indices

    (tmp_path / "first.mp4").write_bytes(b"same video")
    (tmp_path / "copy.mp4").write_bytes(b"same video")
    (tmp_path / "other.mp4").write_bytes(b"different video")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"video_id": name, "video_path": f"{name}.mp4"}) + "\n"
                                for name in ("first", "copy", "other")))

    keep, duplicates = deduplicated_indices(manifest, tmp_path, ["first", "copy", "other"])
    assert keep == [0, 2]
    assert duplicates == [["first", "copy"]]
