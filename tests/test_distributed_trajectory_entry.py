"""Exercise the real training entry point with uneven source shards."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import ModuleType

import torch
import torch.multiprocessing as mp

from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def _worker(rank, port, cache_path, output_path, epochs, resume, curriculum_epochs):
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    # Plotting is outside the distributed computation under test.
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = lambda *_: None
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = ModuleType("matplotlib.pyplot")
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_trajectory

    train_trajectory.plot_history = lambda *_: None
    torch.set_num_threads(1)
    train_trajectory.torch.set_num_threads = lambda *_: None
    sys.argv = ["train_trajectory.py", "--cache", cache_path, "--output", output_path,
                "--epochs", str(epochs), "--curriculum-epochs", str(curriculum_epochs)]
    if resume:
        sys.argv.append("--resume")
    train_trajectory.main()


def _make_cache(tmp_path):
    rows = [dict(video_id=f"{category}_{index}", category=category,
                 label=LABEL_MAP[category], video_path=f"{category}/{index}.mp4",
                 source_sequence=str(index), num_frames=8, fps=4.0,
                 width=16, height=16, duration_seconds=2.0)
            for category in ("clean", "cold") for index in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    torch.manual_seed(7)
    cache = dict(features=torch.randn(4, 2, 6, 16),
                 augmented_features=torch.randn(4, 3, 6, 16),
                 valid=torch.tensor([[1, 1, 1, 1, 0, 0]] * 4, dtype=torch.bool),
                 timestamps=torch.arange(6).float()[None].expand(4, -1) * .5,
                 labels=torch.tensor([0, 0, 1, 1]),
                 ids=[row["video_id"] for row in rows],
                 categories=[row["category"] for row in rows],
                 video_sha256=[f"unique-{index}" for index in range(4)],
                 label_schema=LABEL_SCHEMA, label_map=LABEL_MAP,
                 temporal_context_policy="past_only_prefix_last_tubelet_v1",
                 manifest=str(manifest),
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 normalization={"mean": (.485, .456, .406), "std": (.229, .224, .225)})
    cache_path = tmp_path / "features.pt"
    torch.save(cache, cache_path)
    return cache_path, cache


def _launch(cache_path, output, *, epochs, resume=False, curriculum_epochs=20):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(cache_path), str(output), epochs, resume,
                            curriculum_epochs), nprocs=3)


def test_entry_point_writes_global_report_from_rank_zero(tmp_path):
    cache_path, cache = _make_cache(tmp_path)
    output = tmp_path / "training"
    _launch(cache_path, output, epochs=1)
    report = json.loads((output / "report.json").read_text())
    assert report["distributed_world_size"] == 3
    assert report["rank_source_counts_per_epoch"] == [[2, 1, 1]]
    assert report["sampler_padding_filtered_per_epoch"] == [2]
    assert report["evaluation_rank_source_counts"] == [2, 1, 1]
    assert report["evaluation_sampler_padding_filtered"] == 2
    assert [sum(views) for views in report["training_view_exposures"]] == [2] * 4
    assert report["optimizer_updates"] == 1
    assert (output / "checkpoint.pt").is_file()
    representations = torch.load(output / "representations.pt", map_location="cpu", weights_only=False)
    assert representations["ids"] == cache["ids"]
    assert len(representations["embedding"]) == 8


def test_resume_matches_uninterrupted_distributed_training(tmp_path):
    cache_path, _ = _make_cache(tmp_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    _launch(cache_path, uninterrupted, epochs=4, curriculum_epochs=1)
    _launch(cache_path, resumed, epochs=2, curriculum_epochs=1)
    first = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert first["epochs_completed"] == 2
    _launch(cache_path, resumed, epochs=4, curriculum_epochs=1, resume=True)
    reference = torch.load(uninterrupted / "checkpoint.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert actual["epochs_completed"] == actual["optimizer_updates"] == 4
    assert [row["epoch"] for row in actual["history"]] == list(range(5))
    for got, expected in zip(actual["history"], reference["history"]):
        assert abs(got["total_loss"] - expected["total_loss"]) < 1e-5
    assert actual["rank_source_counts_per_epoch"] == reference["rank_source_counts_per_epoch"]
    torch.testing.assert_close(actual["training_view_exposures"],
                               reference["training_view_exposures"], rtol=0, atol=0)
    for key in reference["model"]:
        torch.testing.assert_close(actual["model"][key], reference["model"][key], rtol=0, atol=3e-3)
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            observed = actual["optimizer"]["state"][key][name]
            if torch.is_tensor(value):
                torch.testing.assert_close(observed, value, rtol=0, atol=3e-3)
            else:
                assert observed == value
    assert all(int(state["step"]) == 3 for state in actual["optimizer"]["state"].values())
    report = json.loads((resumed / "report.json").read_text())
    assert report["resumed_from_epoch"] == 2
    assert report["resume_count"] == 1
    assert report["optimizer_reset_before_update"] == 2
    assert [sum(views) for views in report["training_view_exposures"]] == [8] * 4


def test_resume_can_finish_reports_after_last_update(tmp_path):
    cache_path, _ = _make_cache(tmp_path)
    output = tmp_path / "finish"
    _launch(cache_path, output, epochs=1)
    (output / "report.json").unlink()
    (output / "representations.pt").unlink()
    _launch(cache_path, output, epochs=1, resume=True)
    report = json.loads((output / "report.json").read_text())
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert report["epochs_completed"] == checkpoint["epochs_completed"] == 1
    assert report["resume_count"] == checkpoint["resume_count"] == 1
    assert len(report["history"]) == 2
    assert (output / "representations.pt").is_file()
