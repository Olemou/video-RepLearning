"""The cluster runner must launch the authoritative stages safely."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_trajectory_pipeline as pipeline


def _inputs(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n")
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    encoder = tmp_path / "hf"
    encoder.mkdir()
    for name in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
        (encoder / name).write_bytes(b"checkpoint")
    return ["--manifest", str(manifest), "--dataset-root", str(tmp_path),
            "--encoder-dir", str(encoder), "--visualize-video", str(video),
            "--output-root", str(tmp_path / "run")]


def test_dry_run_lists_two_rank_stages_without_writing(tmp_path, capsys):
    arguments = _inputs(tmp_path)
    pipeline.main(arguments + ["--dry-run"])
    output = capsys.readouterr().out
    assert "[prepare]" in output and "[extract]" in output
    assert "[train]" in output and "[visualize]" in output
    assert output.count("--nproc_per_node=2") == 2
    assert "--target semantic_similarity --prototype-class 1" in output
    assert not (tmp_path / "run").exists()


def test_dry_run_passes_per_class_limit_to_preparation(tmp_path, capsys):
    pipeline.main(_inputs(tmp_path) + ["--videos-per-class", "10",
                                         "--learning-rate", "0.0005", "--dry-run"])
    output = capsys.readouterr().out
    assert "--per-class 10" in output
    assert "--learning-rate 0.0005" in output


def test_missing_two_gpus_stops_before_creating_run(tmp_path, monkeypatch):
    arguments = _inputs(tmp_path)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="Requested 2 GPUs but PyTorch sees 0"):
        pipeline.main(arguments)
    assert not (tmp_path / "run").exists()


def test_prepare_stage_runs_and_resume_checks_settings(tmp_path, monkeypatch):
    arguments = _inputs(tmp_path)
    calls = []

    def fake_run(command, *, cwd, env, check):
        calls.append(command)
        output = tmp_path / "run/balanced/manifest.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("balanced")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.main(arguments + ["--stop-after", "prepare"])
    assert len(calls) == 1
    assert (tmp_path / "run/pipeline_settings.json").is_file()
    pipeline.main(arguments + ["--resume", "--stop-after", "prepare"])
    assert len(calls) == 1  # completed preparation is reused
    with pytest.raises(ValueError, match="settings differ"):
        pipeline.main(arguments + ["--resume", "--stop-after", "prepare",
                                   "--prototype-class", "0"])
