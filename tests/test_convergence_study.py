from pathlib import Path
from types import SimpleNamespace
import json

import torch
import pytest


def test_refinement_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from convergence_study import run

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        cache = dict(features=torch.randn(6, 3, 8), valid=torch.ones(6, 3, dtype=torch.bool),
                     labels=torch.tensor([0, 0, 0, 1, 1, 1]), ids=list("abcdef"),
                     mean_head={"3.bias": torch.zeros(4)}, seed=239, variance_floor=1e-4)
        settings = dict(device="cpu", curriculum_epochs=3, lr=5.25e-4, final_lr=1e-6,
                        cache_sha256="synthetic-test", resume_dir=None, refinement_max_iter=10,
                        refinement_precision="float64")
        for directory in ("short", "resumed", "full"):
            (tmp_path / directory).mkdir()
        short = SimpleNamespace(**settings, epochs=10, calibration_epochs=4, output=tmp_path / "short")
        run(cache, short, "variance_refinement", 17)
        settings["resume_dir"] = tmp_path / "short"
        resumed = SimpleNamespace(**settings, epochs=12, calibration_epochs=6, output=tmp_path / "resumed")
        result = run(cache, resumed, "variance_refinement", 17)
        settings["resume_dir"] = None
        full = SimpleNamespace(**settings, epochs=12, calibration_epochs=6, output=tmp_path / "full")
        run(cache, full, "variance_refinement", 17)
        a = torch.load(tmp_path / "resumed/variance_refinement_seed17.pt", weights_only=True)
        b = torch.load(tmp_path / "full/variance_refinement_seed17.pt", weights_only=True)
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
        assert result["mean_unchanged_during_calibration"]
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("threshold,expected_joint", [(100., 6), (1e-12, 10)])
def test_adaptive_refinement_waits_for_curriculum_and_stability(tmp_path, monkeypatch, threshold, expected_joint):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from convergence_study import run
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        cache = dict(features=torch.randn(6, 3, 8), valid=torch.ones(6, 3, dtype=torch.bool),
                     labels=torch.tensor([0, 0, 0, 1, 1, 1]), ids=list("abcdef"),
                     mean_head={"3.bias": torch.zeros(4)}, seed=239, variance_floor=1e-4)
        args = SimpleNamespace(device="cpu", curriculum_epochs=3, lr=5.25e-4, final_lr=1e-6,
                               cache_sha256="synthetic-test", resume_dir=None, refinement_max_iter=2,
                               refinement_precision="float64",
                               refinement_threshold=threshold, refinement_patience=3,
                               epochs=10, calibration_epochs=4, output=tmp_path)
        result = run(cache, args, "adaptive_refinement", 17)
        history = json.loads((tmp_path / "adaptive_refinement_seed17.json").read_text())["history"]
        assert result["joint_epochs"] == expected_joint
        assert all(r["phase"] == "joint" for r in history[1:expected_joint + 1])
        assert all(r["phase"] == "variance_calibration" for r in history[expected_joint + 1:])
        assert all(r["mean_gradient_norm"] == 0 for r in history[expected_joint + 1:])
        if expected_joint < 10:
            assert result["mean_unchanged_during_calibration"]
            assert history[-1]["uncertainty_regret"] < history[expected_joint]["uncertainty_regret"]
    finally:
        torch.set_num_threads(old_threads)
