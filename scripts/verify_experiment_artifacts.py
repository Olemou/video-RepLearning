#!/usr/bin/env python3
"""Reload every current experiment checkpoint and reproduce its final metrics."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from convergence_study import CachedHeads, diagnostics, evaluate_fixed_batches
from simulate_loss_training import evaluate
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA, validate_feature_cache
from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "checkpoints/frozen_features.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/artifact_verification.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.cache
    cache_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest)
    features, valid, labels = [cache[k].to(device) for k in ("features", "valid", "labels")]
    counts = {str(c): int((labels == c).sum()) for c in (0, 1)}
    checked = []
    folders = args.folders or [ROOT / "reports" / name for name in
                              ("convergence_120_epochs", "convergence_baseline_240_epochs", "convergence_refinement_240_epochs")]
    for folder in folders:
        paths = sorted(folder.glob("*_seed*.json"))
        expected = 12 if folder.name == "convergence_120_epochs" else 3
        if not paths or (args.folders is None and len(paths) != expected):
            raise RuntimeError(f"Incomplete experiment set in {folder}: {len(paths)}")
        for report_path in paths:
            report = json.loads(report_path.read_text())
            state = torch.load(report_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
            for artifact in (report, state):
                if (artifact["label_schema"] != LABEL_SCHEMA or artifact["label_map"] != LABEL_MAP
                        or artifact["feature_cache_sha256"] != cache_hash):
                    raise RuntimeError(f"Stale label/cache metadata: {report_path}")
            if report["class_counts"] != counts:
                raise RuntimeError(f"Wrong class counts: {report_path}")
            if [r["epoch"] for r in report["history"]] != list(range(report["epochs"] + 1)):
                raise RuntimeError(f"Incomplete history: {report_path}")
            model = CachedHeads(features.shape[-1], cache["mean_head"]["3.bias"].numel(), cache["variance_floor"]).to(device)
            model.load_state_dict(state["model"])
            objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(**report["loss_config"]))
            epoch = report["settings"]["curriculum_epochs"]
            actual = evaluate(model, objective, features, valid, labels, cache["ids"], epoch)
            actual.update(diagnostics(model, features, valid))
            actual["eval_fixed_minibatch_loss"] = evaluate_fixed_batches(model, objective, features, valid, labels, cache["ids"], epoch)
            fields = ("loss", "contrastive_excess_loss", "uncertainty_regret", "eval_fixed_minibatch_loss")
            errors = {k: abs(actual[k] - report["history"][-1][k]) for k in fields}
            if max(errors.values()) > 1e-6:
                raise RuntimeError(f"Checkpoint does not reproduce report: {report_path}: {errors}")
            checked.append({"report": str(report_path.resolve().relative_to(ROOT)), "max_absolute_error": max(errors.values())})
    result = dict(status="PASS", label_schema=LABEL_SCHEMA, label_map=LABEL_MAP,
                  class_counts=counts, experiments_verified=len(checked), artifacts=checked)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
