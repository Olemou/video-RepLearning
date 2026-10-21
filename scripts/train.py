#!/usr/bin/env python3
"""Run the validated frozen-encoder training recipe, or inspect its config."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402
from vision_model.data.labels import validate_feature_cache  # noqa: E402

from vision_model.training.loss import (  # noqa: E402
    GaussianContrastiveLossConfig,
    StableUncertaintyContrastiveLoss,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train.yaml")
    parser.add_argument("--inspect", action="store_true", help="Print configuration without training")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if payload["training"]["loss"]["name"] in {"video-supervised-contrastive-v1", "video-margin-contrastive-v2", "vico-pairwise-contrastive-v1"}:
        if args.inspect:
            print(yaml.safe_dump(payload))
            return 0
        from train_video_representation import main as train_representation
        training = payload["training"]
        recipe = training["video_representation"]
        objective_names = {"video-supervised-contrastive-v1": "supcon",
                           "video-margin-contrastive-v2": "margin", "vico-pairwise-contrastive-v1": "vico"}
        command = ["--objective", objective_names[training["loss"]["name"]],
                   "--head-type", recipe.get("head_type", "residual"),
                   "--embedding-dim", str(recipe.get("embedding_dim", 128)),
                   "--curriculum-epochs", str(recipe.get("curriculum_epochs", 100))]
        for key in ("cache", "manifest", "dataset_root", "output"):
            command += ["--" + key.replace("_", "-"), str(PROJECT_ROOT / recipe[key])]
        for key in ("epochs", "seed"):
            command += ["--" + key, str(training[key])]
        for key in ("folds", "bottleneck", "tether", "convergence_tolerance", "convergence_patience"):
            command += ["--" + key.replace("_", "-"), str(recipe[key])]
        if training["loss"]["name"] == "video-margin-contrastive-v2":
            for key in ("positive_margin", "negative_margin"):
                command += ["--" + key.replace("_", "-"), str(training["loss"][key])]
        command += ["--weight-decay", str(payload["optimizer"].get("weight_decay", 0.0))]
        command += ["--adam-epsilon", str(payload["optimizer"].get("eps", 1e-8))]
        command += ["--temperature", str(training["loss"].get("temperature", 0.1)),
                    "--learning-rate", str(payload["optimizer"]["learning_rate"])]
        if args.device:
            command += ["--device", args.device]
        return train_representation(command)
    settings = dict(payload["training"]["loss"])
    name = settings.pop("name")
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(
        total_epochs=int(payload["training"].get("frozen_encoder_recipe", {}).get(
            "curriculum_epochs", payload["training"]["epochs"])), **settings
    ))
    print(f"Configured {name}: {objective.config}")
    if args.inspect:
        return 0
    training = payload["training"]
    recipe = training.get("frozen_encoder_recipe", {})
    if name != StableUncertaintyContrastiveLoss.name:
        raise ValueError(f"Unknown loss name: {name}")
    # This runnable recipe has a fixed, explicitly recorded objective. Reject
    # overrides it does not implement rather than silently ignoring them.
    expected = GaussianContrastiveLossConfig(total_epochs=int(recipe.get("curriculum_epochs", training["epochs"])))
    if objective.config != expected:
        raise ValueError("Frozen recipe requires default loss settings; use convergence_study.py for ablations")
    manifest = PROJECT_ROOT / recipe.get("manifest", "dataset/manifest.jsonl")
    cache = PROJECT_ROOT / recipe.get("feature_cache", "checkpoints/frozen_features.pt")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not cache.exists():
        from simulate_loss_training import main as encode
        encode_args = ["--manifest", str(manifest), "--encode-only", "--feature-cache", str(cache)]
        if args.device:
            encode_args += ["--device", args.device]
        encode(encode_args)
    import torch
    metadata = torch.load(cache, map_location="cpu", weights_only=True)
    validate_feature_cache(metadata, manifest, encoder_padding_policy="masked_attention_v1")
    from convergence_study import main as study
    study_args = ["--cache", str(cache), "--manifest", str(manifest), "--epochs", str(training["epochs"]),
                  "--calibration-epochs", str(recipe.get("calibration_epochs", 160)),
                  "--curriculum-epochs", str(recipe.get("curriculum_epochs", 19)),
                  "--variants", recipe.get("variant", "variance_refinement"),
                  "--output", str(PROJECT_ROOT / recipe.get("output", "reports/training")),
                  "--lr", str(payload["optimizer"]["learning_rate"]),
                  "--refinement-max-iter", str(recipe.get("refinement_max_iter", 10)),
                  "--refinement-precision", recipe.get("refinement_precision", "float64"),
                  "--refinement-threshold", str(recipe.get("refinement_threshold", 1e-3)),
                  "--refinement-patience", str(recipe.get("refinement_patience", 5)),
                  "--seeds", *[str(s) for s in recipe.get("seeds", [training["seed"]])]]
    if args.device:
        study_args += ["--device", args.device]
    return study(study_args)


if __name__ == "__main__":
    raise SystemExit(main())
