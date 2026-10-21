#!/usr/bin/env python3
"""Held-out test of a residual, two-view supervised contrastive video adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from audit_video_generalization import deduplicated_indices, prototype_predict
from simulate_loss_training import balanced_batches
from vision_model.data.labels import validate_feature_cache
from vision_model.training.video_contrastive import (
    ResidualContrastiveHead, masked_temporal_mean, supervised_contrastive_excess, temporal_views,
)


def evaluate(cache: dict, keep: list[int], *, epochs: int, folds: int, seed: int,
             device: str, keep_fraction: float, tether: float, bottleneck: int) -> dict:
    torch.set_num_threads(4)
    features = cache["features"][keep].to(device)
    valid = cache["valid"][keep].to(device)
    labels = cache["labels"][keep].to(device)
    ids = [cache["ids"][i] for i in keep]
    class_counts = torch.bincount(labels.cpu(), minlength=2).tolist()
    if min(class_counts) < folds:
        raise ValueError(f"Need at least {folds} unique videos per class; found {class_counts}")
    raw = F.normalize(masked_temporal_mean(features, valid), dim=-1)
    splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = []
    fit = []
    for fold, (train_np, test_np) in enumerate(splits.split(range(len(labels)), labels.cpu().numpy()), 1):
        train = torch.as_tensor(train_np, device=device)
        test = torch.as_tensor(test_np, device=device)
        torch.manual_seed(seed + fold)
        head = ResidualContrastiveHead(features.shape[-1], bottleneck).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        for epoch in range(epochs):
            head.train()
            generator = torch.Generator().manual_seed(seed * 100000 + fold * 1000 + epoch)
            for local in balanced_batches(labels[train].cpu(), epoch=epoch + seed + fold - 239):
                batch = train[local.to(device)]
                views = temporal_views(valid[batch], keep_fraction=keep_fraction, generator=generator)
                pooled = masked_temporal_mean(
                    features[batch].repeat_interleave(2, dim=0), views.flatten(0, 1))
                raw_views = F.normalize(pooled, dim=-1)
                projected = head(pooled)
                ce, entropy, excess = supervised_contrastive_excess(
                    projected, labels[batch].repeat_interleave(2))
                proximal = (projected - raw_views).square().sum(-1).mean()
                loss = excess + tether * proximal
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        head.eval()
        with torch.no_grad():
            z_train = head(masked_temporal_mean(features[train], valid[train]))
            z_test = head(masked_temporal_mean(features[test], valid[test]))
            pred = prototype_predict(z_train, labels[train], z_test)
            raw_pred = prototype_predict(raw[train], labels[train], raw[test])
            ce, entropy, excess = supervised_contrastive_excess(z_train, labels[train])
            fit.append({"fold": fold, "contrastive_loss": float(ce), "positive_entropy": float(entropy),
                        "contrastive_excess": float(excess),
                        "mean_cosine_to_pretrained": float((z_train * raw[train]).sum(-1).mean())})
            for position, index in enumerate(test.tolist()):
                predictions.append({"fold": fold, "video_id": ids[index], "label": int(labels[index]),
                                    "pretrained_prediction": int(raw_pred[position]),
                                    "adapter_prediction": int(pred[position])})
    return {
        "protocol": "Seven stratified held-out video folds after byte deduplication; scene independence unverified",
        "epochs": epochs, "folds": folds, "seed": seed, "keep_fraction": keep_fraction,
        "tether": tether, "bottleneck": bottleneck,
        "unique_videos": len(ids), "class_counts": class_counts,
        "pretrained_accuracy": sum(p["pretrained_prediction"] == p["label"] for p in predictions) / len(predictions),
        "adapter_accuracy": sum(p["adapter_prediction"] == p["label"] for p in predictions) / len(predictions),
        "mean_training_excess": sum(row["contrastive_excess"] for row in fit) / len(fit),
        "fold_fit": fit, "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-fraction", type=float, default=0.7)
    parser.add_argument("--tether", type=float, default=0.1)
    parser.add_argument("--bottleneck", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2 or args.bottleneck < 1 or args.tether < 0:
        parser.error("Invalid epochs, folds, bottleneck or tether")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest)
    keep, duplicates = deduplicated_indices(args.manifest, args.dataset_root, cache["ids"])
    report = evaluate(cache, keep, epochs=args.epochs, folds=args.folds, seed=args.seed,
                      device=args.device, keep_fraction=args.keep_fraction,
                      tether=args.tether, bottleneck=args.bottleneck)
    report["removed_byte_duplicates"] = duplicates
    report["feature_cache_sha256"] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"predictions", "fold_fit"}}, indent=2))


if __name__ == "__main__":
    main()
