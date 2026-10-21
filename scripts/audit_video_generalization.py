#!/usr/bin/env python3
"""Audit frozen-feature video training with byte-deduplicated held-out folds."""
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

from convergence_study import CachedHeads
from simulate_loss_training import balanced_batches, gaussian_output
from vision_model.data.labels import validate_feature_cache
from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss


def deduplicated_indices(manifest: Path, dataset_root: Path, ids: list[str]) -> tuple[list[int], list[list[str]]]:
    rows = {row["video_id"]: row for line in manifest.read_text().splitlines()
            if (row := json.loads(line))}
    seen: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    keep = []
    for index, video_id in enumerate(ids):
        path = dataset_root / rows[video_id]["video_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(video_id)
        if digest not in seen:
            seen[digest] = video_id
            keep.append(index)
    return keep, [group for group in groups.values() if len(group) > 1]


@torch.no_grad()
def normalized_video_embeddings(model: CachedHeads, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    model.eval()
    return F.normalize(gaussian_output(model, features, valid).video_mean, dim=-1)


def prototype_predict(z_train: torch.Tensor, y_train: torch.Tensor, z_test: torch.Tensor) -> torch.Tensor:
    prototypes = F.normalize(torch.stack([z_train[y_train == label].mean(0) for label in (0, 1)]), dim=-1)
    return (z_test @ prototypes.T).argmax(dim=-1)


def audit(cache: dict, keep: list[int], *, epochs: int, folds: int, seed: int, device: str,
          objective_name: str = "supcon") -> dict:
    torch.set_num_threads(4)
    features = cache["features"][keep].to(device)
    valid = cache["valid"][keep].to(device)
    labels = cache["labels"][keep].to(device)
    ids = [cache["ids"][index] for index in keep]
    counts = torch.bincount(labels.cpu(), minlength=2)
    if min(counts) < folds:
        raise ValueError(f"{folds} folds require at least {folds} unique videos in each class; found {counts.tolist()}")
    latent = cache["mean_head"]["3.bias"].numel()
    raw = F.normalize((features * valid[..., None]).sum(1) / valid.sum(1)[:, None], dim=-1)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if objective_name not in {"supcon", "gaussian_bce", "linear_bce"}:
        raise ValueError(f"Unknown objective: {objective_name}")
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(total_epochs=min(19, epochs)))
    predictions = []
    training_losses = []
    for fold, (train, test) in enumerate(splitter.split(range(len(labels)), labels.cpu().numpy()), 1):
        train = torch.as_tensor(train, device=device)
        test = torch.as_tensor(test, device=device)
        torch.manual_seed(seed + fold)
        if objective_name != "linear_bce":
            model = CachedHeads(features.shape[-1], latent, cache["variance_floor"]).to(device)
            model.mean_head.load_state_dict(cache["mean_head"])
            model.variance_head.load_state_dict(cache["variance_head"])
        if objective_name == "supcon":
            parameters = list(model.parameters())
            learning_rate = 5.25e-4
        elif objective_name == "gaussian_bce":
            classifier = torch.nn.Linear(latent, 1).to(device)
            parameters = list(model.mean_head.parameters()) + list(classifier.parameters())
            learning_rate = 5.25e-4
        else:
            classifier = torch.nn.Linear(features.shape[-1], 1).to(device)
            parameters = list(classifier.parameters())
            learning_rate = 0.05
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.04)
        for epoch in range(epochs):
            if objective_name != "linear_bce":
                model.train()
            else:
                classifier.train()
            for local_indices in balanced_batches(labels[train].cpu(), epoch=epoch + seed + fold - 239):
                batch = train[local_indices.to(device)]
                if objective_name == "supcon":
                    output = gaussian_output(model, features[batch], valid[batch])
                    result = objective(output=output, labels=labels[batch],
                                       original_ids=[ids[int(i)] for i in batch.tolist()], epoch=epoch)
                    loss = result.loss_sum / result.sample_count
                elif objective_name == "gaussian_bce":
                    output = gaussian_output(model, features[batch], valid[batch])
                    loss = F.binary_cross_entropy_with_logits(classifier(output.video_mean).squeeze(-1),
                                                              labels[batch].float())
                else:
                    loss = F.binary_cross_entropy_with_logits(classifier(raw[batch]).squeeze(-1),
                                                              labels[batch].float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
        with torch.no_grad():
            if objective_name == "supcon":
                z_train = normalized_video_embeddings(model, features[train], valid[train])
                z_test = normalized_video_embeddings(model, features[test], valid[test])
                trained_pred = prototype_predict(z_train, labels[train], z_test)
                result = objective(output=gaussian_output(model, features[train], valid[train]),
                                   labels=labels[train], original_ids=[ids[int(i)] for i in train.tolist()],
                                   epoch=epochs)
                training_losses.append(float(result.loss_sum / result.sample_count))
            elif objective_name == "gaussian_bce":
                model.eval()
                z_train = gaussian_output(model, features[train], valid[train]).video_mean
                z_test = gaussian_output(model, features[test], valid[test]).video_mean
                trained_pred = (classifier(z_test).squeeze(-1) >= 0).long()
                training_losses.append(float(F.binary_cross_entropy_with_logits(
                    classifier(z_train).squeeze(-1), labels[train].float())))
            else:
                classifier.eval()
                trained_pred = (classifier(raw[test]).squeeze(-1) >= 0).long()
                training_losses.append(float(F.binary_cross_entropy_with_logits(
                    classifier(raw[train]).squeeze(-1), labels[train].float())))
            raw_pred = prototype_predict(raw[train], labels[train], raw[test])
            for local, index in enumerate(test.tolist()):
                predictions.append({"fold": fold, "video_id": ids[index], "label": int(labels[index]),
                                    "raw_pretrained_prediction": int(raw_pred[local]),
                                    "trained_head_prediction": int(trained_pred[local])})
    return {
        "scope": "Stratified held-out video folds after exact byte deduplication; physical-scene independence unverified",
        "epochs_per_fold": epochs, "folds": folds, "seed": seed,
        "objective": objective_name,
        "training_recipe": "balanced 3+3 AdamW minibatches; no variance-only phase because it freezes the embedding head",
        "training_loss_mean": sum(training_losses) / len(training_losses),
        "training_losses_by_fold": training_losses,
        "unique_videos": len(ids), "class_counts": counts.tolist(),
        "raw_pretrained_accuracy": sum(r["raw_pretrained_prediction"] == r["label"] for r in predictions) / len(predictions),
        "trained_head_accuracy": sum(r["trained_head_prediction"] == r["label"] for r in predictions) / len(predictions),
        "predictions": predictions,
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
    parser.add_argument("--objective", choices=("supcon", "gaussian_bce", "linear_bce"), default="supcon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2:
        parser.error("epochs must be positive and folds at least two")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest)
    keep, duplicates = deduplicated_indices(args.manifest, args.dataset_root, cache["ids"])
    report = audit(cache, keep, epochs=args.epochs, folds=args.folds, seed=args.seed, device=args.device,
                   objective_name=args.objective)
    report["duplicate_groups_removed"] = duplicates
    report["feature_cache_sha256"] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
