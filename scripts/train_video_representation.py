#!/usr/bin/env python3
"""Train and evaluate stage-one supervised contrastive video representations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from audit_video_generalization import deduplicated_indices, prototype_predict
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA, validate_feature_cache
from vision_model.training.video_contrastive import fit_video_representation, masked_temporal_mean, representation_diagnostics
from vision_model.training.vico import vico_pairwise_contrastive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--head-type", choices=("residual", "mlp"), default="mlp")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--curriculum-epochs", type=int, default=100)
    parser.add_argument("--positive-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--objective", choices=("supcon", "margin", "vico"), default="vico")
    parser.add_argument("--positive-margin", type=float, default=0.9)
    parser.add_argument("--negative-margin", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-12)
    parser.add_argument("--reset-optimizer-after-curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tether", type=float, default=0.0)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--convergence-patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest, encoder_padding_policy="masked_attention_v1")
    keep, duplicates = deduplicated_indices(args.manifest, args.dataset_root, cache["ids"])
    # Stable video-ID ordering makes folds comparable across equivalent caches.
    keep.sort(key=lambda index: cache["ids"][index])
    features = cache["features"][keep].to(args.device)
    valid = cache["valid"][keep].to(args.device)
    labels = cache["labels"][keep].to(args.device)
    augmented = cache.get("horizontal_flip_features")
    if augmented is not None:
        augmented = augmented[keep].to(args.device)
    ids = [cache["ids"][i] for i in keep]
    counts = [int((labels == c).sum()) for c in (0, 1)]
    if args.folds < 2 or min(counts) < args.folds:
        parser.error(f"Need at least folds unique videos per class; found {counts}")
    if any(n - (n + args.folds - 1) // args.folds < 2 for n in counts):
        parser.error("Each training fold needs two distinct videos per class")
    settings = dict(epochs=args.epochs, bottleneck=args.bottleneck,
                    learning_rate=args.learning_rate, temperature=args.temperature, tether=args.tether,
                    convergence_tolerance=args.convergence_tolerance, convergence_patience=args.convergence_patience,
                    objective=args.objective, positive_margin=args.positive_margin, negative_margin=args.negative_margin,
                    weight_decay=args.weight_decay, head_type=args.head_type,
                    embedding_dim=args.embedding_dim, curriculum_epochs=args.curriculum_epochs,
                    positive_weighting=args.positive_weighting, negative_weighting=args.negative_weighting,
                    adam_epsilon=args.adam_epsilon,
                    reset_optimizer_after_curriculum=args.reset_optimizer_after_curriculum)
    raw = F.normalize(masked_temporal_mean(features, valid), dim=-1)
    predictions = []
    fold_convergence = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_np, test_np) in enumerate(splitter.split(ids, labels.cpu().numpy()), 1):
        train = torch.as_tensor(train_np, device=args.device)
        test = torch.as_tensor(test_np, device=args.device)
        model, fold_history = fit_video_representation(features[train], valid[train], labels[train],
                                            seed=args.seed + fold, augmented_features=augmented[train] if augmented is not None else None, **settings)
        fold_convergence.append({"fold": fold, "epochs_run": len(fold_history),
                                 "converged": fold_history[-1]["converged"],
                                 **({"max_anchor_excess": fold_history[-1]["max_anchor_excess"]}
                                    if args.objective != "vico" else {}),
                                 "objective_loss": fold_history[-1]["objective_loss"],
                                 **({"max_positive_pair_loss": fold_history[-1]["max_positive_pair_loss"]}
                                    if args.objective == "vico" else {}),
                                 **({"max_margin_violation": fold_history[-1]["max_margin_violation"]}
                                    if args.objective == "margin" else {})})
        print(f"fold {fold}/{args.folds}: epochs={len(fold_history)}, converged={fold_history[-1]['converged']}", flush=True)
        result = model.predict(features[test], valid[test])
        baseline = prototype_predict(raw[train], labels[train], raw[test])
        for position, index in enumerate(test.tolist()):
            predictions.append({"video_id": ids[index], "fold": fold, "label": int(labels[index]),
                                "prediction": int(result["labels"][position]),
                                "pretrained_prediction": int(baseline[position])})
    # Fixed hyperparameters and training-only stopping: held-out labels never select a checkpoint.
    model, history = fit_video_representation(features, valid, labels, seed=args.seed, augmented_features=augmented, **settings)
    provenance = {"feature_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "training_video_ids": ids, "removed_byte_duplicates": duplicates}
    augmentation_metadata = dict(cache.get("augmentation", {"policy": "none"}))
    augmentation_metadata.pop("same_source_positive_pairs", None)
    augmentation_metadata["training_positive_policy"] = {
        "vico": "same-source positive views included with weight one",
        "supcon": "all same-class views except the anchor are positives",
        "margin": "same-source positive pairs excluded",
    }[args.objective]
    report = {"protocol": "Stratified held-out whole videos after byte deduplication; scene independence unverified",
              "optimized_equation": "loss_doc/VicoLWIR.pdf Eq. 10" if args.objective == "vico" else args.objective,
              "settings": dict(settings, seed=args.seed, folds=args.folds),
              "class_counts": counts, "predictions": predictions, "fold_convergence": fold_convergence,
              "epochs_run": len(history), "converged": history[-1]["converged"],
              "final_objective_metrics": history[-1],
              "normalization": cache.get("normalization"),
              "spatial_policy": cache.get("spatial_policy", "checkpoint_center_crop"),
              "augmentation": augmentation_metadata,
              "held_out_accuracy": sum(p["prediction"] == p["label"] for p in predictions) / len(predictions),
              "pretrained_accuracy": sum(p["pretrained_prediction"] == p["label"] for p in predictions) / len(predictions),
              "history": history,
              "supcon_reference_on_original_views": representation_diagnostics(model(features, valid), labels, temperature=args.temperature),
              "pretrained_supcon_reference": representation_diagnostics(raw, labels, temperature=args.temperature),
              **provenance}
    if args.objective == "vico":
        _, original_metrics = vico_pairwise_contrastive(
            model(features, valid), labels, temperature=args.temperature,
            epoch=args.curriculum_epochs, curriculum_epochs=args.curriculum_epochs,
            positive_weighting=args.positive_weighting, negative_weighting=args.negative_weighting)
        report["vico_diagnostics_on_original_views"] = {key: float(value) for key, value in original_metrics.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "spill_video_representation_v3", "width": features.shape[-1],
                "bottleneck": args.bottleneck, "head_type": args.head_type,
                "embedding_dim": args.embedding_dim, "state_dict": model.cpu().state_dict(),
                "label_schema": LABEL_SCHEMA, "label_map": LABEL_MAP,
                "normalization": cache.get("normalization"),
                "spatial_policy": cache.get("spatial_policy", "checkpoint_center_crop"),
                "model_id": cache.get("model_id"),
                "sampling": cache.get("sampling"),
                "settings": report["settings"], **provenance}, args.output / "video_representation.pt")
    result = model.predict(features.cpu(), valid.cpu())
    torch.save({**result, "video_ids": ids, "label_schema": LABEL_SCHEMA}, args.output / "video_embeddings.pt")
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("held_out_accuracy", "pretrained_accuracy", "class_counts", "epochs_run", "converged")}, indent=2))
    print(f"Final {args.objective} loss: {history[-1]['objective_loss']:.12g}")
    print(f"Saved representation model, embeddings and report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
