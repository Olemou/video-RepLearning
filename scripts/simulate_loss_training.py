#!/usr/bin/env python3
"""Run a 20-epoch frozen-encoder experiment on the complete real dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from vision_model.data.sampling import TemporalSamplingConfig  # noqa: E402
from vision_model.data.config import ThermalAugConfig  # noqa: E402
from vision_model.data.transforms import OfficialVJEPA2Transform  # noqa: E402
from vision_model.data.video_dataset import VideoManifestDataset  # noqa: E402
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA  # noqa: E402
from vision_model.models.encoder import load_pretrained_encoder  # noqa: E402
from vision_model.models.gaussian import temporal_gaussian  # noqa: E402
from vision_model.training.loss import (  # noqa: E402
    GaussianContrastiveLossConfig,
    StableUncertaintyContrastiveLoss,
)


def gaussian_output(model, features: torch.Tensor, valid: torch.Tensor):
    mean_t, variance_t, mean, variance = temporal_gaussian(
        model.mean_head, model.variance_head, features, valid, model.variance_floor)
    return SimpleNamespace(
        temporal_mean=mean_t,
        temporal_variance=variance_t,
        video_mean=mean,
        video_variance=variance,
        temporal_valid_mask=valid,
    )


def balanced_batches(labels: torch.Tensor, *, epoch: int, per_class: int = 3) -> list[torch.Tensor]:
    if labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Balanced batches require one-dimensional binary labels")
    if per_class < 2:
        raise ValueError("per_class must be at least two to provide distinct positives")
    generator = torch.Generator().manual_seed(239 + epoch)
    zero = torch.where(labels == 0)[0]
    one = torch.where(labels == 1)[0]
    if len(zero) < per_class or len(one) < per_class:
        raise RuntimeError(f"At least {per_class} distinct videos from each label are required")
    zero = zero[torch.randperm(len(zero), generator=generator)]
    one = one[torch.randperm(len(one), generator=generator)]
    steps = math.ceil(max(len(zero), len(one)) / per_class)
    batches = []
    for step in range(steps):
        offsets = torch.arange(per_class) + step * per_class
        zero_batch = zero[offsets % len(zero)]
        one_batch = one[offsets % len(one)]
        combined = torch.cat((zero_batch, one_batch))
        batches.append(combined[torch.randperm(len(combined), generator=generator)])
    return batches


@torch.no_grad()
def evaluate(model, objective, features, valid, labels, ids, epoch: int) -> dict[str, float]:
    model.mean_head.eval()
    model.variance_head.eval()
    output = gaussian_output(model, features, valid)
    result = objective(output=output, labels=labels, original_ids=ids, epoch=epoch)
    normalized = torch.nn.functional.normalize(output.video_mean.float(), dim=-1)
    similarities = normalized @ normalized.T
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    negative = ~labels[:, None].eq(labels[None, :])
    valid_variance = output.temporal_variance[valid]
    residual_square = (output.temporal_mean - output.video_mean.unsqueeze(1)).square()[valid]
    log_variance = valid_variance.clamp_min(1e-12).log().flatten()
    log_residual = residual_square.clamp_min(1e-12).log().flatten()
    correlation = torch.corrcoef(torch.stack((log_variance, log_residual)))[0, 1]
    nearest = similarities.masked_fill(diagonal, -torch.inf).argmax(dim=1)
    positive_pairs = int(positive.sum().item())
    return {
        "loss": float(result.loss_sum / result.sample_count),
        "contrastive_loss": float(result.contrastive_sum / result.sample_count),
        "contrastive_reference_entropy": float(result.positive_entropy_sum / result.sample_count),
        "contrastive_excess_loss": float(
            (result.contrastive_sum - result.positive_entropy_sum) / result.sample_count
        ),
        "uncertainty_loss": float(result.uncertainty_sum / result.sample_count),
        "positive_cosine": float(similarities[positive].mean()),
        "negative_cosine": float(similarities[negative].mean()),
        "cosine_margin": float(similarities[positive].mean() - similarities[negative].mean()),
        "temporal_variance_mean": float(valid_variance.mean()),
        "temporal_variance_min": float(valid_variance.min()),
        "temporal_variance_max": float(valid_variance.max()),
        "temporal_variance_median": float(valid_variance.median()),
        "variance_near_floor_fraction": float((valid_variance <= model.variance_floor * 2).float().mean()),
        "residual_square_mean": float(residual_square.mean()),
        "variance_to_residual_ratio": float(valid_variance.mean() / residual_square.mean().clamp_min(1e-12)),
        "log_variance_residual_correlation": float(correlation),
        "video_variance_mean": float(output.video_variance.mean()),
        "leave_one_out_nearest_neighbor_accuracy": float((labels[nearest] == labels).float().mean()),
        "mean_positive_pair_weight": float(result.positive_weight_sum / max(positive_pairs, 1)),
        "mean_negative_pair_weight": float(result.negative_weight_sum / max(int(negative.sum()), 1)),
        "positive_lambda": result.positive_lambda,
        "negative_temperature": result.negative_temperature,
    }


def slope(values: list[float]) -> float:
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    denominator = sum((x - x_mean) ** 2 for x in range(n))
    return sum((x - x_mean) * (y - y_mean) for x, y in enumerate(values)) / denominator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset" / "manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path,
                        help="directory used to resolve video_path entries (default: manifest directory)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "loss_simulation_20_epochs.json")
    parser.add_argument("--feature-cache", type=Path, help="Save frozen features for reproducible head experiments")
    parser.add_argument("--preserve-field-of-view", action="store_true",
                        help="Letterbox the complete frame to the checkpoint grid instead of center cropping")
    parser.add_argument("--horizontal-flip-view", action="store_true",
                        help="Cache a whole-clip horizontal mirror as a second contrastive training view")
    parser.add_argument("--encode-only", action="store_true", help="Save features and stop before training")
    parser.add_argument("--mask-encoder-padding", action=argparse.BooleanOptionalAction, default=True,
                        help="Exclude padded tubelets from every attention layer; disable only for historical ablations")
    parser.add_argument(
        "--normalization",
        choices=("thermal", "checkpoint"),
        default="checkpoint",
        help="Use dataset thermal statistics or the pretrained checkpoint statistics",
    )
    args = parser.parse_args(argv)
    if args.epochs < 2:
        raise ValueError("At least two epochs are required")
    if args.encode_only and args.feature_cache is None:
        parser.error("--encode-only requires --feature-cache")

    random.seed(239)
    torch.manual_seed(239)
    device = torch.device(args.device)
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    thermal_config = ThermalAugConfig(image_size=256) if args.normalization == "thermal" else None
    transform = OfficialVJEPA2Transform(model_id, thermal_config=thermal_config,
                                      preserve_field_of_view=args.preserve_field_of_view)
    dataset = VideoManifestDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        sampling=TemporalSamplingConfig(clip_length=64, target_fps=4, training=False, short_video_policy="repeat_last"),
        transform=transform,
    )
    model = load_pretrained_encoder(model_id, local_files_only=True)
    model.freeze_encoder(True)
    encoder_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=encoder_dtype).eval()

    cached_features, cached_valid, labels, ids = [], [], [], []
    cached_flipped = []
    for index in range(len(dataset)):
        item = dataset[index]
        pixels = item["pixel_values_videos"].unsqueeze(0).to(device)
        frame_valid = item["temporal_valid_mask"].unsqueeze(0).to(device)
        tubelets = pixels.shape[1] // model.encoder.architecture.tubelet_size
        valid = frame_valid.unflatten(1, (tubelets, model.encoder.architecture.tubelet_size)).all(dim=-1)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=encoder_dtype,
            enabled=device.type == "cuda" and encoder_dtype != torch.float32,
        ):
            tokens = model.encoder(pixels.permute(0, 2, 1, 3, 4).contiguous(),
                                   temporal_valid_mask=valid if args.mask_encoder_padding else None)
            spatial = (pixels.shape[-2] // 16) * (pixels.shape[-1] // 16)
            features = tokens.unflatten(1, (tubelets, spatial)).mean(dim=2)
        if args.horizontal_flip_view:
            # Mirror every frame together before preprocessing. Preserve temporal
            # support and spill temperature cues; spatial policy is identical for both views.
            flipped_pixels = transform(item["video"].flip(-1)).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=encoder_dtype,
                enabled=device.type == "cuda" and encoder_dtype != torch.float32,
            ):
                flipped_tokens = model.encoder(flipped_pixels.permute(0, 2, 1, 3, 4).contiguous(),
                    temporal_valid_mask=valid if args.mask_encoder_padding else None)
                flipped_features = flipped_tokens.unflatten(1, (tubelets, spatial)).mean(dim=2)
            cached_flipped.append(flipped_features.float().cpu())
        cached_features.append(features.float().cpu())
        cached_valid.append(valid.cpu())
        labels.append(item["label"])
        ids.append(item["video_id"])
        print(f"encoded {index + 1:02d}/{len(dataset)} {item['video_id']}", flush=True)

    features = torch.cat(cached_features).to(device)
    valid = torch.cat(cached_valid).to(device)
    labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    model.encoder.to("cpu")
    model.mean_head.to(device=device, dtype=torch.float32)
    model.variance_head.to(device=device, dtype=torch.float32)
    if args.feature_cache is not None:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2,
            **({"horizontal_flip_features": torch.cat(cached_flipped),
                 "augmentation": {"policy": "identity_and_whole_clip_horizontal_flip"}}
               if args.horizontal_flip_view else {"augmentation": {"policy": "none"}}),
            "encoder_padding_policy": "masked_attention_v1" if args.mask_encoder_padding else "unmasked_legacy",
            "label_schema": LABEL_SCHEMA, "label_map": LABEL_MAP,
            "categories": [record.category for record in dataset.records],
            "features": features.cpu(), "valid": valid.cpu(),
            "labels": labels_tensor.cpu(), "ids": ids,
            "source_sequences": [record.source_sequence for record in dataset.records],
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "model_id": model_id, "normalization": transform.normalization,
            "spatial_policy": "mean_letterbox_full_frame" if args.preserve_field_of_view else "checkpoint_center_crop",
            "sampling": dataset.sampler.config.__dict__,
            "encoder_dtype": str(encoder_dtype), "seed": 239,
            "mean_head": {k: v.cpu() for k, v in model.mean_head.state_dict().items()},
            "variance_head": {k: v.cpu() for k, v in model.variance_head.state_dict().items()},
            "variance_floor": model.variance_floor,
        }, args.feature_cache)
        print(f"saved frozen features: {args.feature_cache}", flush=True)
    if args.encode_only:
        return 0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loss_config = GaussianContrastiveLossConfig(total_epochs=args.epochs - 1)
    objective = StableUncertaintyContrastiveLoss(loss_config)
    parameters = list(model.mean_head.parameters()) + list(model.variance_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5.25e-4, weight_decay=0.04)

    no_training_metrics = {
        "training_loss": None,
        "training_contrastive_loss": None,
        "training_contrastive_reference_entropy": None,
        "training_contrastive_excess_loss": None,
        "training_uncertainty_loss": None,
    }
    history = [{
        "epoch": 0,
        "phase": "before_training",
        **evaluate(model, objective, features, valid, labels_tensor, ids, 0),
        **no_training_metrics,
        "gradient_norm": 0.0,
    }]
    print(json.dumps(history[-1], sort_keys=True), flush=True)
    for epoch in range(args.epochs):
        model.mean_head.train()
        model.variance_head.train()
        gradient_norms = []
        training_loss_sum = 0.0
        training_contrastive_sum = 0.0
        training_positive_entropy_sum = 0.0
        training_uncertainty_sum = 0.0
        training_sample_count = 0
        for batch_indices in balanced_batches(labels_tensor.cpu(), epoch=epoch):
            batch_indices = batch_indices.to(device)
            output = gaussian_output(model, features[batch_indices], valid[batch_indices])
            result = objective(
                output=output,
                labels=labels_tensor[batch_indices],
                original_ids=[ids[index] for index in batch_indices.cpu().tolist()],
                epoch=epoch,
            )
            loss = result.loss_sum / result.sample_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            gradient_norms.append(float(gradient_norm))
            training_loss_sum += float(result.loss_sum.detach())
            training_contrastive_sum += float(result.contrastive_sum)
            training_positive_entropy_sum += float(result.positive_entropy_sum)
            training_uncertainty_sum += float(result.uncertainty_sum)
            training_sample_count += result.sample_count
        metrics = evaluate(model, objective, features, valid, labels_tensor, ids, epoch)
        metrics.update(
            epoch=epoch + 1,
            phase="trained",
            training_loss=training_loss_sum / training_sample_count,
            training_contrastive_loss=training_contrastive_sum / training_sample_count,
            training_contrastive_reference_entropy=training_positive_entropy_sum / training_sample_count,
            training_contrastive_excess_loss=(
                training_contrastive_sum - training_positive_entropy_sum
            ) / training_sample_count,
            training_uncertainty_loss=training_uncertainty_sum / training_sample_count,
            gradient_norm=sum(gradient_norms) / len(gradient_norms),
        )
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

    initial, final = history[0], history[-1]
    last_five = [row["contrastive_loss"] for row in history[-5:]]
    last_five_excess = [row["contrastive_excess_loss"] for row in history[-5:]]
    finite = all(math.isfinite(value) for row in history for key, value in row.items() if isinstance(value, float))
    assessment = {
        "finite": finite,
        "contrastive_reduction_fraction": (initial["contrastive_loss"] - final["contrastive_loss"]) / max(abs(initial["contrastive_loss"]), 1e-12),
        "contrastive_excess_reduction_fraction": (
            initial["contrastive_excess_loss"] - final["contrastive_excess_loss"]
        ) / max(abs(initial["contrastive_excess_loss"]), 1e-12),
        "total_reduction_fraction": (initial["loss"] - final["loss"]) / max(abs(initial["loss"]), 1e-12),
        "margin_improvement": final["cosine_margin"] - initial["cosine_margin"],
        "last_five_contrastive_slope": slope(last_five),
        "last_five_contrastive_excess_slope": slope(last_five_excess),
        "variance_collapsed": final["temporal_variance_min"] <= model.variance_floor * 1.01,
    }
    assessment["contrastive_near_floor"] = bool(
        finite
        and final["contrastive_excess_loss"] < 0.001
        and final["training_contrastive_excess_loss"] < 0.001
        and assessment["margin_improvement"] > 0
    )
    assessment["convergence_scope"] = "Contrastive fit only; use convergence_study.py to assess variance fitting and stability"
    payload = {
        "experiment": "frozen pretrained encoder; real 24-video dataset; balanced 3+3 batches",
        "labels": LABEL_MAP,
        "label_schema": LABEL_SCHEMA,
        "class_counts": {"0": labels.count(0), "1": labels.count(1)},
        "epochs": args.epochs,
        "normalization": {
            "mode": args.normalization,
            **transform.normalization,
        },
        "metric_note": (
            "Raw multi-positive contrastive loss includes irreducible positive entropy; "
            "contrastive_excess_loss measures the gap above the current entropy lower bound, not proof of a global optimum."
        ),
        "loss_config": loss_config.__dict__,
        "history": history,
        "assessment": assessment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    checkpoint = PROJECT_ROOT / "checkpoints" / "loss_simulation_heads.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean_head": model.mean_head.state_dict(),
        "variance_head": model.variance_head.state_dict(),
        "loss_config": loss_config.__dict__,
        "assessment": assessment,
    }, checkpoint)
    print(json.dumps({"assessment": assessment, "json": str(args.output), "csv": str(csv_path), "checkpoint": str(checkpoint)}, indent=2), flush=True)
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
