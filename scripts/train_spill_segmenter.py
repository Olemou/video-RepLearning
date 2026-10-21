#!/usr/bin/env python3
"""Train and evaluate a spill heatmap head from explicit pixel masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from vision_model.data.config import ThermalAugConfig
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.spill_localizer import SpillSegmentationHead, spill_segmentation_loss
from train_spill_localizer import local_model_source, validate_cache


def load_mask(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if array.shape != (256, 256):
        raise ValueError(f"Mask must be 256x256: {path} has {array.shape}")
    # Max pooling preserves small annotated spill regions on the 16x16 target grid.
    return F.adaptive_max_pool2d(torch.from_numpy(array.copy()).float()[None, None], (16, 16))[0, 0] / 255


def annotation_tensors(cache: dict, root: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    targets = torch.zeros(cache["tokens"].shape[:4], dtype=torch.float32)
    supervised = torch.zeros(cache["valid"].shape, dtype=torch.bool)
    counts, annotated_positive_videos = {}, []
    mask_files: list[Path] = []
    for index, (video_id, label) in enumerate(zip(cache["ids"], cache["labels"].tolist())):
        if label == 0:
            # A clean video supplies exact all-background supervision.
            supervised[index] = cache["valid"][index]
            counts[video_id] = int(supervised[index].sum())
            continue
        mask_dir = root / video_id / "masks"
        for tubelet in torch.where(cache["valid"][index])[0].tolist():
            pair = []
            for slot in (2 * tubelet, 2 * tubelet + 1):
                found = sorted(mask_dir.glob(f"slot_{slot:03d}_source_*.png"))
                if len(found) > 1:
                    raise ValueError(f"Multiple masks for {video_id} slot {slot}")
                pair.extend(found)
            # Both constituent frames must be explicit; one frame cannot define
            # the two-frame V-JEPA tubelet target.
            if len(pair) == 2:
                targets[index, tubelet] = torch.stack([load_mask(path) for path in pair]).amax(0)
                supervised[index, tubelet] = True
                mask_files.extend(pair)
        counts[video_id] = int(supervised[index].sum())
        if supervised[index].any():
            annotated_positive_videos.append(video_id)
    missing = [cache["ids"][i] for i in torch.where(cache["labels"] == 1)[0].tolist()
               if not supervised[i].any()]
    if missing:
        raise ValueError(
            "Frames were exported, but every positive video needs at least one fully "
            "annotated two-frame pair saved under dataset/spill_annotations/<video_id>/masks/. "
            f"Missing: {', '.join(missing)}. Start the editor with "
            "`PYTHONPATH=src .venv/bin/python -m vision_model.spill annotate --path dataset/spill_annotations`, "
            "open http://127.0.0.1:8765, draw each spill, and press Save + next."
        )
    positive_pixels = int(targets[supervised].sum())
    if positive_pixels == 0:
        raise ValueError("Annotations contain no positive spill pixels")
    digest = hashlib.sha256()
    for path in sorted(mask_files):
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return targets, supervised, {
        "annotated_tubelets_by_video": counts,
        "annotated_positive_videos": annotated_positive_videos,
        "positive_target_pixels_16x16": positive_pixels,
        "mask_sha256": digest.hexdigest(),
    }


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, supervised: torch.Tensor) -> dict:
    prediction = logits[supervised].sigmoid() >= .5
    truth = targets[supervised] >= .5
    tp = int((prediction & truth).sum()); fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum()); tn = int((~prediction & ~truth).sum())
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    return {
        "iou": divide(tp, tp + fp + fn),
        "dice": divide(2 * tp, 2 * tp + fp + fn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@torch.no_grad()
def predict(model, cache, indices, device, batch_size):
    model.eval(); result = []
    for batch in indices.split(batch_size):
        result.append(model(cache["tokens"][batch].to(device)).cpu())
    return torch.cat(result)


def fit(cache, targets, supervised, indices, args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SpillSegmentationHead(decoder_size=args.decoder_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    selected_targets = targets[indices][supervised[indices]]
    positives = selected_targets.sum(); negatives = selected_targets.numel() - positives
    positive_weight = float((negatives / positives.clamp_min(1)).clamp(1, 50))
    generator = torch.Generator().manual_seed(seed); history = []
    for epoch in range(1, args.epochs + 1):
        order = indices[torch.randperm(len(indices), generator=generator)]
        total = bce_total = dice_total = seen = 0.
        model.train()
        for batch in order.split(args.batch_size):
            logits = model(cache["tokens"][batch].to(args.device))
            loss, parts = spill_segmentation_loss(
                logits, targets[batch].to(args.device), supervised[batch].to(args.device),
                positive_weight=positive_weight, dice_weight=args.dice_weight,
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch); bce_total += float(parts["binary_cross_entropy"]) * len(batch)
            dice_total += float(parts["dice_loss"]) * len(batch); seen += len(batch)
        logits = predict(model, cache, indices, torch.device(args.device), args.batch_size)
        row = {"epoch": epoch, "loss": total / seen, "bce": bce_total / seen,
               "dice_loss": dice_total / seen,
               **segmentation_metrics(logits, targets[indices], supervised[indices])}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps({"seed": seed, **row}), flush=True)
    return model.cpu(), history, positive_weight


def cross_validate(cache, targets, supervised, args):
    from sklearn.model_selection import StratifiedKFold
    labels = cache["labels"].numpy(); all_logits = torch.zeros_like(targets)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed); folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        model, history, weight = fit(cache, targets, supervised, torch.tensor(train), args, args.seed + fold)
        test_indices = torch.tensor(test)
        all_logits[test_indices] = predict(model.to(args.device), cache, test_indices, torch.device(args.device), args.batch_size)
        folds.append({"fold": fold, "train_ids": [cache["ids"][i] for i in train],
                      "test_ids": [cache["ids"][i] for i in test], "positive_weight": weight,
                      "metrics": segmentation_metrics(all_logits[test_indices], targets[test_indices], supervised[test_indices]),
                      "final_training": history[-1]})
    return {"folds": folds, "pooled_out_of_fold": segmentation_metrics(all_logits, targets, supervised)}


def render_mp4(frames, path):
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "4", "-i", "pipe:0",
                             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                            input=(frames * 255).round().astype(np.uint8).tobytes(), capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.decode())


@torch.no_grad()
def visualize(cache, model, args):
    transform = OfficialVJEPA2Transform(local_model_source(cache["model_id"]), thermal_config=ThermalAugConfig(image_size=256))
    dataset = VideoManifestDataset(args.manifest, dataset_root=args.dataset_root,
                                   sampling=TemporalSamplingConfig(**cache["sampling"]), transform=transform)
    mean = torch.tensor(cache["normalization"]["mean"])[None, :, None, None]
    std = torch.tensor(cache["normalization"]["std"])[None, :, None, None]
    model = model.to(args.device).eval(); rows = []
    for index, item in enumerate(dataset):
        probability = model(cache["tokens"][index:index+1].to(args.device))[0].sigmoid().cpu()
        probability[~cache["valid"][index]] = 0
        heat = F.interpolate(probability[:, None], (256, 256), mode="bilinear", align_corners=False)[:, 0]
        heat = heat.repeat_interleave(2, 0).numpy()
        rgb = (item["pixel_values_videos"].float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        selected = np.flatnonzero(cache["valid"][index].repeat_interleave(2).numpy())
        overlay = (.45 * rgb + .55 * plt.get_cmap("coolwarm")(heat)[..., :3]).clip(0, 1)
        sample = selected[np.linspace(0, len(selected)-1, min(6, len(selected))).astype(int)]
        folder = args.output / "videos" / f"video_{index+1:03d}"; folder.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, len(sample), figsize=(3*len(sample), 6), squeeze=False, constrained_layout=True)
        for column, frame in enumerate(sample):
            axes[0,column].imshow(rgb[frame]); axes[1,column].imshow(overlay[frame])
            axes[0,column].set_title(f"source frame {int(item['frame_indices'][frame])}")
            for axis in axes[:,column]: axis.axis("off")
        fig.suptitle(f"{item['video_id']} | mask-supervised spill probability | fixed blue=0, red=1")
        fig.savefig(folder / "spill_segmentation.png", dpi=120); plt.close(fig)
        render_mp4(np.concatenate((rgb[selected], overlay[selected]), 2), folder / "spill_segmentation.mp4")
        np.savez_compressed(folder / "spill_segmentation.npz", probability=probability.numpy(),
                            sampled_frame_indices=item["frame_indices"].numpy(), tubelet_valid=cache["valid"][index].numpy())
        rows.append({"video_id": item["video_id"], "label": item["label"], "directory": str(folder.relative_to(args.output))})
        print(f"rendered {index+1:02d}/{len(dataset)} {item['video_id']}", flush=True)
    (args.output / "inference_results.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path,
                        help="directory used to resolve video_path entries (default: manifest directory)")
    parser.add_argument("--token-cache", type=Path, default=ROOT / "checkpoints/local_spatial_video_tokens.pt")
    parser.add_argument("--annotations", "--path", dest="annotations", type=Path,
                        default=ROOT / "dataset/spill_annotations",
                        help="folder containing per-video frames and masks from the annotation editor")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/spill_segmentation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decoder-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2 or args.batch_size < 1: parser.error("epochs, folds, and batch size must be positive")
    cache = torch.load(args.token_cache, map_location="cpu", weights_only=True); validate_cache(cache, args.manifest)
    if cache["token_layer"] != "patch_embed": raise ValueError("Segmentation needs local patch-embedding tokens")
    targets, supervised, annotation_report = annotation_tensors(cache, args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    cv = cross_validate(cache, targets, supervised, args)
    indices = torch.arange(len(cache["labels"])); model, history, positive_weight = fit(cache, targets, supervised, indices, args, args.seed)
    final_logits = predict(model.to(args.device), cache, indices, torch.device(args.device), args.batch_size)
    report = {"objective": "mask-supervised weighted BCE plus soft Dice", "label_map": LABEL_MAP,
              "annotation_report": annotation_report, "positive_weight": positive_weight,
              "training_metrics": segmentation_metrics(final_logits, targets, supervised),
              "cross_validation": cv, "history": history,
              "limitations": ["The encoder remains frozen.", "Report held-out pixel metrics; training maps alone do not prove generalization."]}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "decoder_size": args.decoder_size, "label_schema": LABEL_SCHEMA,
                "label_map": LABEL_MAP, "mask_sha256": annotation_report["mask_sha256"]}, args.output / "spill_segmenter.pt")
    visualize(cache, model, args)
    from plot_experiment_loss import main as plot_loss
    old_argv = sys.argv
    try:
        sys.argv = ["plot_experiment_loss.py", str(args.output / "training_report.json"),
                    "--output", str(args.output / "loss_curve.png")]
        plot_loss()
    finally:
        sys.argv = old_argv


if __name__ == "__main__": main()
