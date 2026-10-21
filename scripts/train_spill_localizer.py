#!/usr/bin/env python3
"""Train and render weakly supervised spill localization from real video tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vision_model.data.config import ThermalAugConfig
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset, read_manifest
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.models.spill_localizer import SpillMILHead, spill_mil_loss


def local_model_source(model_id: str) -> str:
    cache = ROOT / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--"))
    ref = cache / "refs/main"
    snapshot = cache / "snapshots" / ref.read_text().strip() if ref.exists() else None
    return str(snapshot if snapshot is not None and snapshot.exists() else model_id)


def extract(args):
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    source = local_model_source(model_id)
    transform = OfficialVJEPA2Transform(source, thermal_config=ThermalAugConfig(image_size=256))
    sampling = TemporalSamplingConfig(clip_length=64, target_fps=4, training=False, short_video_policy="repeat_last")
    dataset = VideoManifestDataset(args.manifest, dataset_root=args.dataset_root,
                                   sampling=sampling, transform=transform)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = load_pretrained_encoder(source, local_files_only=True).encoder.to(device=device, dtype=dtype).eval()
    tokens, masks = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        pixels = item["pixel_values_videos"].unsqueeze(0).to(device)
        valid = item["temporal_valid_mask"].reshape(1, 32, 2).all(-1).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            video = pixels.permute(0, 2, 1, 3, 4).contiguous()
            encoded = (encoder.patch_embed(video) if args.token_layer == "patch_embed"
                       else encoder(video, temporal_valid_mask=valid))
        tokens.append(encoded.reshape(1, 32, 16, 16, 1024).to(dtype=torch.float16, device="cpu"))
        masks.append(valid.cpu())
        print(f"encoded spatial tokens {index + 1:02d}/{len(dataset)} {item['video_id']}", flush=True)
    payload = {
        "format_version": 1,
        "purpose": "weakly_supervised_spill_localization",
        "token_layer": args.token_layer,
        "encoder_padding_policy": "masked_attention_v1",
        "model_id": model_id,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "label_schema": LABEL_SCHEMA,
        "label_map": LABEL_MAP,
        "tokens": torch.cat(tokens),
        "valid": torch.cat(masks),
        "labels": torch.tensor([r.label for r in dataset.records]),
        "ids": [r.video_id for r in dataset.records],
        "categories": [r.category for r in dataset.records],
        "video_paths": [r.video_path for r in dataset.records],
        "sampling": sampling.__dict__,
        "normalization": transform.normalization,
    }
    args.token_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.token_cache)
    del encoder
    if device.type == "cuda": torch.cuda.empty_cache()


def validate_cache(cache, manifest):
    records = read_manifest(manifest)
    if cache.get("label_schema") != LABEL_SCHEMA or cache.get("label_map") != LABEL_MAP:
        raise ValueError("Token cache label schema mismatch")
    if cache.get("encoder_padding_policy") != "masked_attention_v1":
        raise ValueError("Token cache does not mask encoder padding")
    if cache.get("token_layer") not in ("patch_embed", "last_encoder_layer"):
        raise ValueError("Token cache does not record a supported spatial layer")
    if cache.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Token cache manifest mismatch")
    if cache["ids"] != [r.video_id for r in records] or cache["labels"].tolist() != [r.label for r in records]:
        raise ValueError("Token cache IDs or labels do not match manifest")
    if cache["tokens"].shape != (len(records), 32, 16, 16, 1024):
        raise ValueError(f"Unexpected token cache shape: {tuple(cache['tokens'].shape)}")


def batches(indices, batch_size, generator):
    order = indices[torch.randperm(len(indices), generator=generator)]
    return order.split(batch_size)


@torch.no_grad()
def predict(model, cache, indices, device, batch_size=4):
    model.eval(); logits, labels = [], []
    for batch in indices.split(batch_size):
        output = model(cache["tokens"][batch].to(device), cache["valid"][batch].to(device))
        logits.append(output.video_logits.cpu()); labels.append(cache["labels"][batch])
    return torch.cat(logits), torch.cat(labels)


def metrics(logits, labels):
    from sklearn.metrics import roc_auc_score
    predicted = (logits >= 0).long()
    return {
        "accuracy": float((predicted == labels).float().mean()),
        "balanced_accuracy": float(torch.stack([(predicted[labels == c] == c).float().mean() for c in (0, 1)]).mean()),
        "roc_auc": float(roc_auc_score(labels.numpy(), logits.numpy())) if labels.unique().numel() == 2 else None,
        "mean_clean_logit": float(logits[labels == 0].mean()) if (labels == 0).any() else None,
        "mean_spill_logit": float(logits[labels == 1].mean()) if (labels == 1).any() else None,
    }


def fit(cache, train_indices, args, seed, *, epochs=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    model = SpillMILHead(top_fraction=args.top_fraction).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, (epochs or args.epochs) + 1):
        model.train(); totals = {"loss": 0., "video": 0., "clean_tokens": 0., "positive_top": 0.,
                                 "positive_sparsity": 0., "smoothness": 0.}; count = 0
        for batch in batches(train_indices, args.batch_size, generator):
            tokens = cache["tokens"][batch].to(device)
            valid = cache["valid"][batch].to(device)
            labels = cache["labels"][batch].to(device)
            output = model(tokens, valid)
            loss, parts = spill_mil_loss(output, labels, valid,
                                         sparsity_weight=args.sparsity_weight,
                                         smoothness_weight=args.smoothness_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            totals["loss"] += float(loss.detach()) * len(batch)
            for key, value in parts.items(): totals[key] += float(value) * len(batch)
            count += len(batch)
        if epoch == 1 or epoch % 25 == 0 or epoch == (epochs or args.epochs):
            train_logits, train_labels = predict(model, cache, train_indices, device, args.batch_size)
            row = {"epoch": epoch, **{k: v / count for k, v in totals.items()}, **metrics(train_logits, train_labels)}
            history.append(row); print(json.dumps({"seed": seed, **row}), flush=True)
    return model.cpu(), history


def cross_validate(cache, args):
    from sklearn.model_selection import StratifiedKFold
    labels = cache["labels"].numpy()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions = torch.full((len(labels),), torch.nan)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        model, history = fit(cache, torch.tensor(train), args, args.seed + fold)
        logits, actual = predict(model.to(args.device), cache, torch.tensor(test), torch.device(args.device), args.batch_size)
        predictions[test] = logits
        folds.append({"fold": fold, "train_ids": [cache["ids"][i] for i in train],
                      "test_ids": [cache["ids"][i] for i in test], "metrics": metrics(logits, actual),
                      "final_training": history[-1]})
    return {"folds": folds, "pooled_out_of_fold": metrics(predictions, cache["labels"]),
            "predictions": [{"video_id": cache["ids"][i], "label": int(cache["labels"][i]),
                             "logit": float(predictions[i])} for i in range(len(labels))]}


def train(args, cache):
    indices = torch.arange(len(cache["labels"]))
    cv = cross_validate(cache, args)
    model, history = fit(cache, indices, args, args.seed)
    logits, labels = predict(model.to(args.device), cache, indices, torch.device(args.device), args.batch_size)
    report = {
        "objective": "weakly supervised top-k multiple-instance spill localization",
        "label_map": LABEL_MAP,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "training_metrics": metrics(logits, labels),
        "cross_validation": cv,
        "history": history,
        "limitations": [
            "Only video-level labels are available; localization is not validated against pixel masks.",
            "Cross-validation has only seven positive videos and may share acquisition conditions.",
            "The pretrained encoder remains frozen; the new spatial MIL head is trained.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "settings": report["settings"],
                "token_cache_sha256": hashlib.sha256(args.token_cache.read_bytes()).hexdigest(),
                "label_schema": LABEL_SCHEMA, "label_map": LABEL_MAP}, args.output / "spill_localizer.pt")
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return model


def render_mp4(frames, path):
    process = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "4", "-i", "pipe:0",
                              "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                             input=(frames * 255).round().astype(np.uint8).tobytes(), capture_output=True)
    if process.returncode: raise RuntimeError(process.stderr.decode())


@torch.no_grad()
def visualize(args, cache, model):
    transform = OfficialVJEPA2Transform(local_model_source(cache["model_id"]), thermal_config=ThermalAugConfig(image_size=256))
    dataset = VideoManifestDataset(args.manifest, dataset_root=args.dataset_root,
                                   sampling=TemporalSamplingConfig(**cache["sampling"]), transform=transform)
    model = model.to(args.device).eval()
    rows = []
    for index, item in enumerate(dataset):
        valid = cache["valid"][index:index + 1].to(args.device)
        output = model(cache["tokens"][index:index + 1].to(args.device), valid)
        probability = output.token_logits[0].sigmoid().cpu()
        probability[~cache["valid"][index]] = 0
        heat = F.interpolate(probability[:, None], (256, 256), mode="bilinear", align_corners=False)[:, 0]
        heat = heat.repeat_interleave(2, 0).numpy()
        pixels = item["pixel_values_videos"].float()
        mean = torch.tensor(cache["normalization"]["mean"])[None, :, None, None]
        std = torch.tensor(cache["normalization"]["std"])[None, :, None, None]
        rgb = (pixels * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        observed = cache["valid"][index].repeat_interleave(2).numpy()
        selected = np.flatnonzero(observed)
        # Fixed 0..1 scale across every video: red is high learned spill evidence,
        # blue is low. No per-video normalization can manufacture a hotspot.
        color = plt.get_cmap("coolwarm")(heat)[..., :3]
        overlay = (.45 * rgb + .55 * color).clip(0, 1)
        sample = selected[np.linspace(0, len(selected) - 1, min(6, len(selected))).astype(int)]
        folder = args.output / "videos" / f"video_{index + 1:03d}"; folder.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6), squeeze=False, constrained_layout=True)
        for column, frame in enumerate(sample):
            axes[0, column].imshow(rgb[frame]); axes[1, column].imshow(overlay[frame])
            axes[0, column].set_title(f"source frame {int(item['frame_indices'][frame])}\n{float(item['frame_indices'][frame]) / item['source_fps']:.2f}s")
            for ax in axes[:, column]: ax.axis("off")
        prediction = int(output.video_logits[0] >= 0)
        fig.suptitle(f"{item['video_id']} | true {item['label']} | predicted {prediction} | video logit {float(output.video_logits[0]):.3f}\nWeakly supervised spill evidence: blue=0, red=1 on a fixed probability scale")
        fig.savefig(folder / "spill_localization.png", dpi=120); plt.close(fig)
        render_mp4(np.concatenate((rgb[selected], overlay[selected]), axis=2), folder / "spill_localization.mp4")
        np.savez_compressed(folder / "spill_localization.npz", probability=probability.numpy(),
                            sampled_frame_indices=item["frame_indices"].numpy(), tubelet_valid=cache["valid"][index].numpy())
        row = {"video_id": item["video_id"], "category": item["category"], "true_label": item["label"],
               "predicted_label": prediction, "video_logit": float(output.video_logits[0]),
               "maximum_patch_probability": float(probability.max()),
               "mean_valid_patch_probability": float(probability[cache["valid"][index]].mean()),
               "directory": str(folder.relative_to(args.output))}
        (folder / "inference.json").write_text(json.dumps(row, indent=2) + "\n"); rows.append(row)
        print(f"rendered {index + 1:02d}/{len(dataset)} {item['video_id']}", flush=True)
    (args.output / "inference_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    cards = ["<!doctype html><meta charset='utf-8'><title>Spill localization</title>",
             "<h1>Weakly supervised spill localization</h1><p>Blue=low and red=high spill evidence on a fixed 0–1 scale. "
             "These maps come from a class-trained spatial MIL head, not generic transformer attention. Pixel masks are needed to measure localization accuracy.</p>"]
    for row in rows:
        directory = row["directory"]
        cards.append(f"<h2>{row['video_id']} — true {row['true_label']}, predicted {row['predicted_label']}</h2>"
                     f"<img style='max-width:100%' src='{directory}/spill_localization.png'>"
                     f"<p><a href='{directory}/spill_localization.mp4'>Overlay video</a> · "
                     f"<a href='{directory}/inference.json'>Scores</a></p>")
    (args.output / "index.html").write_text("\n".join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path,
                        help="directory used to resolve video_path entries (default: manifest directory)")
    parser.add_argument("--token-cache", type=Path, default=ROOT / "checkpoints/spatial_video_tokens.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/spill_localization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-fraction", type=float, default=.05)
    parser.add_argument("--sparsity-weight", type=float, default=.05)
    parser.add_argument("--smoothness-weight", type=float, default=.02)
    parser.add_argument("--token-layer", choices=("patch_embed", "last_encoder_layer"), default="patch_embed")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-only", action="store_true",
                        help="extract and validate the spatial-token cache, then stop")
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2 or args.batch_size < 1: parser.error("epochs, folds and batch size must be positive")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if args.rebuild_cache or not args.token_cache.exists(): extract(args)
    cache = torch.load(args.token_cache, map_location="cpu", weights_only=True); validate_cache(cache, args.manifest)
    if cache["token_layer"] != args.token_layer:
        raise ValueError(f"Requested {args.token_layer}, but cache contains {cache['token_layer']}; use a new cache or --rebuild-cache")
    if args.encode_only:
        print(f"validated spatial-token cache: {args.token_cache}")
        return
    model = train(args, cache); visualize(args, cache, model)


if __name__ == "__main__": main()
