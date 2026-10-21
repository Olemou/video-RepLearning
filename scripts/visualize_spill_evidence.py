#!/usr/bin/env python3
"""Create a spatial spill-evidence video for the trained trajectory classifier.

The classifier pools V-JEPA patch tokens spatially. This is an internal
leave-one-token-out intervention, not pixel-level causal localization or raw
Transformer attention. Globally contextualized tokens may encode other regions.
Each heatmap value is the decrease in the spill-vs-clean prototype margin after
replacing one patch token by that tubelet's spatial mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel, standardized


def semantic_margin(model, features, valid, timestamps, spill_proto, clean_proto):
    out = model(features, valid, timestamps)
    return out["embedding"] @ (spill_proto - clean_proto), out


def reverse_letterbox(heat, height, width):
    """Map a 16x16 checkpoint-grid heatmap back onto the source aspect ratio."""
    scale = min(256 / height, 256 / width)
    resized_h, resized_w = max(1, min(256, round(height * scale))), max(1, min(256, round(width * scale)))
    top, left = (256 - resized_h) // 2, (256 - resized_w) // 2
    # Preserve raw token cells. Cubic interpolation invents round-looking blobs
    # and can smear padding-token contributions across the content boundary.
    square = cv2.resize(heat.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
    content = square[top:top + resized_h, left:left + resized_w]
    return cv2.resize(content, (width, height), interpolation=cv2.INTER_NEAREST)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video-id", default="warm_scene_1_hot")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-fraction", type=float, default=.10)
    p.add_argument("--sparse-quantile", type=float, default=.95,
                   help="Keep only positive evidence above this global clip quantile")
    args = p.parse_args()
    if not 0 < args.top_fraction < 1:
        p.error("--top-fraction must be between zero and one")
    if not 0 < args.sparse_quantile < 1:
        p.error("--sparse-quantile must be between zero and one")
    torch.manual_seed(239)
    torch.set_num_threads(4)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.video_id not in cache["ids"]:
        p.error(f"Unknown video ID {args.video_id!r}")
    target_index = cache["ids"].index(args.video_id)
    model = TrajectoryModel(saved["width"], saved["hidden"], saved["embedding_dim"])
    model.load_state_dict(saved["model"])
    model.eval()

    # Leave the displayed source out of both prototypes so its score is not
    # compared to a prototype that contains its own embeddings.
    n, _, time, _ = cache["features"].shape
    mask2 = cache["valid"][:, None].expand(n, 2, time).reshape(n * 2, time)
    stamp2 = cache["timestamps"][:, None].expand(n, 2, time).reshape(n * 2, time)
    with torch.no_grad():
        embeddings = model(cache["features"].reshape(n * 2, time, -1), mask2, stamp2)["embedding"]
    source = torch.arange(n).repeat_interleave(2)
    repeated_labels = cache["labels"].repeat_interleave(2)
    keep = source != target_index
    clean_proto = F.normalize(embeddings[keep & (repeated_labels == 0)].mean(0), dim=0)
    spill_proto = F.normalize(embeddings[keep & (repeated_labels == 1)].mean(0), dim=0)

    transform = OfficialVJEPA2Transform(cache["model_id"], preserve_field_of_view=True)
    dataset = VideoManifestDataset(cache["manifest"], dataset_root=cache["dataset_root"],
        sampling=TemporalSamplingConfig(clip_length=64, target_fps=4, training=False,
                                         short_video_policy="repeat_last"), transform=transform)
    dataset_index = [r.video_id for r in dataset.records].index(args.video_id)
    item = dataset[dataset_index]
    tubelets = int(item["temporal_valid_mask"].sum()) // 2
    pixels = item["pixel_values_videos"][:2 * tubelets]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = load_pretrained_encoder(cache["model_id"], local_files_only=True)
    teacher.freeze_encoder(True)
    encoder = teacher.encoder.to(device=device, dtype=dtype).eval()
    dense = []
    for end in range(1, tubelets + 1):
        clip = pixels[:2 * end].permute(1, 0, 2, 3)[None].to(device=device, dtype=dtype)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype,
                                                    enabled=device.type == "cuda"):
            tokens = encoder(clip)
        dense.append(tokens[0, -256:].float().cpu())
        print(f"dense prefix {end}/{tubelets}", flush=True)
    encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dense = torch.stack(dense)  # [T,256,D]
    base = dense.mean(1)
    valid = torch.ones(1, tubelets, dtype=torch.bool)
    timestamps = item["frame_indices"][:2 * tubelets][1::2].float()[None] / float(item["source_fps"])
    with torch.no_grad():
        original_margin, output = semantic_margin(model, base[None], valid, timestamps,
                                                   spill_proto, clean_proto)
        state = standardized(output["states"])
        temporal_score = model.event_score(state).squeeze(-1).softmax(1)[0]

    # Create all single-patch counterfactuals. Replacing a token with its local
    # spatial mean changes only the exact mean input consumed by the classifier.
    perturbations, coordinates = [], []
    for t in range(tubelets):
        for patch in range(256):
            modified = base.clone()
            modified[t] += (base[t] - dense[t, patch]) / 256
            perturbations.append(modified)
            coordinates.append((t, patch))
    occluded_margins = []
    with torch.no_grad():
        for start in range(0, len(perturbations), args.batch_size):
            batch = torch.stack(perturbations[start:start + args.batch_size])
            b = len(batch)
            margins, _ = semantic_margin(model, batch, valid.expand(b, -1),
                timestamps.expand(b, -1), spill_proto, clean_proto)
            occluded_margins.append(margins)
    attribution = (original_margin - torch.cat(occluded_margins)).reshape(tubelets, 16, 16)

    # Internal-feature deletion check only. This does not remove input pixels.
    flat = attribution.flatten()
    delete_count = max(1, round(len(flat) * args.top_fraction))
    top_indices = flat.topk(delete_count).indices
    def jointly_delete(indices):
        modified = base.clone()
        for flat_index in indices.tolist():
            t, patch = divmod(flat_index, 256)
            modified[t] += (base[t] - dense[t, patch]) / 256
        with torch.no_grad():
            value, _ = semantic_margin(model, modified[None], valid, timestamps,
                                        spill_proto, clean_proto)
        return float(value)
    top_margin = jointly_delete(top_indices)
    generator = torch.Generator().manual_seed(239)
    random_margins = [jointly_delete(torch.randperm(len(flat), generator=generator)[:delete_count])
                      for _ in range(50)]

    raw_frames = item["video"][:2 * tubelets].permute(0, 2, 3, 1).numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / f"{args.video_id}_sparse_spill_evidence.mp4"
    height, width = raw_frames.shape[1:3]
    panel_width = width * 3
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (panel_width, height))
    positive = attribution.clamp_min(0).numpy()
    # One threshold over the complete clip is essential: per-frame normalization
    # would manufacture a hot region even when a frame has no strong evidence.
    positive_values = positive[positive > 0]
    threshold = (float(np.quantile(positive_values, args.sparse_quantile))
                 if len(positive_values) else float("inf"))
    upper = (float(np.quantile(positive_values, .995))
             if len(positive_values) else threshold + 1.)
    scale = max(upper - threshold, 1e-12)
    sparse = np.clip((positive - threshold) / scale, 0, 1)
    retained_per_tubelet = (positive > threshold).sum(axis=(1, 2)).astype(int)
    rendered = []
    for frame_number, rgb in enumerate(raw_frames):
        t = frame_number // 2
        heat = np.clip(reverse_letterbox(sparse[t], height, width), 0, 1)
        # Blue means below this clip's display threshold, NOT no-spill probability.
        color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        alpha = (.75 * heat)[..., None]
        overlay = np.uint8(np.clip((1 - alpha) * bgr + alpha * color, 0, 255))
        panel = np.concatenate((bgr, color, overlay), axis=1)
        cv2.putText(panel, "original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        cv2.putText(panel, "token contribution", (width + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
        cv2.putText(panel, "overlay", (2 * width + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        line = (f"{args.video_id} | tubelet {t+1}/{tubelets} | "
                f"hot patches {retained_per_tubelet[t]}/256")
        cv2.putText(panel, line, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,255,255), 2)
        writer.write(panel)
        rendered.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    writer.release()
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("Failed to write the evidence video")

    chosen = np.linspace(0, len(rendered) - 1, 5).round().astype(int)
    fig, axes = plt.subplots(5, 1, figsize=(14, max(6, 5 * height / panel_width * 14)), constrained_layout=True)
    for ax, index in zip(axes, chosen):
        ax.imshow(rendered[index])
        ax.set_title(f"t={float(timestamps[0, index // 2]):.2f}s, temporal weight={float(temporal_score[index // 2]):.3f}")
        ax.axis("off")
    fig.savefig(args.output / f"{args.video_id}_sparse_contact_sheet.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.bar(np.arange(tubelets) + 1, temporal_score.numpy())
    ax.set(xlabel="Causal tubelet", ylabel="Event-pooling weight",
           title=f"Temporal attention: {args.video_id}")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(args.output / f"{args.video_id}_temporal_attention.png", dpi=160)
    plt.close(fig)

    random_mean, random_std = float(np.mean(random_margins)), float(np.std(random_margins))
    result = {
        "video_id": args.video_id,
        "true_label": "spill" if int(cache["labels"][target_index]) == 1 else "clean",
        "method": "contextual_token_mean_replacement",
        "rendering": "nearest-neighbor cells; clip-relative positive quantile; not probability",
        "prototype_policy": "leave_target_source_out; both views of remaining sources",
        "original_spill_minus_clean_margin": float(original_margin),
        "predicted_label": "spill" if float(original_margin) > 0 else "clean",
        "positive_patch_fraction": float((flat > 0).float().mean()),
        "sparse_global_quantile": args.sparse_quantile,
        "sparse_attribution_threshold": threshold,
        "sparse_retained_patch_fraction": float((positive > threshold).mean()),
        "sparse_retained_patches_per_tubelet": retained_per_tubelet.tolist(),
        "top_deletion_fraction": args.top_fraction, "top_deletion_count": delete_count,
        "top_deletion_margin": top_margin,
        "top_deletion_margin_drop": float(original_margin) - top_margin,
        "random_deletion_margin_mean": random_mean, "random_deletion_margin_std": random_std,
        "random_deletion_mean_drop": float(original_margin) - random_mean,
        "temporal_event_weights": temporal_score.tolist(),
        "limitations": [
            "This is patch-occlusion attribution, not a learned spatial-attention matrix.",
            "Contextual token removal does not establish causality of the pixels at that location.",
            "Blue means below the per-clip display threshold, not verified no-spill content.",
            "No pixel-level spill mask exists, so localization accuracy cannot be measured.",
            "The displayed video was part of training; this does not test unseen-video generalization."
        ],
        "artifacts": {"video": str(video_path),
            "contact_sheet": str(args.output / f"{args.video_id}_sparse_contact_sheet.png"),
            "temporal_attention": str(args.output / f"{args.video_id}_temporal_attention.png")}
    }
    (args.output / f"{args.video_id}_diagnostics.json").write_text(json.dumps(result, indent=2))
    torch.save({"attribution": attribution, "temporal_event_weights": temporal_score,
                "dense_tokens": dense, "timestamps": timestamps, "result": result},
               args.output / f"{args.video_id}_attribution.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
