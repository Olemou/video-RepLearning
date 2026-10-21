#!/usr/bin/env python3
"""Local spill evidence from distance to clean pretrained patch tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vision_model.data.config import ThermalAugConfig
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from train_spill_localizer import local_model_source, validate_cache


def render_mp4(frames, path):
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "4", "-i", "pipe:0",
                             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                            input=(frames * 255).round().astype(np.uint8).tobytes(), capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--token-cache", type=Path, default=ROOT / "checkpoints/local_spatial_video_tokens.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/spill_patch_anomaly")
    parser.add_argument("--references-per-clean-video", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.token_cache, map_location="cpu", weights_only=True); validate_cache(cache, args.manifest)
    if cache["token_layer"] != "patch_embed": raise ValueError("Patch anomaly requires local patch-embedding tokens")
    generator = torch.Generator().manual_seed(239)
    references, owners = [], []
    for index in torch.where(cache["labels"] == 0)[0].tolist():
        values = cache["tokens"][index][cache["valid"][index]].reshape(-1, 1024).float()
        take = min(args.references_per_clean_video, len(values))
        choice = torch.randperm(len(values), generator=generator)[:take]
        references.append(F.normalize(values[choice], dim=-1)); owners += [index] * take
    reference = torch.cat(references).to(args.device)
    owners = torch.tensor(owners, device=args.device)
    raw_maps, video_scores = [], []
    for index in range(len(cache["labels"])):
        valid = cache["valid"][index]
        values = cache["tokens"][index][valid].reshape(-1, 1024).float().to(args.device)
        values = F.normalize(values, dim=-1)
        best = []
        allowed = owners != index if cache["labels"][index] == 0 else torch.ones_like(owners, dtype=torch.bool)
        keys = reference[allowed]
        for chunk in values.split(512): best.append(1 - (chunk @ keys.T).max(1).values)
        score = torch.cat(best).reshape(int(valid.sum()), 16, 16)
        score = F.avg_pool2d(score[:, None], 3, stride=1, padding=1)[:, 0]
        full = torch.zeros(32, 16, 16); full[valid] = score.cpu(); raw_maps.append(full)
        count = max(1, int(score.numel() * .05)); video_scores.append(float(score.flatten().topk(count).values.mean()))
    # Fixed calibration from leave-video-out clean patches. Values at/below the
    # 95th clean percentile stay blue; the 99.9th percentile maps to red.
    clean_values = torch.cat([raw_maps[i][cache["valid"][i]].flatten() for i in torch.where(cache["labels"] == 0)[0]])
    low, high = (float(torch.quantile(clean_values, q)) for q in (.95, .999))
    probability = [((value - low) / max(high - low, 1e-12)).clamp(0, 1) for value in raw_maps]
    labels = cache["labels"].numpy(); scores = np.array(video_scores)
    from sklearn.metrics import roc_auc_score
    report = {"method": "nearest clean local patch cosine anomaly", "clean_calibration": {"q95": low, "q999": high},
              "video_roc_auc": float(roc_auc_score(labels, scores)),
              "mean_clean_video_score": float(scores[labels == 0].mean()),
              "mean_spill_video_score": float(scores[labels == 1].mean()),
              "limitations": "No pixel masks: localization cannot be scored. Reference patches are from all 17 clean training videos."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

    transform = OfficialVJEPA2Transform(local_model_source(cache["model_id"]), thermal_config=ThermalAugConfig(image_size=256))
    dataset = VideoManifestDataset(args.manifest, sampling=TemporalSamplingConfig(**cache["sampling"]), transform=transform)
    rows = []
    for index, item in enumerate(dataset):
        pixels = item["pixel_values_videos"].float(); mean = torch.tensor(cache["normalization"]["mean"])[None, :, None, None]
        std = torch.tensor(cache["normalization"]["std"])[None, :, None, None]
        rgb = (pixels * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        heat = F.interpolate(probability[index][:, None], (256, 256), mode="bilinear", align_corners=False)[:, 0]
        heat = heat.repeat_interleave(2, 0).numpy(); selected = np.flatnonzero(cache["valid"][index].repeat_interleave(2).numpy())
        overlay = (.45 * rgb + .55 * plt.get_cmap("coolwarm")(heat)[..., :3]).clip(0, 1)
        sample = selected[np.linspace(0, len(selected)-1, min(6, len(selected))).astype(int)]
        folder = args.output / "videos" / f"video_{index+1:03d}"; folder.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, len(sample), figsize=(3*len(sample), 6), squeeze=False, constrained_layout=True)
        for column, frame in enumerate(sample):
            axes[0,column].imshow(rgb[frame]); axes[1,column].imshow(overlay[frame])
            axes[0,column].set_title(f"source frame {int(item['frame_indices'][frame])}\n{float(item['frame_indices'][frame])/item['source_fps']:.2f}s")
            for ax in axes[:,column]: ax.axis("off")
        fig.suptitle(f"{item['video_id']} | class {item['label']} | anomaly {video_scores[index]:.4f}\nLocal distance from clean patches: blue≤clean q95, red≥clean q99.9")
        fig.savefig(folder / "spill_anomaly.png", dpi=120); plt.close(fig)
        render_mp4(np.concatenate((rgb[selected], overlay[selected]), 2), folder / "spill_anomaly.mp4")
        np.savez_compressed(folder / "spill_anomaly.npz", raw_anomaly=raw_maps[index].numpy(),
                            calibrated_heat=probability[index].numpy(), tubelet_valid=cache["valid"][index].numpy())
        rows.append({"video_id": item["video_id"], "label": item["label"], "video_anomaly": video_scores[index],
                     "directory": str(folder.relative_to(args.output))})
        print(f"rendered anomaly {index+1:02d}/{len(dataset)} {item['video_id']}", flush=True)
    (args.output / "inference_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    page = ["<!doctype html><meta charset='utf-8'><title>Spill patch anomaly</title>",
            "<h1>Local distance from clean patches</h1><p>Fixed clean-derived scale. Red means the local pretrained patch pattern is rare among clean videos; it is not pixel-mask supervision.</p>"]
    for row in rows:
        page.append(f"<h2>{row['video_id']} — class {row['label']}</h2><img style='max-width:100%' src='{row['directory']}/spill_anomaly.png'><p><a href='{row['directory']}/spill_anomaly.mp4'>video</a></p>")
    (args.output / "index.html").write_text("\n".join(page))


if __name__ == "__main__": main()
