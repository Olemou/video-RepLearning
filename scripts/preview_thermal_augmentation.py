#!/usr/bin/env python3
"""Display exact training thermal views after full-frame checkpoint preprocessing."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vision_model.data.transforms import ThermalAugmentor, OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.data.sampling import TemporalSamplingConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    base = OfficialVJEPA2Transform()
    augmentor = ThermalAugmentor()
    transform = OfficialVJEPA2Transform(augmentation=augmentor)
    dataset = VideoManifestDataset(ROOT / "reports/experiments/unique_7_clean_7_spill/manifest.jsonl",
        dataset_root=ROOT / "dataset", sampling=TemporalSamplingConfig(clip_length=64, target_fps=4,
          training=False, short_video_policy="repeat_last"))
    args.output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(base.normalization["mean"])[None, :, None, None]
    std = torch.tensor(base.normalization["std"])[None, :, None, None]
    def inverse(pixels):
        return ((pixels * std + mean) * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    metadata = []
    for row, video_id in enumerate(("clean_videoFrame99", "cold_scene3", "warm_scene_1_hot")):
        i = [r.video_id for r in dataset.records].index(video_id)
        item = dataset[i]
        t = int(item["temporal_valid_mask"].sum()) // 2
        raw = item["video"][:2*t]
        views = [inverse(base(raw))]
        descriptions = ["Identity / full frame"]
        for j in range(3):
            seed = 239 + i * 1000 + j
            parameters = augmentor.sample(torch.Generator().manual_seed(seed))
            views.append(inverse(transform(raw, generator=torch.Generator().manual_seed(seed))))
            descriptions.append(f"Flip={parameters['horizontal_flip']}\nGain={parameters['brightness']:.3f}, contrast={parameters['contrast']:.3f}")
            metadata.append(dict(video_id=video_id, seed=seed, **parameters))
        for col in range(4):
            axes[row, col].imshow(views[col][-1])
            axes[row, col].set_title(f"{video_id}\n{descriptions[col]}", fontsize=9)
            axes[row, col].axis("off")
        if video_id == "warm_scene_1_hot":
            writer = cv2.VideoWriter(str(args.output / "thermal_views.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    4, (256 * 4, 256))
            for frame in range(2*t):
                writer.write(cv2.cvtColor(np.concatenate([v[frame] for v in views], axis=1), cv2.COLOR_RGB2BGR))
            writer.release()
    fig.suptitle("Actual training views after preprocessing: entire floor and frame preserved", fontsize=13)
    fig.savefig(args.output / "thermal_augmentation_preview.png", dpi=160)
    plt.close(fig)
    (args.output / "thermal_augmentation_preview.json").write_text(json.dumps(dict(policy=augmentor.policy, views=metadata), indent=2))
    print(f"Saved preview: {args.output}", flush=True)


if __name__ == "__main__":
    main()
