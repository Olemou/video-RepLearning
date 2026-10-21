#!/usr/bin/env python3
"""Audit contextual-token maps using actual input occlusion and static controls.

No training or threshold tuning. Broad floor boxes are diagnostic proxies, not
annotated spill masks. Input ablations recompute every causal encoder prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vision_model.training.trajectory import TrajectoryModel
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.video_dataset import VideoManifestDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "reports/trajectory_100")
    p.add_argument("--video-id", default="warm_scene_1_hot")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(239)
    cache = torch.load(args.run / "causal_features.pt", map_location="cpu", weights_only=False)
    ckpt = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = TrajectoryModel(ckpt["width"], ckpt["hidden"], ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids, labels = cache["ids"], cache["labels"]
    index = ids.index(args.video_id)
    n, views, time, width = cache["features"].shape
    x = cache["features"].reshape(n * views, time, width)
    valid = cache["valid"][:, None].expand(n, views, time).reshape(n * views, time)
    stamps = cache["timestamps"][:, None].expand(n, views, time).reshape(n * views, time)
    with torch.no_grad():
        all_z = model(x, valid, stamps)["embedding"].reshape(n, views, -1)
    # Excludes a source only from its prototype, not from representation training.
    def prototypes(exclude):
        keep = torch.arange(n) != exclude
        return torch.stack([F.normalize(all_z[keep & (labels == c)].mean((0, 1)), dim=0)
                            for c in (0, 1)])
    proto = prototypes(index)
    feature_controls = {}
    for kind in ("original", "first_state_only", "first_state_repeated", "last_state_repeated", "reversed_states"):
        fx = cache["features"][:, 0].clone()
        mask, clock = cache["valid"].clone(), cache["timestamps"].clone()
        for i in range(n):
            t = int(mask[i].sum())
            if kind == "first_state_only":
                mask[i, 1:] = False
            elif kind == "first_state_repeated":
                fx[i, :t] = fx[i, 0].clone()
            elif kind == "last_state_repeated":
                fx[i, :t] = fx[i, t-1].clone()
            elif kind == "reversed_states":
                fx[i, :t] = fx[i, :t].flip(0)
        with torch.no_grad():
            z = model(fx, mask, clock)["embedding"]
        rows = []
        for i in range(n):
            scores = z[i] @ prototypes(i).T
            rows.append(dict(id=ids[i], true_label=int(labels[i]), predicted_label=int(scores.argmax()),
                             margin=float(scores[1] - scores[0])))
        feature_controls[kind] = dict(accuracy=sum(r["true_label"] == r["predicted_label"] for r in rows) / n,
                                      selected_video=rows[index], videos=rows)

    transform = OfficialVJEPA2Transform(cache["model_id"], preserve_field_of_view=True)
    dataset = VideoManifestDataset(cache["manifest"], dataset_root=cache["dataset_root"],
        sampling=TemporalSamplingConfig(clip_length=64, target_fps=4, training=False,
                                         short_video_policy="repeat_last"), transform=transform)
    di = [r.video_id for r in dataset.records].index(args.video_id)
    item = dataset[di]
    count = int(cache["valid"][index].sum())
    pixels = item["pixel_values_videos"][:2 * count]
    source_h, source_w = item["video"].shape[-2:]
    resize = min(256 / source_h, 256 / source_w)
    h, w = round(source_h * resize), round(source_w * resize)
    top, left = (256 - h) // 2, (256 - w) // 2
    bottom, right = top + h, left + w
    content = torch.zeros(256, 256, dtype=torch.bool)
    content[top:bottom, left:right] = True
    floor = torch.zeros_like(content)
    floor_start = top + round(.65 * h)
    floor[floor_start:bottom, left:right] = True
    border = content.clone()
    bh, bw = round(.1 * h), round(.1 * w)
    border[top + bh:bottom - bh, left + bw:right - bw] = False
    masks = {"remove_floor_bottom35": floor, "remove_upper65": content & ~floor,
             "remove_content_border10": border, "keep_floor_only": content & ~floor,
             "uniform_content": content}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = load_pretrained_encoder(cache["model_id"], local_files_only=True)
    teacher.freeze_encoder(True)
    encoder = teacher.encoder.to(device=device, dtype=dtype).eval()
    times = cache["timestamps"][index:index + 1, :count]
    ones = torch.ones(1, count, dtype=torch.bool)

    def encode(video):
        features = []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype,
                                                   enabled=device.type == "cuda"):
            for end in range(1, count + 1):
                clip = video[:2 * end].permute(1, 0, 2, 3)[None].to(device=device, dtype=dtype)
                tokens = encoder(clip)
                features.append(tokens[0, -256:].mean(0).float().cpu())
        return torch.stack(features)

    def evaluate(features):
        with torch.no_grad():
            z = model(features[None], ones, times)["embedding"][0]
            scores = z @ proto.T
        return dict(margin=float(scores[1] - scores[0]), predicted_label=int(scores.argmax()),
                    cosine_clean=float(scores[0]), cosine_spill=float(scores[1]))

    original_features = encode(pixels)
    conditions = {"original": evaluate(original_features)}
    replay_max_error = float((original_features - cache["features"][index, 0, :count]).abs().max())
    for name in ("remove_floor_bottom35", "remove_upper65", "remove_content_border10", "uniform_content"):
        altered = pixels.clone()
        # Zero normalized pixels equal the checkpoint mean; padding stays intact.
        altered[:, :, masks[name]] = 0
        conditions[name] = evaluate(encode(altered))
        print(f"{name}: {conditions[name]}", flush=True)
    conditions["keep_floor_only"] = conditions["remove_upper65"].copy()
    for name, altered in (("repeat_first_input_frame", pixels[:1].expand_as(pixels)),
                           ("reverse_input_frames", pixels.flip(0))):
        conditions[name] = evaluate(encode(altered))
        print(f"{name}: {conditions[name]}", flush=True)

    # True spatial input occlusion: 16 disjoint source-content cells, kept
    # consistent through time and recomputed through all encoder prefixes.
    grid = np.zeros((4, 4), dtype=float)
    for row in range(4):
        for col in range(4):
            ys, ye = top + round(row * h / 4), top + round((row + 1) * h / 4)
            xs, xe = left + round(col * w / 4), left + round((col + 1) * w / 4)
            altered = pixels.clone()
            altered[:, :, ys:ye, xs:xe] = 0
            result = evaluate(encode(altered))
            name = f"cell_{row}_{col}"
            conditions[name] = result
            grid[row, col] = conditions["original"]["margin"] - result["margin"]
        print(f"input occlusion grid row {row+1}/4 complete", flush=True)
    for condition in conditions.values():
        condition["margin_drop"] = conditions["original"]["margin"] - condition["margin"]

    old = torch.load(args.run / "attention_sparse" / f"{args.video_id}_attribution.pt",
                     map_location="cpu", weights_only=False)
    attr = old["attribution"]
    # Fractional geometric coverage for tokens intersecting letterbox boundaries.
    support = F.avg_pool2d(content.float()[None, None], 16, 16)[0, 0]
    positive = attr.clamp_min(0)
    cutoff = old["result"]["sparse_attribution_threshold"]
    retained = attr > cutoff
    spatial_audit = dict(content_bounds_xyxy=[left, top, right, bottom],
        letterbox_pixel_fraction=float((~content).float().mean()),
        pooling_policy="all 256 contextual tokens including letterbox locations",
        positive_token_mass_on_padding_fraction=float((positive * (1-support)).sum() / positive.sum().clamp_min(1e-12)),
        retained_token_fraction_on_padding=float((retained * (1-support)).sum() / retained.sum().clamp_min(1)),
        retained_tokens_count=int(retained.sum()))
    out = args.run / "localization_audit"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(video_id=args.video_id, protocol="trained-model diagnostic; no held-out representation training",
        checkpoint_sha256=hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest(),
        cache_replay_max_absolute_error=replay_max_error, feature_controls=feature_controls,
        input_ablation=conditions, actual_input_occlusion_grid=grid.tolist(), spatial_audit=spatial_audit,
        floor_roi="Broad bottom 35% of source image, unannotated proxy; includes feet and floor, not a spill mask",
        input_baseline="Zero after checkpoint normalization, i.e. channel-mean color",
        caveats=["Masking can introduce out-of-distribution edges; results depend on the baseline.",
                 "A single video's diagnostics do not establish localization accuracy or generalization.",
                 "A percentile threshold always retains positive features when any exist; blue is not a no-spill probability.",
                 "Contextual output-token ablation does not establish pixel-region causality."])
    (out / "report.json").write_text(json.dumps(report, indent=2))

    frame = item["video"][2 * count - 1].permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].imshow(frame)
    axes[0].axhline(source_h * .65, color="yellow", linestyle="--", label="Bottom 35% proxy")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Source frame: broad floor proxy")
    axes[0].axis("off")
    visible = attr[:, 2:14] if (top, bottom) == (32, 224) else attr
    mean_token = visible.mean(0).numpy()
    vmax = max(float(np.abs(mean_token).max()), 1e-12)
    im = axes[1].imshow(mean_token, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[1].set_title("Previous output-token evidence\nTime mean; raw cells, no smoothing")
    fig.colorbar(im, ax=axes[1], label="Token-level margin drop", shrink=.65)
    vmax = max(float(np.abs(grid).max()), 1e-12)
    im = axes[2].imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for r in range(4):
        for c in range(4):
            axes[2].text(c, r, f"{grid[r,c]:+.3f}", ha="center", va="center", fontsize=9,
                         color="black" if abs(grid[r,c]) < .65*vmax else "white")
    axes[2].set_title("Actual input-region removal\n4x4 cells; all causal prefixes re-encoded")
    fig.colorbar(im, ax=axes[2], label="Input-level margin drop", shrink=.65)
    fig.suptitle("Positive / red = removing region reduces spill score; negative / blue = increases score")
    fig.savefig(out / "token_vs_input_evidence.png", dpi=170)
    plt.close(fig)
    print(json.dumps(dict(output=str(out), input_ablation=conditions, spatial_audit=spatial_audit,
                         feature_control_accuracy={k:v["accuracy"] for k,v in feature_controls.items()}), indent=2), flush=True)


if __name__ == "__main__":
    main()
