#!/usr/bin/env python3
"""Coarse real-input occlusion of a fixed episode decision, bypassing gradients."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.online_trajectory import OnlinePrefixEncoder, collect_prefix_features
from attribute_experiment81 import load_case, margins_for_group, evaluation_groups
from control_experiment81 import transform_bank
from prepare_experiment81 import normalize_bank, digest


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", choices=("frozen", "full"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include", nargs="*", default=["cold_folderImage56"])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(); torch.set_num_threads(4)
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the allocated GPU")
    device = torch.device("cuda", 0)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    encoder, model, reps, proto, settings = load_case(a.root, a.case, device)
    transform = OfficialVJEPA2Transform(settings["teacher_metadata"]["encoder_dir"], preserve_field_of_view=True)
    wrapper = OnlinePrefixEncoder(encoder)
    groups = evaluation_groups(reps, settings)
    sources = sorted(set(reps["selection_indices"].tolist()) | {i for i, name in enumerate(c["ids"]) if name in a.include})
    a.output.mkdir(parents=True, exist_ok=a.resume)
    checkpoint_sha = digest(a.root / a.case / "checkpoint.pt")
    d = proto[1] - proto[0]
    z = reps["embeddings"][:, 0].to(device)
    margin = z @ d
    tangent = (d[None] - margin[:, None] * z).norm(dim=1)
    summaries = []
    for i in sources:
        output = a.output / f"{c['ids'][i]}.pt"
        if a.resume and output.exists():
            old = torch.load(output, map_location="cpu", weights_only=False)
            if old["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("Cannot resume occlusion with different weights")
            summaries.append(old["summary"])
            continue
        ids = next(g for g in groups if i in g)
        local = int((ids == i).nonzero()[0])
        pixels = torch.stack([normalize_bank(c["pixels"][int(j), :2], c["normalization"]) for j in ids])

        def forward(px):
            feature = collect_prefix_features(wrapper, px, c["valid"][ids], device, chunk_sources=len(ids))
            scores = margins_for_group(model, reps, ids, feature, proto, device)
            return float(scores[local]), feature[local].cpu()

        baseline, feature = forward(pixels)
        error = float((feature - reps["features"][i]).abs().max())
        if error > 1e-6 or abs(baseline - float(margin[i])) > 1e-5:
            raise RuntimeError(f"Occlusion baseline replay failed: {c['ids'][i]}, {error}")
        rgb = c["rgb"][i]
        h, w = rgb.shape[-2:]
        if h % 6 or w % 8:
            raise ValueError("The predeclared 6x8 grid does not divide this source")
        fills = {"mean": rgb.float().mean((-2, -1), keepdim=True).expand_as(rgb).round().byte(),
                 "blur33": F.avg_pool2d(F.pad(rgb.float(), (16, 16, 16, 16), mode="reflect"), 33, 1).round().clamp(0, 255).byte()}
        maps = {}
        for name, fill in fills.items():
            drops = torch.zeros(6, 8)
            for row in range(6):
                for column in range(8):
                    ys, xs = slice(row * h // 6, (row + 1) * h // 6), slice(column * w // 8, (column + 1) * w // 8)
                    changed = rgb.clone(); changed[..., ys, xs] = fill[..., ys, xs]
                    px = pixels.clone()
                    px[local] = transform_bank(changed, [], transform, include_augmentations=False)
                    score, _ = forward(px)
                    drops[row, column] = baseline - score
            maps[name] = drops
        summary = dict(id=c["ids"][i], label=int(c["labels"][i]),
            split="selection" if i in reps["selection_indices"] else "training_diagnostic",
            baseline_margin=baseline, replay_feature_max_error=error,
            mean_fill_max_drop=float(maps["mean"].max()), blur_fill_max_drop=float(maps["blur33"].max()),
            score_direction_tangent_norm=float(tangent[i]),
            score_direction_alignment=float(margin[i] / d.norm()),
            fill_map_cosine=float(F.cosine_similarity(maps["mean"].flatten(), maps["blur33"].flatten(), dim=0)))
        payload = dict(maps=maps, summary=summary, checkpoint_sha256=checkpoint_sha, grid=[6, 8],
            scope="Episode-wide fixed region removal over all sampled frames; not framewise attention",
            target="Actual fixed-prototype spill-minus-clean cosine margin", localization_validated=False)
        torch.save(payload, output); summaries.append(summary)
        (a.output / "progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    (a.output / "completed.json").write_text(json.dumps(dict(case=a.case, sources=summaries,
        checkpoint_sha256=checkpoint_sha, native_regions=48, fills=["mean", "blur33"],
        display_policy="Signed decision changes; any rendering must declare a fixed common margin scale",
        localization_validated=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
