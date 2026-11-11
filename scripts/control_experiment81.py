#!/usr/bin/env python3
"""Fresh-input, fixed-prototype margin removal controls; no localization labels."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_model.data.transforms import OfficialVJEPA2Transform, ThermalAugmentor
from vision_model.interpretability.map_geometry import inverse_letterbox_map, letterbox_bounds
from vision_model.training.distributed import initialize_distributed
from vision_model.training.online_trajectory import OnlinePrefixEncoder, collect_prefix_features
from attribute_experiment81 import load_case, margins_for_group, evaluation_groups
from prepare_experiment81 import normalize_bank, digest


def content_cells(height, width):
    top, left, h, w = letterbox_bounds(height, width)
    rows = torch.arange(16) * 16
    cols = torch.arange(16) * 16
    return ((rows[:, None] >= top) & (rows[:, None] + 16 <= top + h)
            & (cols[None] >= left) & (cols[None] + 16 <= left + w))


def transform_bank(frames, seeds, transform, *, include_augmentations):
    views = [frames, frames.flip(-1)]
    if include_augmentations:
        augmentor = ThermalAugmentor()
        views += [augmentor(frames, generator=torch.Generator().manual_seed(int(seed))) for seed in seeds]
    official = torch.stack([transform(view) for view in views])
    mean = torch.tensor(transform.normalization["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(transform.normalization["std"]).view(1, 3, 1, 1)
    pixels = ((official * std + mean) * 255).round().clamp(0, 255).byte()
    return normalize_bank(pixels, transform.normalization)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", choices=("frozen", "full"), required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fractions", type=float, nargs="+", default=[.05, .1, .2])
    p.add_argument("--random-seeds", type=int, default=3)
    p.add_argument("--include", nargs="*", default=["cold_folderImage56"])
    a = p.parse_args(); torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    if any(not 0 < f < 1 for f in a.fractions) or a.random_seeds < 1:
        raise ValueError("Invalid removal controls")
    ctx = initialize_distributed()
    if ctx.device.type != "cuda":
        raise RuntimeError("Real GPU controls required")
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    encoder, model, reps, proto, settings = load_case(a.root, a.case, ctx.device)
    transform = OfficialVJEPA2Transform(settings["teacher_metadata"]["encoder_dir"], preserve_field_of_view=True)
    wrapper = OnlinePrefixEncoder(encoder)
    checkpoint_sha = digest(a.root / a.case / "checkpoint.pt")
    maps = {}
    for path in sorted(a.maps.glob("group_*.pt")):
        group = torch.load(path, map_location="cpu", weights_only=False)
        if group["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Map checkpoint differs")
        for j, i in enumerate(group["ids"].tolist()):
            maps[i] = group["multilayer_positive"][j]
    sources = sorted(set(reps["selection_indices"].tolist()) |
                     {i for i, name in enumerate(c["ids"]) if name in a.include})
    if any(i not in maps for i in sources):
        raise ValueError("Missing decision attribution for a control source")
    groups = evaluation_groups(reps, settings)
    if ctx.is_main:
        a.output.mkdir(parents=True, exist_ok=False)
    if ctx.world_size > 1:
        dist.barrier()
    rows = []

    def forward_input(source, override=None):
        if a.case == "frozen":
            ids = torch.tensor([source])
            px = normalize_bank(c["pixels"][source], c["normalization"]) if override is None else override
            feature = collect_prefix_features(wrapper, px[None], c["valid"][ids], ctx.device, chunk_sources=1)[:, :2]
        else:
            ids = next(g for g in groups if source in g)
            px = torch.stack([normalize_bank(c["pixels"][int(i), :2], c["normalization"]) for i in ids])
            if override is not None:
                px[int((ids == source).nonzero()[0])] = override[:2]
            feature = collect_prefix_features(wrapper, px, c["valid"][ids], ctx.device, chunk_sources=len(ids))
        scores = margins_for_group(model, reps, ids, feature, proto, ctx.device)
        j = int((ids == source).nonzero()[0])
        return float(scores[j]), feature[j].cpu()

    for i in sources[ctx.rank::ctx.world_size]:
        rgb = c["rgb"][i]
        h, w = rgb.shape[-2:]
        content = content_cells(h, w)
        allowed = content.flatten().nonzero().flatten()
        importance = maps[i].sum(0).flatten()
        ranked = allowed[importance[allowed].argsort(descending=True, stable=True)]
        baseline_score, baseline_feature = forward_input(i)
        expected_scores = margins_for_group(model, reps, torch.tensor([i]),
                                             reps["features"][i:i + 1].to(ctx.device), proto, ctx.device)
        feature_error = float((baseline_feature - reps["features"][i]).abs().max())
        margin_error = abs(baseline_score - float(expected_scores[0]))
        if feature_error > 1e-6 or margin_error > 1e-5:
            raise RuntimeError(f"Unperturbed input replay failed for {c['ids'][i]}: features {feature_error}, margin {margin_error}")
        # Also require full decode/preprocessing replay before any removal.
        fresh = transform_bank(rgb, c["augmentation_seeds"][i], transform,
                               include_augmentations=a.case == "frozen")
        original = normalize_bank(c["pixels"][i, :len(fresh)], c["normalization"])
        if not torch.equal(fresh, original):
            raise RuntimeError(f"Fresh input normalization differs: {c['ids'][i]}")
        mean_fill = rgb.float().mean((-2, -1), keepdim=True).expand_as(rgb).round().byte()
        blur_fill = F.avg_pool2d(F.pad(rgb.float(), (16, 16, 16, 16), mode="reflect"), 33, stride=1).round().clamp(0, 255).byte()
        records = []
        for fraction in a.fractions:
            count = max(1, round(len(allowed) * fraction))
            candidates = [("top", ranked[:count])]
            for seed in range(a.random_seeds):
                generator = torch.Generator().manual_seed(239 + i * 1009 + seed)
                candidates.append((f"random_{seed}", allowed[torch.randperm(len(allowed), generator=generator)[:count]]))
            for fill_name, fill in (("mean", mean_fill), ("blur33", blur_fill)):
                for kind, positions in candidates:
                    mask = torch.zeros(256, dtype=torch.bool); mask[positions] = True
                    raw_mask = inverse_letterbox_map(mask.reshape(16, 16).float(), h, w, interpolation="nearest") > .5
                    changed = torch.where(raw_mask[None, None], fill, rgb)
                    px = transform_bank(changed, c["augmentation_seeds"][i], transform,
                                        include_augmentations=a.case == "frozen")
                    score, _ = forward_input(i, px)
                    records.append(dict(fraction=fraction, content_cells_removed=count, kind=kind, fill=fill_name,
                        native_positions=positions.tolist(), original_pixel_fraction=float(raw_mask.float().mean()),
                        margin=score, margin_drop=baseline_score - score))
        row = dict(index=i, id=c["ids"][i], label=int(c["labels"][i]),
            split="selection" if i in reps["selection_indices"] else "training_diagnostic",
            baseline_margin=baseline_score, baseline_replay_feature_error=feature_error,
            baseline_replay_margin_error=margin_error, perturbations=records)
        rows.append(row)
        print(json.dumps(dict(case=a.case, id=c["ids"][i], baseline_margin=baseline_score,
                              replay_feature_error=feature_error, perturbations=len(records))), flush=True)
    gathered = [None] * ctx.world_size if ctx.is_main else None
    if ctx.world_size > 1:
        dist.gather_object(rows, gathered, dst=0)
    else:
        gathered = [rows]
    if ctx.is_main:
        rows = sorted([row for part in gathered for row in part], key=lambda r: r["index"])
        if [r["index"] for r in rows] != sources:
            raise RuntimeError("Missing/duplicated input controls")
        (a.output / "results.json").write_text(json.dumps(dict(case=a.case, sources=rows,
            target="fixed training-prototype spill-minus-clean cosine margin",
            input_path="Original RGB -> official full-frame transform -> BF16 encoder -> trajectory",
            model_checkpoint_sha256=checkpoint_sha,
            fractions=a.fractions, random_seeds=a.random_seeds, manual_annotations=0,
            localization_metric=False, localization_validated=False), indent=2) + "\n")
    if ctx.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
