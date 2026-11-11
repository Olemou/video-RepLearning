#!/usr/bin/env python3
"""Counterfactual recognition tests for brightness, spatial structure and clocks.

Evaluation only: no optimizer update, retraining, masks or new spatial labels.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from vision_model.training.trajectory import trajectory_loss
from vision_model.training.online_trajectory import OnlinePrefixEncoder, collect_prefix_features
from attribute_experiment81 import load_case, evaluation_groups
from control_experiment81 import transform_bank
from prepare_experiment81 import digest
from diagnose_spill_representation import recognition
from vision_model.data.transforms import OfficialVJEPA2Transform


def image_control(frames, kind, *, seed=239):
    if frames.ndim != 4 or frames.dtype != torch.uint8:
        raise ValueError("Expected uint8 [T,3,H,W]")
    if kind == "spatial_mean_only":
        return frames.float().mean((-2, -1), keepdim=True).round().byte().expand_as(frames).clone()
    if kind == "fixed_gain_offset":
        return (frames.float() * .5 + 64).round().byte()
    if kind == "matched_frame_mean":
        value = .5 * (frames.float() - frames.float().mean((-2, -1), keepdim=True)) + 127.5
        if value.min() < 0 or value.max() > 255:
            raise RuntimeError("Mean-matching clipped physical contrast")
        return value.round().byte()
    if kind == "spatial_patch_shuffle":
        t, channels, h, w = frames.shape
        common = math.gcd(h, w)
        block = common // 4 if common % 4 == 0 else common // 2 if common % 2 == 0 else 1
        r, c = h // block, w // block
        patches = frames.reshape(t, channels, r, block, c, block).permute(0, 1, 2, 4, 3, 5).reshape(t, channels, r * c, block, block)
        order = torch.randperm(r * c, generator=torch.Generator().manual_seed(seed))
        return patches[:, :, order].reshape(t, channels, r, c, block, block).permute(0, 1, 2, 4, 3, 5).reshape_as(frames)
    raise ValueError(kind)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", choices=("frozen", "full"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(); torch.set_num_threads(4)
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the actual allocated GPU")
    device = torch.device("cuda", 0)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    encoder, model, reps, proto, settings = load_case(a.root, a.case, device)
    transform = OfficialVJEPA2Transform(settings["teacher_metadata"]["encoder_dir"], preserve_field_of_view=True)
    groups = evaluation_groups(reps, settings)
    wrapper = OnlinePrefixEncoder(encoder)
    train, selection, labels = reps["train_indices"], reps["selection_indices"], reps["labels"]

    def metrics(features, timestamps, *, temporal=False):
        embeddings = torch.zeros(len(labels), 2, 128)
        terms = {}
        for split, ids in (("train", train), ("selection", selection)):
            n, views, time, width = features[ids].shape
            valid = reps["valid"][ids].to(device)
            clocks = timestamps[ids].to(device)
            out = model(features[ids].flatten(0, 1).to(device), valid[:, None].expand(n, views, time).flatten(0, 1),
                        clocks[:, None].expand(n, views, time).flatten(0, 1), detach_teacher=False)
            embeddings[ids] = out["embedding"].reshape(n, views, -1).cpu()
            if temporal:
                _, detail, _ = trajectory_loss(model, c["teacher"][ids, :2].to(device), valid,
                    clocks, labels[ids].to(device), epoch=20, curriculum_epochs=20,
                    student_features=features[ids].to(device))
                terms[split] = {k: float(detail[k]) for k in ("total_loss", "prediction_loss", "persistence_loss")}
        return {split: dict(recognition(embeddings[:, 0], labels, train, ids,
            exclude_self=split == "train", prototypes=proto.cpu()), **terms.get(split, {}))
            for split, ids in (("train", train), ("selection", selection))}

    result = dict(case=a.case, checkpoint_sha256=digest(a.root / a.case / "checkpoint.pt"),
        original=metrics(reps["features"], reps["timestamps"], temporal=True), image_controls={}, clock_controls={},
        notes=["Fixed original training prototypes; no refitting or recalibration on altered inputs",
            "Image controls are inference stress tests, not training augmentations or localization metrics",
            "Spatial-mean inputs contain no within-frame liquid shape; failure on them does not exclude shortcuts",
            "Matched-mean and fixed-gain controls share gain .5; both avoid clipping",
            "Clock shifts preserve actual timestamp deltas, tubelet order and image features"], localization_validated=False)
    a.output.mkdir(parents=True, exist_ok=a.resume)
    if a.resume and (a.output / "results.json").exists():
        old = json.loads((a.output / "results.json").read_text())
        if old["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("Resume checkpoint differs")
        result["image_controls"].update(old.get("image_controls", {}))
    origin = reps["timestamps"][:, :1]
    relative = reps["timestamps"] - origin
    generator = torch.Generator().manual_seed(239)
    random_origins = origin[train][torch.randint(len(train), (len(labels),), generator=generator)]
    for name, clocks in (("relative_origin_zero", relative),
                         ("common_training_origin", relative + origin[train].mean()),
                         ("random_training_origins", relative + random_origins)):
        torch.testing.assert_close(clocks[:, 1:] - clocks[:, :-1],
                                   reps["timestamps"][:, 1:] - reps["timestamps"][:, :-1], atol=5e-6, rtol=0)
        result["clock_controls"][name] = metrics(reps["features"], clocks, temporal=True)
    for kind in ("spatial_mean_only", "fixed_gain_offset", "matched_frame_mean", "spatial_patch_shuffle"):
        if kind in result["image_controls"]:
            continue
        features = torch.zeros_like(reps["features"])
        for ids in groups:
            pixels = torch.stack([transform_bank(image_control(c["rgb"][int(i)], kind, seed=239 + int(i)),
                [], transform, include_augmentations=False) for i in ids])
            features[ids] = collect_prefix_features(wrapper, pixels, c["valid"][ids], device,
                                                   chunk_sources=len(ids)).cpu()
        result["image_controls"][kind] = metrics(features, reps["timestamps"], temporal=False)
        print(json.dumps(dict(case=a.case, control=kind,
            selection_accuracy=result["image_controls"][kind]["selection"]["accuracy"],
            selection_auc=result["image_controls"][kind]["selection"]["roc_auc"])), flush=True)
        (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
