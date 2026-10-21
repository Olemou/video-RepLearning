#!/usr/bin/env python3
"""Reproduce recognition, provenance and temporal controls without encoder fitting.

Writes a new audit directory; never modifies an input experiment. All prototype
and retrieval galleries contain training sources only. Selection-set results
are explicitly not an independent test. Run with the environment in cluster.md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from PIL import Image, ImageDraw

from vision_model.training.trajectory import TrajectoryModel, trajectory_loss
from vision_model.training.vico import vico_pairwise_contrastive
from vision_model.interpretability.temporal_attribution import AttributionTarget, attribute_trajectory


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recognition(z, labels, train, query, *, exclude_self=False, prototypes=None):
    z = F.normalize(z.float(), dim=-1)
    gallery = z[train]
    proto = (torch.stack([F.normalize(gallery[labels[train] == c].mean(0), dim=0)
                          for c in (0, 1)]) if prototypes is None else prototypes)
    scores = z[query] @ proto.T
    # Training-fit prototype scores include the query in its own class mean.
    margin = (scores[:, 1] - scores[:, 0]).numpy()
    truth = labels[query].numpy()
    prediction = scores.argmax(1).numpy()
    similarities = z[query] @ gallery.T
    if exclude_self:
        similarities = similarities.masked_fill(query[:, None] == train[None], -torch.inf)
    ranked = similarities.argsort(1, descending=True)
    relevance = labels[train][ranked].eq(labels[query, None])
    if exclude_self:
        relevance &= train[ranked] != query[:, None]
    ap = (relevance.cumsum(1) / torch.arange(1, len(train) + 1) * relevance).sum(1) / relevance.sum(1)
    return dict(n=len(query), accuracy=float(accuracy_score(truth, prediction)),
                confusion_matrix=confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
                roc_auc=float(roc_auc_score(truth, margin)),
                average_precision=float(average_precision_score(truth, margin)),
                retrieval_top1=float(relevance[:, 0].float().mean()),
                retrieval_precision_at5=float(relevance[:, :5].float().mean()),
                retrieval_mean_average_precision=float(ap.mean()),
                minimum_correct_class_margin=float((torch.as_tensor(margin) * (labels[query] * 2 - 1)).min()),
                predictions=prediction.tolist(), spill_minus_clean_cosine=margin.tolist())


def line_figure(path, rows):
    """Honest loss panels: fixed validation vs batch training; linear axes."""
    canvas = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(canvas)
    for panel, (title, keys) in enumerate([
        ("Total (batch training and fixed validation differ)", ["training_loss", "validation_loss"]),
        ("Semantic (training curriculum changes)", ["training_semantic_loss", "validation_semantic_loss"]),
        ("Prediction and persistence", ["training_prediction_loss", "validation_prediction_loss", "validation_persistence_loss"]),
        ("Validation regularizers", ["validation_alignment_loss", "validation_retention_loss"]) ]):
        x0, y0 = 50 + panel % 2 * 590, 40 + panel // 2 * 340
        draw.text((x0, y0), title, fill="black")
        xmax = max(float(r["update"]) for r in rows)
        ymax = max(float(r[k]) for r in rows for k in keys if r.get(k))
        draw.rectangle((x0, y0 + 25, x0 + 520, y0 + 235), outline="black")
        draw.text((x0, y0 + 240), f"updates 0..{xmax:g}; y 0..{ymax:.4g}", fill="black")
        for offset, (key, color) in enumerate(zip(keys, ("blue", "red", "green"))):
            points = [(x0 + 520 * float(r["update"]) / xmax,
                       y0 + 235 - 210 * float(r[key]) / ymax) for r in rows if r.get(key)]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            draw.text((x0, y0 + 258 + 15 * offset), key, fill=color)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(239)
    checkpoint_path = a.experiment / "training/trial_01_checkpoint.pt"
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    c = torch.load(a.cache, map_location="cpu", weights_only=False)
    model = TrajectoryModel(ck["width"], ck["hidden"], ck["embedding_dim"]).eval()
    model.load_state_dict(ck["model"], strict=True)
    ids, y = c["ids"], c["labels"]
    train = torch.tensor([ids.index(v) for v in ck["train_source_ids"]])
    val = torch.tensor([ids.index(v) for v in ck["validation_source_ids"]])
    x, valid, times = c["features"][:, 0], c["valid"], c["timestamps"]
    with torch.no_grad():
        z = model(x, valid, times)["embedding"]
    proto = torch.stack([F.normalize(z[train][y[train] == label].mean(0), dim=0) for label in (0, 1)])
    results = dict()
    results["baseline"] = {"training_fit": recognition(z, y, train, train, exclude_self=True),
                           "source_validation_selection_set": recognition(z, y, train, val)}
    for split, indices in (("training", train), ("validation", val)):
        with torch.no_grad():
            _, detail, _ = trajectory_loss(model, c["features"][indices], valid[indices],
                                           times[indices], y[indices], epoch=20, curriculum_epochs=20)
        results["baseline"][split + "_fixed_identity_mirror_objective"] = {k: float(v) for k, v in detail.items()}
    for name in ("first_state_only", "last_state_only", "first_state_repeated", "last_state_repeated",
                 "mean_state_repeated", "reversed_states", "shuffled_states", "zero_timestamps",
                 "remove_temporal_residual", "uniform_event_pooling"):
        changed, mask, clock = x.clone(), valid.clone(), times.clone()
        evaluated = model
        for i in range(len(x)):
            n = int(valid[i].sum())
            if name == "first_state_only":
                mask[i, 1:] = False
            elif name == "last_state_only":
                changed[i, 0] = x[i, n - 1]
                clock[i, 0] = times[i, n - 1]
                mask[i, 1:] = False
            elif name in ("first_state_repeated", "last_state_repeated", "mean_state_repeated"):
                changed[i, :n] = {"first_state_repeated": x[i, 0], "last_state_repeated": x[i, n - 1],
                                   "mean_state_repeated": x[i, :n].mean(0)}[name]
            elif name == "reversed_states":
                changed[i, :n] = x[i, :n].flip(0)
            elif name == "shuffled_states":
                changed[i, :n] = x[i, :n][torch.randperm(n, generator=torch.Generator().manual_seed(239 + i))]
        if name == "zero_timestamps":
            clock.zero_()
        if name in ("remove_temporal_residual", "uniform_event_pooling"):
            evaluated = copy.deepcopy(model)
            with torch.no_grad():
                layer = evaluated.up if name == "remove_temporal_residual" else evaluated.event_score
                layer.weight.zero_()
                layer.bias.zero_()
        with torch.no_grad():
            altered_z = evaluated(changed, mask, clock)["embedding"]
        results[name] = recognition(altered_z, y, train, val, prototypes=proto)
        results[name]["mean_validation_embedding_cosine_to_original"] = float((altered_z[val] * z[val]).sum(-1).mean())
    frozen_mean = F.normalize((x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True), dim=-1)
    results["frozen_teacher_mean_prototypes"] = recognition(frozen_mean, y, train, val)
    results["frozen_teacher_first_prototypes"] = recognition(x[:, 0], y, train, val)
    # Reproduce saved target at exact cached identity features (bf16 cache).
    metadata = json.loads((a.experiment / "attention_folderImage13/attribution.json").read_text())
    index = ids.index("cold_folderImage13")
    target = AttributionTarget("semantic_similarity", reference=torch.tensor(metadata["target"]["reference_embedding"]))
    attr = attribute_trajectory(model, x[index], valid[index], times[index], target,
                               video_id=ids[index], include_maps=True)
    gradient = attr.signed_gradients
    results["temporal_gradient_audit"] = dict(saved_target=metadata["target_value"],
        cached_target=attr.target_value, min=float(gradient.min()), max=float(gradient.max()),
        negative_count=int((gradient < 0).sum()), positive_count=int((gradient > 0).sum()),
        zero_positive_map=bool(attr.relevance.count_nonzero() == 0),
        negative_mass=float(gradient.clamp_max(0).abs().sum()),
        positive_mass=float(gradient.clamp_min(0).sum()),
        event_pooling=attr.event_pooling_weights.tolist(),
        signed_mean_over_heads_queries_layers=gradient.mean((0, 1, 2)).tolist(),
        explanation="Positive clipping can erase a connected, nonzero negative gradient.")
    torch.save(dict(embedding=z, prototypes=proto, ids=ids, labels=y,
                    train_indices=train, validation_indices=val,
                    checkpoint_sha256=digest(checkpoint_path)), a.output / "audit_representations.pt")
    manifest = [json.loads(line) for line in Path(c["manifest"]).read_text().splitlines() if line.strip()]
    sequences = c["source_sequences"]
    rows = list(csv.DictReader((a.experiment / "training/loss_history.csv").open()))
    slopes = {}
    for key in ("validation_loss", "validation_semantic_loss", "validation_prediction_loss", "training_loss"):
        tail = [r for r in rows if r.get(key) and int(r["update"]) >= 80]
        slopes[key] = dict(slope_per_update=float(np.polyfit([float(r["update"]) for r in tail], [float(r[key]) for r in tail], 1)[0]),
                           update80=float(tail[0][key]), update100=float(tail[-1][key]))
    line_figure(a.output / "audited_loss_panels.png", rows)
    old = a.experiment.parent / "trajectory_80_sources_real_ddp_100updates_20261021"
    old_ck_path = old / "training/trial_01_checkpoint.pt"
    old_ck = torch.load(old_ck_path, map_location="cpu", weights_only=False)
    weight_difference = max(float((v - old_ck["model"][k]).abs().max()) for k, v in ck["model"].items())
    provenance = dict(checkpoint_sha256=digest(checkpoint_path), cache_sha256=digest(a.cache),
        recorded_cache_sha256=ck["cache_sha256"], encoder_sha256=c["checkpoint_sha256"],
        attention_checkpoint_matches=metadata["checkpoint_id"] == digest(checkpoint_path),
        neighboring_tsne=json.loads((old / "class_tsne.json").read_text()),
        neighboring_checkpoint_sha256=digest(old_ck_path),
        neighboring_max_model_tensor_difference=weight_difference,
        source_id_overlap=sorted(set(ck["train_source_ids"]) & set(ck["validation_source_ids"])),
        source_group_overlap=sorted({sequences[i] for i in train} & {sequences[i] for i in val}),
        byte_hash_overlap=sorted({c["video_sha256"][i] for i in train} & {c["video_sha256"][i] for i in val}),
        unique_cache_video_hashes=len(set(c["video_sha256"])),
        scene_group_metadata_available=any("scene" in r for r in manifest),
        categories={k:c["categories"].count(k) for k in sorted(set(c["categories"]))},
        optimizer_state_steps=sorted({int(v["step"]) for v in ck["optimizer"]["state"].values()}),
        source_inventory=[dict(id=ids[i], split="train" if i in train else "validation", **manifest[[r["video_id"] for r in manifest].index(ids[i])]) for i in range(len(ids))],
        input_artifacts=[dict(path=str(f.relative_to(a.experiment)), bytes=f.stat().st_size, sha256=digest(f))
                         for f in sorted(a.experiment.rglob("*")) if f.is_file() and a.output not in f.parents],
        audit_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        audit_git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        original_code_version="Not recorded in checkpoint or visualization; current working tree is not a historical snapshot.",
        python=sys.version, torch=torch.__version__, slopes_last21_updates=slopes)
    (a.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (a.output / "recognition_and_controls.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: ({s:v.get("accuracy") for s,v in r.items() if isinstance(v,dict)} if k=="baseline" else r.get("accuracy")) for k,r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
