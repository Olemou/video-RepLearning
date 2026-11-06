#!/usr/bin/env python3
"""Dependency-light measured loss curves, t-SNE and report for Experiment81."""
from __future__ import annotations
import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from prepare_experiment81 import digest


def font(size=16):
    return ImageFont.load_default(size=size)


COLORS = ("#1552ad", "#78a2e0", "#c13b21", "#e89379")


def panel(draw, box, series, title):
    x0, y0, x1, y1 = box
    draw.text((x0, y0), title, fill="black", font=font(17))
    plot = (x0 + 65, y0 + 35, x1 - 12, y1 - 95)
    values = [v for _, points in series for _, v in points]
    if not values:
        return
    lo, hi = min(0., min(values)), max(values)
    hi = max(hi, lo + 1e-12)
    hi += (hi - lo) * .04
    xmax = max(1, max(x for _, points in series for x, _ in points))
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4
        y = plot[3] - tick * (plot[3] - plot[1]) / 4
        draw.line((plot[0], y, plot[2], y), fill="#dddddd")
        draw.text((x0, y - 8), f"{value:.3g}", fill="black", font=font(12))
    draw.rectangle(plot, outline="black")
    reset = plot[0] + 21 / xmax * (plot[2] - plot[0])
    if xmax >= 21:
        for y in range(int(plot[1]), int(plot[3]), 8):
            draw.line((reset, y, reset, min(y + 4, plot[3])), fill="gray")
    for j, (label, points) in enumerate(series):
        xy = [(plot[0] + x / xmax * (plot[2] - plot[0]),
               plot[3] - (v - lo) / (hi - lo) * (plot[3] - plot[1])) for x, v in points]
        if len(xy) > 1:
            draw.line(xy, fill=COLORS[j], width=3)
        draw.text((x0 + (j % 2) * (x1 - x0) / 2, y1 - 65 + (j // 2) * 22),
                  label, fill=COLORS[j], font=font(13))
    draw.text((plot[0], plot[3] + 10), f"Optimizer updates 0–{xmax:g}; reset before 21", fill="black", font=font(12))


def curve_figure(path, histories, keys):
    rows = (len(keys) + 1) // 2
    canvas = Image.new("RGB", (1500, rows * 390 + 60), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), "Experiment 81 | fixed final-curriculum evaluation | source selection is not a final test",
              fill="black", font=font(19))
    for k, (key, label, weight) in enumerate(keys):
        series = []
        for case in ("frozen", "full"):
            for split in ("train", "selection"):
                points = [(r["update"], r[split][key] * weight) for r in histories[case] if split in r]
                series.append((f"{case} {split}", points))
        x, y = 20 + k % 2 * 750, 65 + k // 2 * 390
        panel(draw, (x, y, x + 715, y + 375), series, label)
    canvas.save(path)


def tsne(path, representation, *, title):
    values = representation["embeddings"][:, 0].numpy()
    kwargs = dict(n_components=2, perplexity=15, random_state=239, init="pca", learning_rate="auto")
    kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 2000
    xy = TSNE(**kwargs).fit_transform(values)
    lo, hi = xy.min(0), xy.max(0)
    coords = (xy - lo) / np.maximum(hi - lo, 1e-10)
    canvas = Image.new("RGB", (1000, 850), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), title, fill="black", font=font(21))
    draw.text((25, 49), "Blue: clean; red: cold spill | circle: train; square: source selection", fill="black", font=font(17))
    train = set(representation["train_indices"].tolist())
    rows = []
    for i, (x, y) in enumerate(coords):
        x, y = 60 + x * 880, 110 + y * 660
        color = "#1552ad" if int(representation["labels"][i]) == 0 else "#c13b21"
        shape = draw.ellipse if i in train else draw.rectangle
        shape((x - 7, y - 7, x + 7, y + 7), fill=color, outline="black", width=1)
        rows.append(dict(id=representation["ids"][i], label=int(representation["labels"][i]),
                         split="train" if i in train else "selection", coordinates=xy[i].tolist()))
    draw.text((25, 805), "Separate t-SNE fits: axis locations cannot be compared across figures; not localization evidence.",
              fill="black", font=font(15))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(dict(settings=kwargs, sources=rows), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    out = a.root / "figures"
    out.mkdir(exist_ok=True)
    histories, settings = {}, {}
    for case in ("frozen", "full"):
        histories[case] = json.loads((a.root / case / "history.json").read_text())
        settings[case] = json.loads((a.root / case / "settings.json").read_text())
        for stage, filename in (("initial", "initial_representations.pt"), ("final", "representations.pt")):
            r = torch.load(a.root / case / filename, map_location="cpu", weights_only=False)
            tsne(out / f"tsne_{case}_{stage}.png", r, title=f"Experiment81 {case}, {stage}: 80 identity embeddings")
    if settings["frozen"]["initial_trajectory_sha256"] != settings["full"]["initial_trajectory_sha256"]:
        raise RuntimeError("The paired initial trajectory weights differ")
    if settings["frozen"]["prepared_sha256"] != settings["full"]["prepared_sha256"]:
        raise RuntimeError("The paired data/teachers differ")
    keys = [("total_loss", "Weighted total loss", 1), ("semantic_loss", "ViCo episode loss", 1),
            ("alignment_loss", "Alignment, raw", 1), ("prediction_loss", "Future prediction, raw", 1),
            ("retention_loss", "Retention, raw", 1), ("persistence_loss", "Frozen persistence", 1)]
    curve_figure(out / "loss_components.png", histories, keys)
    curve_figure(out / "weighted_components.png", histories,
                 [(k, name + f" ×{w:g}", w) for k, name, w in [
                     ("semantic_loss", "ViCo", 1), ("alignment_loss", "Alignment", .1),
                     ("prediction_loss", "Prediction", 1), ("retention_loss", "Retention", .1)]])
    # Dedicated single panel of the complete objective, separate from components.
    canvas = Image.new("RGB", (1200, 650), "white"); draw = ImageDraw.Draw(canvas)
    series = [(f"{case} {split}", [(r["update"], r[split]["total_loss"]) for r in histories[case] if split in r])
              for case in ("frozen", "full") for split in ("train", "selection")]
    panel(draw, (25, 20, 1175, 630), series, "Experiment81: ViCo + 0.1 alignment + prediction + 0.1 retention")
    canvas.save(out / "total_loss.png")
    rows = []
    for case in histories:
        for r in histories[case]:
            for split in ("train", "selection"):
                if split in r:
                    rows.append(dict(case=case, split=split, update=r["update"],
                                     **{k: r[split][k] for k, _, _ in keys}))
    with (out / "loss_history.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    measured = {}
    for case in histories:
        h = histories[case]
        last = h[-1]
        tail = [r for r in h if r["update"] >= max(0, last["update"] - 20) and "selection" in r]
        slopes = {key: float(np.polyfit([r["update"] for r in tail], [r["selection"][key] for r in tail], 1)[0])
                  for key in ("total_loss", "semantic_loss", "prediction_loss")}
        measured[case] = dict(update=last["update"], train=last["train"], selection=last["selection"],
            selection_last20_slopes=slopes, checkpoint_sha256=digest(a.root / case / "checkpoint.pt"),
            replay_max_feature_error=max(r.get("replay_max_feature_error", 0.) for r in h))
    (a.root / "measured_results.json").write_text(json.dumps(measured, indent=2) + "\n")
    report = ["# Experiment 81: measured online full-backbone comparison", "",
        "Original four losses retained. No manual spatial annotations, MIL or decoder. "
        "The 16 repeatedly used selection sources are not an independent scene test.", "",
        "[Single total-loss curve](figures/total_loss.png) · [Loss components](figures/loss_components.png) · "
        "[Weighted components](figures/weighted_components.png) · [CSV](figures/loss_history.csv)", "",
        "| Arm | Updates | Selection total | Selection ViCo | Prediction | Persistence | Prototype accuracy | Retrieval mAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, result in measured.items():
        m = result["selection"]
        report.append(f"| {case} | {result['update']} | {m['total_loss']:.7g} | {m['semantic_loss']:.7g} | "
            f"{m['prediction_loss']:.7g} | {m['persistence_loss']:.7g} | {m['accuracy']:.4f} | {m['retrieval_mean_average_precision']:.4f} |")
    report += ["", "[Frozen t-SNE](figures/tsne_frozen_final.png) · [Full fine-tuned t-SNE](figures/tsne_full_final.png)", "",
        "Both t-SNEs include 64 training and 16 selection identity embeddings. Neither is evidence of liquid localization.", "",
        "The teacher was recomputed from the original immutable pretrained checkpoint with FP32 master weights "
        "and BF16 forward computation, matching the new student precision. The historical cache used BF16 weights; "
        "its measured numerical differences are preserved in prepared/metadata.json. This is not an exact numerical "
        "reproduction of historical curves. The new two arms share identical targets, views, split and initial trajectory weights.", "",
        f"Actual hardware: {settings['full']['world_size']} allocated GPU(s), "
        f"{settings['full']['gpu_names']}, host {settings['full']['host']}, allocation {settings['full']['job_id']}. "
        "The report proposed two H100s; the requested allocation provided one. No unused GPU is claimed.", "",
        "[Complete metrics](measured_results.json) · [Full training history](full/history.json) · "
        "[Gradient audit](full/initial_feature_gradient_audit.json)", "",
        "## Spatial evidence", "",
        "Classification and loss results above do not establish spill focus. See the separately generated "
        "spatial gallery and input-removal controls when present. Without real masks, no IoU, pixel AP "
        "or real pointing-game accuracy can be reported. Qualitative acceptance remains unverified until maps are inspected.", "",
        "Neither a fixed update budget nor near-zero semantic loss establishes full convergence. "
        "Late-update temporal slopes are recorded in measured_results.json.", ""]
    (a.root / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
