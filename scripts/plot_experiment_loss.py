#!/usr/bin/env python3
"""Plot comparable video-objective or segmentation losses from one report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def video_plot(payload, output):
    history = [row for row in payload["history"] if row["epoch"] > 0]
    epochs = [row["epoch"] for row in history]
    total = [row["eval_full_dataset_loss"] for row in history]
    excess = [max(row["contrastive_excess_loss"], 1e-12) for row in history]
    variance = [max(row["uncertainty_regret"], 1e-12) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(epochs, total, color="#1769aa", linewidth=2, label="Full-dataset objective")
    axes[0].set(title="Video training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, excess, color="#cc5c33", label="Contrastive excess")
    axes[1].plot(epochs, variance, color="#28865f", label="Variance regret")
    axes[1].set(title="Distance from component optima", xlabel="Epoch", ylabel="Loss gap", yscale="log")
    axes[1].legend()
    for axis in axes: axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    counts = payload["class_counts"]
    fig.suptitle(f"Frozen V-JEPA 2 · clean {counts['0']} / spill {counts['1']} · seed {payload['seed']}")
    rows = [{"epoch": row["epoch"], "phase": row["phase"],
             "eval_full_dataset_loss": row["eval_full_dataset_loss"],
             "contrastive_excess_loss": row["contrastive_excess_loss"],
             "uncertainty_regret": row["uncertainty_regret"]} for row in history]
    return fig, rows


def segmentation_plot(payload, output):
    history = payload["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for key, label, color in (("loss", "Total", "#1769aa"), ("bce", "Weighted BCE", "#cc5c33"),
                              ("dice_loss", "Dice", "#28865f")):
        axes[0].plot(epochs, [row[key] for row in history], label=label, color=color)
    axes[0].set(title="Segmentation training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for key, label, color in (("iou", "IoU", "#7b4ab5"), ("dice", "Dice score", "#28865f"),
                              ("recall", "Recall", "#cc5c33")):
        axes[1].plot(epochs, [row[key] for row in history], label=label, color=color)
    axes[1].set(title="Annotated training pixels", xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    axes[1].legend()
    for axis in axes: axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mask-supervised spill segmentation · weighted BCE + Dice")
    return fig, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    output = args.output or args.report.with_name("loss_curve.png")
    if "variant" in payload and "eval_full_dataset_loss" in payload["history"][-1]:
        fig, rows = video_plot(payload, output)
    elif payload.get("objective", "").startswith("mask-supervised"):
        fig, rows = segmentation_plot(payload, output)
    else:
        raise ValueError("Unsupported report format")
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=190, facecolor="white"); plt.close(fig)
    fields = list(rows[0])
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"png": str(output), "csv": str(output.with_suffix('.csv')), "points": len(rows)}, indent=2))


if __name__ == "__main__": main()
