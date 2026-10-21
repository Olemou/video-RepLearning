#!/usr/bin/env python3
"""Export the measured training and full-data loss curves from one run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    history = payload["history"]
    trained = [r for r in history if r["epoch"] > 0]
    expected = list(range(1, payload["epochs"] + 1))
    if [r["epoch"] for r in trained] != expected:
        raise ValueError("Report does not contain every completed epoch")
    if any(r.get("batch_objective_scope") != "balanced_minibatches_pre_update" for r in trained):
        raise ValueError("This plot requires a consistent mini-batch training phase throughout the run")
    output = args.output or args.report.with_name("loss_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [
        ("training_loss", "Training loss: six-video batches", "#245f9e"),
        ("eval_full_dataset_loss", "End-of-epoch loss: all videos", "#bd5728"),
    ]
    for ax, (field, title, color) in zip(axes, panels):
        values = [r[field] for r in trained]
        ax.plot(expected, values, color=color, linewidth=1.8)
        ax.scatter([expected[-1]], [values[-1]], color=color, s=22, zorder=3)
        ax.annotate(f"Epoch {expected[-1]}: {values[-1]:.6f}",
                    xy=(expected[-1], values[-1]), xytext=(-10, 23),
                    textcoords="offset points", ha="right", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=color))
        ax.set(title=title, xlabel="Completed epoch", ylabel="Loss")
        ax.set_xlim(1, expected[-1])
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    counts = payload["class_counts"]
    fig.suptitle(f"Real-video training · {payload['epochs']} epochs · seed {payload['seed']}", fontsize=15, y=.97)
    fig.text(.5, .895, f"Clean = 0 ({counts['0']} videos) | Warm/cold = 1 ({counts['1']} videos) | Frozen V-JEPA 2 + trainable Gaussian heads",
             ha="center", fontsize=10)
    fig.text(.5, .025,
             "Measured, unsmoothed curves. Left: mean of pre-update batch losses. Right: final model, full dataset, final curriculum.\n"
             "Different candidate sets imply different loss scales. Both measurements use training videos; neither is held-out validation.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(top=.79, bottom=.19, left=.08, right=.97, wspace=.25)
    fig.savefig(output, dpi=200, facecolor="white")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        fields = ["epoch", "training_loss", "eval_fixed_minibatch_loss", "eval_full_dataset_loss",
                  "contrastive_excess_loss", "uncertainty_regret"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trained)
    print(json.dumps({"png": str(output), "epochs": len(trained),
                      "first_training_loss": trained[0]["training_loss"],
                      "final_training_loss": trained[-1]["training_loss"],
                      "final_full_dataset_loss": trained[-1]["eval_full_dataset_loss"]}, indent=2))


if __name__ == "__main__":
    main()
