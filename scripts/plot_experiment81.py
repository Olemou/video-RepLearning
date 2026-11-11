#!/usr/bin/env python3
"""Matplotlib export of measured Experiment81 curves and saved t-SNE coordinates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "scientific_figures"; out.mkdir(exist_ok=False)
    histories = {case: json.loads((a.root / case / "history.json").read_text()) for case in ("frozen", "full")}
    colors = {"frozen": "#2056a8", "full": "#bc3724"}

    def lines(ax, key, weight=1, minimum=0):
        for case, history in histories.items():
            for split, style in (("train", "-"), ("selection", "--")):
                rows = [r for r in history if r["update"] >= minimum and split in r]
                ax.plot([r["update"] for r in rows], [weight * r[split][key] for r in rows],
                        color=colors[case], linestyle=style, label=f"{case}: {split}", linewidth=1.6)
        ax.axvline(21, color="gray", linewidth=.8, linestyle=":", label="optimizer reset before 21")
        ax.set_xlabel("Optimizer update"); ax.grid(alpha=.2)

    def save(fig, name):
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for minimum, name in ((0, "total_loss"), (20, "late_total_loss")):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        lines(ax, "total_loss", minimum=minimum)
        ax.set_ylabel("ViCo + 0.1 alignment + prediction + 0.1 retention")
        ax.set_title("Experiment 81: complete four-loss objective" + (" (updates 20 onward)" if minimum else ""))
        ax.set_ylim(bottom=0); ax.legend(fontsize=8)
        save(fig, name)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    keys = [("semantic_loss", "Episode ViCo"), ("alignment_loss", "Alignment"),
            ("prediction_loss", "Future prediction"), ("retention_loss", "Retention"),
            ("persistence_loss", "Frozen persistence"), ("total_loss", "Weighted total")]
    for ax, (key, title) in zip(axes.flat, keys):
        lines(ax, key); ax.set_title(title); ax.set_ylabel("Loss")
        if key == "semantic_loss":
            ax.set_yscale("log"); ax.set_ylabel("Loss (log scale)")
        else:
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed final-curriculum evaluation; 64 training / 16 reused source-selection videos")
    save(fig, "loss_components")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, case in zip(axes, ("frozen", "full")):
        data = json.loads((a.root / "figures" / f"tsne_{case}_final.json").read_text())
        for label, color, category in ((0, "#2056a8", "clean"), (1, "#bc3724", "cold spill")):
            for split, marker in (("train", "o"), ("selection", "s")):
                rows = [r for r in data["sources"] if r["label"] == label and r["split"] == split]
                ax.scatter([r["coordinates"][0] for r in rows], [r["coordinates"][1] for r in rows],
                           c=color, marker=marker, s=35, edgecolors="black", linewidths=.4, label=f"{category}, {split}")
        ax.set_title(case + " encoder + trajectory"); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle("Actual episode embeddings; independent projections; not localization evidence")
    save(fig, "tsne_comparison")
    (out / "provenance.json").write_text(json.dumps(dict(matplotlib_version=matplotlib.__version__,
        loss_input="Unmodified history.json in each arm", tsne_input="Saved 80-source t-SNE coordinate JSONs",
        independent_tsne_fits=True, localization_evidence=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
