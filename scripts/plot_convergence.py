#!/usr/bin/env python3
"""Plot the measured long-run controls and fixed-target refinement."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for folder, variant, label, color in [
        ("convergence_baseline_240_epochs", "constant_lr", "Original AdamW recipe", "#c26428"),
        ("convergence_refinement_240_epochs", "variance_refinement", "Fixed-target variance refinement", "#2467a2"),
    ]:
        histories = [json.loads((ROOT / "reports" / folder / f"{variant}_seed{s}.json").read_text())["history"]
                     for s in (239, 240, 241)]
        epochs = [r["epoch"] for r in histories[0]]
        for ax, key in zip(axes, ["contrastive_excess_loss", "uncertainty_regret"]):
            values = np.array([[r[key] for r in h] for h in histories])
            ax.plot(epochs, values.mean(0), label=label, color=color, linewidth=2)
            ax.fill_between(epochs, values.min(0), values.max(0), color=color, alpha=.16)
    for ax, title in zip(axes, ["Contrastive excess above positive entropy", "Uncertainty error above residual optimum"]):
        ax.set(title=title, xlabel="Epoch / refinement round", yscale="log")
        ax.axhline(1e-3, color="#555555", linewidth=.8, linestyle=":", label="Fit tolerance: 0.001")
        ax.axvline(80, color="#888888", linewidth=.8, linestyle="--")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Loss gap (log scale)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Clean (17) vs warm/cold (7) · three seeds · training data", fontsize=12)
    fig.savefig(ROOT / "reports/convergence_comparison.png", dpi=180)
    fig.savefig(ROOT / "reports/convergence_comparison.pdf")


if __name__ == "__main__":
    main()
