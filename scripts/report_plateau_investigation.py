#!/usr/bin/env python3
"""Render the preserved real-video ablations, with comparable evaluation scopes."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/real_video_100_epochs"
OUT = BASE / "investigation"


def read(path):
    return json.loads(path.read_text())


def main():
    experiments = [("Original attention / AdamW", BASE / "constant_lr_seed239.json")]
    for folder, variant, label in (
        ("masked", "constant_lr", "Masked attention / AdamW"),
        ("masked", "adaptive_refinement", "Adaptive / 20 iterations / float32 objective"),
        ("masked_refined", "adaptive_refinement", "Adaptive / 40 iterations / float32 objective"),
        ("corrected", "adaptive_refinement", "Corrected / 40 iterations / float64 objective"),
        ("legacy_refined", "adaptive_refinement", "Original attention / corrected optimizer"),
    ):
        experiments += [(label, OUT / folder / f"{variant}_seed{seed}.json") for seed in (239, 240, 241)]
    loaded = [(label, path, read(path)) for label, path in experiments]
    fields = ["epoch", "loss", "eval_fixed_minibatch_loss", "contrastive_excess_loss", "uncertainty_regret"]
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "seed", *fields])
        writer.writeheader()
        for label, _, result in loaded:
            for row in result["history"]:
                writer.writerow(dict(experiment=label, seed=result["seed"], **{k: row[k] for k in fields}))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    keys = fields[1:]
    titles = ["Full 24-video objective (raw)", "Fixed 3+3 batches (raw)",
              "Contrastive error above positive entropy", "Variance NLL error above fixed-residual optimum"]
    chosen = [loaded[0], loaded[1], next(x for x in loaded if x[0].startswith("Corrected") and x[2]["seed"] == 239)]
    colors = ["#7c8895", "#da8128", "#147d70"]
    for ax, key, title in zip(axes.flat, keys, titles):
        for (label, _, result), color in zip(chosen, colors):
            h = result["history"]
            short = "Corrected training" if label.startswith("Corrected") else label
            ax.plot([r["epoch"] for r in h], [r[key] for r in h], label=short, color=color, lw=2)
        ax.axvline(24, color="#147d70", ls=":", alpha=.5)
        ax.set(xlabel="Epoch", title=title, xlim=(0, 100))
        ax.grid(alpha=.2)
        if key in ("contrastive_excess_loss", "uncertainty_regret"):
            ax.set_yscale("log")
            ax.axhline(.001, color="black", ls="--", alpha=.45, label="Fit tolerance: 0.001")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Real videos: clean=0, warm/cold=1 | seed 239 | frozen V-JEPA 2 encoder", fontsize=14)
    fig.savefig(OUT / "loss_comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Real-video plateau investigation",
        "",
        "## Finding",
        "",
        "The raw loss near 2.4 mostly reflects the multi-positive objective's nonzero floor. "
        "There were also two demonstrated defects: padded frames leaked through encoder attention, "
        "and float32 evaluation caused late L-BFGS line searches to stagnate. Both are fixed. "
        "Adaptive mean freezing addresses the remaining variance optimization bottleneck. "
        "No constant was subtracted from the training loss.",
        "",
        "## Evidence and fixes",
        "",
        "1. Padding was excluded from pooling and NLL, but still participated as keys/values in every transformer block. "
        "Before the fix, three padding invariance tests failed: changing invalid tail frames changed mean outputs "
        "by up to 0.092 in the tiny deterministic regression model, and all 3,072 invalid pixel inputs received gradients. "
        "The fix masks invalid tubelets in every attention layer. The tests now verify padding invariance, zero "
        "padding gradients, equality to a truncated valid clip and unchanged fully valid behavior. "
        "This is an input correctness fix; a lower training loss alone cannot measure its generalization benefit.",
        "2. All 24 MP4s were re-encoded into `checkpoints/masked_video_features.pt`. Old caches are preserved. "
        "68.88% of temporal tubelets are padding; valid-feature relative L2 change after masking was 0.3754 "
        "(mean cosine similarity 0.9280). The training entry point now rejects caches without the corrected policy.",
        "3. Both heads initially train with AdamW. After the objective curriculum, five consecutive full-data "
        "contrastive excess values below 0.001 trigger mean freezing. All three seeds switched after epoch 24. "
        "The variance head then fits fixed residuals using full-data L-BFGS. A 20-iteration budget helped, "
        "but did not meet the variance tolerance by epoch 100; those runs remain in `masked/`.",
        "4. Increasing the budget to 40 iterations exposed flat late-epoch trajectories despite variance gradient "
        "norms around 0.003–0.004. Evaluating the same residual NLL in float64, while keeping heads and their "
        "parameters in float32, allowed further progress in all three seeds. Compare `masked_refined/` with "
        "`corrected/`: same data, seeds, frozen mean targets and iteration budget; scalar-objective precision differs.",
        "",
        "## Architecture and gradients",
        "",
        "Frozen pretrained V-JEPA 2 ViT-L: 24 blocks, width 1024, 16 attention heads, 16×16 patches, "
        "two-frame tubelets. Input is 64×3×256×256. Spatial pooling gives 32 temporal vectors of width 1024. "
        "Each Gaussian head is LayerNorm → Linear(1024,1024) → GELU → Linear(1024,256). "
        "Temporal variances are 1e-4 + softplus(raw). Masked video means and total variances aggregate "
        "temporal predictions. Video embeddings are L2-normalized means. The encoder is frozen in these runs.",
        "",
        "The objective is L = L_con + 0.01 L_unc. L_con uses cosine logits at temperature 0.1, "
        "detached Gaussian distances for positive weights and detached hard-negative weights. "
        "L_unc = average[r²/v + log(v)], where the entire temporal residual r = mu_t − mu_video is detached. "
        "Thus L_unc trains the variance head, not the mean head. At fixed targets, its variance derivative "
        "is proportional to 1/v − r²/v² and its elementwise constrained optimum is max(r², 1e-4). "
        "The shared head need not exactly attain every elementwise optimum.",
        "",
        "Analytical gradient tests independently verify the cosine/normalization derivative and the variance "
        "derivative, including detached weights and padding. Finite differences that recompute detached pair "
        "weights would test a different derivative. During L-BFGS only the fixed-residual NLL is optimized: "
        "using the full forward value with changing detached weights would invalidate the smooth line search.",
        "",
        "## Why 0.5 is not the target",
        "",
        "For each anchor, normalize positive weights to q. Its contrastive loss equals H(q) plus "
        "KL(q || softmax(positive logits)) plus log(1 + negative_partition / positive_partition). "
        "Consequently H(q) is a pointwise lower bound, not zero. It changes as detached weights change "
        "and is not necessarily simultaneously attainable by shared embeddings. With uniform positives, "
        "17 clean and 7 warm/cold videos give (17 log(16) + 7 log(6))/24 = 2.486514; balanced 3+3 "
        "batches give log(2) = 0.693147. The NLL may be negative and shifts the total below these numbers.",
        "",
        "A conservative bound also rules out total loss 0.5 for the final full-data objective: positive weights "
        "are between 0.9 and 1, so H(q) ≥ log(0.9 P), and r²/v + log(v) ≥ log(1e-4). "
        f"Therefore full-data L ≥ {((17 * math.log(16) + 7 * math.log(6)) / 24 + math.log(.9) + .01 * math.log(1e-4)):.6f}. "
        "This bound applies to the final curriculum, the recorded labels and variance floor; it is conservative, "
        "not a prediction of the attained minimum.",
        "",
        "At the original epoch-100 checkpoint, total loss was 2.4447155 and the bound with its residuals and "
        "positive entropy held fixed was 2.4418621: only 0.0028534 improvement separates those quantities. "
        "Rescaling means by 2 and variances by 4 barely changed the contrastive term but increased total loss "
        "by 0.0138631. Raw loss also depends on latent units. The cosine loss has essentially no radial "
        "gradient (maximum measured radial fraction 2.19e-6). These facts do not indicate broken autograd.",
        "",
        "Hard-negative weights in the original fitted checkpoint concentrated on about one effective negative "
        "per anchor (about 9.92 available). This is the specified temperature-driven mining behavior, not proof "
        "of an implementation bug. Contrastive excess was already tiny. Generalization and alternative mining "
        "policies require a separate held-out comparison.",
        "",
        "## All 100-epoch ablations",
        "",
        "| Experiment | Seed | Full loss | Fixed 3+3 loss | Contrastive excess | Variance error | NLL closure calls | Fit criterion |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, path, result in loaded:
        row = result["history"][-1]
        closures = sum(r.get("closure_evaluations", 0) for r in result["history"])
        fit = result["assessment"]["training_fit_converged"]
        lines.append(f"| [{label}]({path.relative_to(OUT) if path.is_relative_to(OUT) else '../' + path.name}) | {result['seed']} "
                     f"| {row['loss']:.7f} | {row['eval_fixed_minibatch_loss']:.7f} "
                     f"| {row['contrastive_excess_loss']:.7f} | {row['uncertainty_regret']:.7f} "
                     f"| {closures} | {'Pass' if fit else 'Not met'} |")
    lines += [
        "",
        "Fit criterion: contrastive excess <0.001, variance error <0.001, and absolute last-ten-epoch "
        "slopes below 1e-5 and 1e-4 respectively. This is a declared approximate training-fit criterion, "
        "not exact stationarity. Variance gradient norms remain nonzero; further fitting is possible.",
        "",
        "All comparisons use the same real-video labels, frozen encoder weights, head initialization and "
        "batch order within each seed. `legacy_refined/` isolates the optimizer change on old features. "
        "The corrected attention and original attention do not share features or final residual targets, "
        "so their raw loss difference cannot be attributed exclusively to optimization quality. "
        "L-BFGS permits 40 iterations plus line search per refinement epoch; AdamW has six minibatch steps. "
        "These are equal-epoch, not equal-compute comparisons. Hyperparameters were selected during this investigation.",
        "",
        "Curves use fixed end-of-epoch evaluation scopes. `training_loss` in the JSON changes scope from "
        "pre-update minibatch averages to post-update full-data loss during refinement; do not splice it "
        "into a single comparable curve. Variance error is averaged over valid elements; the optimized "
        "NLL weights videos equally, so small length differences make their averaging slightly different.",
        "",
        "## Artifacts and reproduction",
        "",
        "- [Loss comparison PNG](loss_comparison.png) and [all curve values](comparison.csv).",
        "- [Original checkpoint diagnostics](diagnostics.json).",
        "- [Corrected runs](corrected/summary.json) and [checkpoint replay verification](artifact_verification.json).",
        "- Final regression suite: **60 passed**, including analytical gradients, padding invariance, "
        "distributed training, adaptive phase transitions, cache policy and exact refinement resume. "
        "Independent checkpoint replay verified **16 reports**: 12 with masked features and four with "
        "legacy features; the accepted metric error was at most 1e-6.",
        "- `PYTHONPATH=src .venv/bin/python scripts/train.py --device cuda` runs the corrected 100-epoch "
        "recipe from `configs/train.yaml` into a separate output directory.",
        "- `scripts/convergence_study.py --cache checkpoints/masked_video_features.pt --epochs 100 "
        "--variants adaptive_refinement --seeds 239 240 241 --refinement-max-iter 40 --refinement-precision float64 "
        "--output reports/reproduction` reproduces the corrected experiment (use the project virtual environment).",
        "",
        "All 24 videos are training data. No held-out accuracy, 70% generalization claim, probability calibration "
        "or improvement from unfreezing the encoder has been established. Keep session/scene-related clips "
        "together when constructing a future validation split. Existing original reports remain intact.",
    ]
    (OUT / "investigation.md").write_text("\n".join(lines) + "\n")
    print(OUT / "loss_comparison.png")
    print(OUT / "investigation.md")


if __name__ == "__main__":
    main()
