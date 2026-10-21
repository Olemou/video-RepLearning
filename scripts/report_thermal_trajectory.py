#!/usr/bin/env python3
"""Compare the two real 100-epoch runs and evaluate all cached thermal views."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vision_model.training.trajectory import TrajectoryModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "reports/trajectory_thermal_aug_100")
    p.add_argument("--baseline", type=Path, default=ROOT / "reports/trajectory_100")
    args = p.parse_args()
    torch.set_num_threads(4)
    baseline = json.loads((args.baseline / "report.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    cache = torch.load(args.run / "causal_features.pt", map_location="cpu", weights_only=False)
    ckpt = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    bank = torch.cat((cache["features"], cache["augmented_features"]), dim=1)
    n, v, t, d = bank.shape
    model = TrajectoryModel(ckpt["width"], ckpt["hidden"], ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    mask = cache["valid"][:, None].expand(n, v, t).reshape(n*v, t)
    times = cache["timestamps"][:, None].expand(n, v, t).reshape(n*v, t)
    with torch.no_grad():
        z = model(bank.reshape(n*v, t, d), mask, times)["embedding"]
    y = cache["labels"].repeat_interleave(v)
    cosine = z.double() @ z.double().T
    same = y[:, None] == y[None, :]
    positive = same & ~torch.eye(n*v, dtype=torch.bool)
    negative = ~same
    predictions = []
    for i in range(n):
        use = torch.arange(n) != i
        grouped = z.reshape(n, v, -1)
        proto = torch.stack([F.normalize(grouped[use & (cache["labels"] == c)].mean((0, 1)), dim=0)
                             for c in (0, 1)])
        predictions.append((grouped[i] @ proto.T).argmax(1))
    accuracy = float((torch.stack(predictions) == cache["labels"][:, None]).float().mean())
    metrics = dict(count_source_videos=n, count_views_per_source=v, count_evaluated_views=n*v,
        minimum_positive_cosine=float(cosine[positive].min()),
        maximum_negative_cosine=float(cosine[negative].max()),
        source_excluded_prototype_accuracy=accuracy,
        protocol="All training-bank views; source excluded from prototypes only, not from representation training")
    result = dict(baseline_final=baseline["final"], augmented_final=report["final"], all_bank_views=metrics,
                  augmentation=report["thermal_augmentation_policy"])
    (args.run / "comparison.json").write_text(json.dumps(result, indent=2))
    torch.save(dict(embeddings=z, labels=y, cosine=cosine), args.run / "all_view_embeddings.pt")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for r, title in ((baseline, "Original + mirror"), (report, "Thermal augmentation bank")):
        e = [row["epoch"] for row in r["history"]]
        axes[0, 0].semilogy(e, [row["semantic_loss"] for row in r["history"]], label=title)
        axes[0, 1].semilogy(e, [row["total_loss"] for row in r["history"]], label=title)
        axes[1, 0].plot(e, [row["minimum_pair_separation"] for row in r["history"]], label=title)
    for ax, title in zip((axes[0, 0], axes[0, 1], axes[1, 0]),
                         ("ViCo Eq.10 class loss", "Joint loss", "Minimum per-anchor class gap")):
        ax.set_title(title)
        ax.set_xlabel("Epoch / full-batch update")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    im = axes[1, 1].imshow(cosine.numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    boundary = int((y == 0).sum()) - .5
    axes[1, 1].axhline(boundary, color="black", linewidth=.6)
    axes[1, 1].axvline(boundary, color="black", linewidth=.6)
    axes[1, 1].set_title(f"All {n*v} training-bank views: cosine similarity")
    axes[1, 1].set_xlabel("Clean views first; spill views second")
    fig.colorbar(im, ax=axes[1, 1], shrink=.8)
    fig.suptitle("Thermal augmentation experiment: training fit; no independent scene test")
    fig.savefig(args.run / "comparison.png", dpi=170)
    fig.savefig(args.run / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
