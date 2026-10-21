#!/usr/bin/env python3
"""Fixed-budget joint trajectory optimization on genuine causal video features.

This is an optimization study on all available unique videos, not an independent
test of generalization. One epoch is one full-batch update over both views.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vision_model.training.trajectory import TrajectoryModel, trajectory_loss


def select_training_views(bank, *, seed, epoch):
    """Two distinct views for each source; repeatable but different each epoch."""
    if bank.ndim != 4 or bank.shape[1] < 2:
        raise ValueError("A training bank needs at least two views per video")
    generator = torch.Generator().manual_seed(seed + epoch * 104729)
    indices = torch.stack([torch.randperm(bank.shape[1], generator=generator)[:2]
                           for _ in range(len(bank))])
    return bank[torch.arange(len(bank))[:, None], indices], indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "reports/trajectory_100")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=239)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--curriculum-epochs", type=int, default=20)
    p.add_argument("--align-weight", type=float, default=.1)
    p.add_argument("--predict-weight", type=float, default=1.)
    p.add_argument("--keep-weight", type=float, default=.1)
    args = p.parse_args()
    if args.epochs < 1 or args.curriculum_epochs < 1:
        p.error("Epoch counts must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("temporal_context_policy") != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Future prediction requires genuine past-only encoder features")
    if len(set(cache["video_sha256"])) != len(cache["ids"]):
        raise ValueError("Byte-duplicate videos are not permitted")
    manifest = Path(cache["manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != cache["manifest_sha256"]:
        raise ValueError("Cache manifest changed")
    x, valid, times, labels = (cache[k] for k in ("features", "valid", "timestamps", "labels"))
    bank = (torch.cat((x, cache["augmented_features"]), dim=1)
            if "augmented_features" in cache else x)
    if bank.shape[0] != x.shape[0] or bank.shape[2:] != x.shape[2:]:
        raise ValueError("Augmentation bank does not match evaluation trajectories")
    view_exposures = torch.zeros(len(labels), bank.shape[1], dtype=torch.long)
    model = TrajectoryModel(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0, eps=1e-12)
    weights = dict(curriculum_epochs=args.curriculum_epochs, align_weight=args.align_weight,
                   predict_weight=args.predict_weight, keep_weight=args.keep_weight)
    history = []

    @torch.no_grad()
    def evaluate(epoch):
        model.eval()
        # Keep reference weighting fixed across epochs so this curve is comparable.
        _, metrics, out = trajectory_loss(model, x, valid, times, labels,
                                          epoch=args.curriculum_epochs, **weights)
        row = {"epoch": epoch, **{k: float(v) for k, v in metrics.items()}}
        z = out["embedding"].reshape(len(labels), 2, -1)[:, 0]
        prototypes = torch.stack([z[labels == c].mean(0) for c in (0, 1)])
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        row["training_prototype_accuracy"] = float(((z @ prototypes.T).argmax(1) == labels).float().mean())
        history.append(row)
        return row

    evaluate(0)
    for epoch in range(1, args.epochs + 1):
        if epoch == args.curriculum_epochs + 1:
            # A single documented moment reset after mining curriculum stabilizes.
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                           weight_decay=0, eps=1e-12)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_x, chosen_views = (select_training_views(bank, seed=args.seed, epoch=epoch)
                                    if "augmented_features" in cache else (x, torch.arange(2).expand(len(x), -1)))
        view_exposures.scatter_add_(1, chosen_views, torch.ones_like(chosen_views))
        loss, _, _ = trajectory_loss(model, training_x, valid, times, labels,
                                     epoch=epoch, **weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
        optimizer.step()
        row = evaluate(epoch)
        row["scheduled_loss_before_update"] = float(loss.detach())
        row["gradient_norm_before_clipping"] = float(norm)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({k: row[k] for k in ("epoch", "total_loss", "semantic_loss",
                  "prediction_loss", "retention_loss", "minimum_pair_separation")}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "history.csv").open("w") as f:
        keys = list(history[-1])
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(history)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(settings=settings, protocol="all-unique-video optimization study; no held-out test",
        epochs_completed=args.epochs, optimizer_updates=args.epochs, seed=args.seed,
        count_videos=len(labels), class_counts=torch.bincount(labels).tolist(), views_per_video=2,
        loss=f"ViCo-LWIR Eq.10 + {args.align_weight} alignment + {args.predict_weight} prediction + {args.keep_weight} retention",
        curve_policy="post-update; fixed final-curriculum reference weights at every epoch",
        temporal_context_policy=cache["temporal_context_policy"],
        cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        normalization=cache["normalization"], ids=cache["ids"],
        thermal_augmentation_policy=cache.get("thermal_augmentation_policy"),
        training_views_in_bank=bank.shape[1],
        training_view_exposures=view_exposures.tolist(),
        training_view_policy="two distinct seeded views per source per epoch; finite pre-encoded bank",
        evaluation_views="fixed identity and whole-clip horizontal flip",
        initial=history[0], final=history[-1], history=history)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(dict(model=model.state_dict(), width=x.shape[-1], hidden=128, embedding_dim=128,
                    settings=settings, cache_metadata={k: v for k, v in cache.items()
                        if not isinstance(v, torch.Tensor)},
                    optimizer=optimizer.state_dict(), epochs_completed=args.epochs), args.output / "checkpoint.pt")
    model.eval()
    with torch.no_grad():
        _, _, outputs = trajectory_loss(model, x, valid, times, labels, epoch=args.curriculum_epochs, **weights)
    torch.save(dict(**outputs, labels=labels, valid=valid, timestamps=times, ids=cache["ids"]),
               args.output / "representations.pt")
    plot_history(history, args.output, len(labels), args.epochs)


def plot_history(history, output, count_videos, count_epochs):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, title in (("total_loss", "Joint loss"), ("semantic_loss", "ViCo-LWIR Eq. 10 class loss")):
        ax = axes[0, 0 if key == "total_loss" else 1]
        ax.semilogy(epochs, [r[key] for r in history])
        ax.set_title(title)
        ax.set_ylabel("Loss (log scale)")
    ax = axes[1, 0]
    for key, title in (("alignment_loss", "View alignment"), ("prediction_loss", "Future prediction"),
                       ("retention_loss", "Teacher retention"), ("persistence_loss", "Frozen persistence baseline")):
        # Zero retention at initialization is exact; do not invent a log floor.
        ax.semilogy(epochs, [r[key] if r[key] > 0 else float("nan") for r in history], label=title)
    ax.legend(fontsize=8)
    ax.set_title("Unweighted temporal components")
    ax.text(.02, .03, "Retention = 0 at initialization", transform=ax.transAxes, fontsize=8)
    ax = axes[1, 1]
    for key, title in (("minimum_positive_cosine", "Minimum same-class cosine"),
                       ("maximum_negative_cosine", "Maximum cross-class cosine")):
        ax.plot(epochs, [r[key] for r in history], label=title)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Worst-pair class separation")
    for ax in axes.flat:
        ax.set_xlabel("Epoch (one full-batch update)")
        ax.grid(alpha=.25)
    fig.suptitle(f"{count_epochs} epochs · {count_videos} unique real videos · training optimization only")
    fig.savefig(output / "curves.png", dpi=180)
    fig.savefig(output / "curves.pdf")
    plt.close(fig)
    print(f"Saved {output / 'curves.png'}", flush=True)


if __name__ == "__main__":
    main()
