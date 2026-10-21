#!/usr/bin/env python3
"""Fixed-budget joint trajectory optimization on genuine causal video features.

This is an optimization study on all available unique videos, not an independent
test of generalization. One epoch is one full-batch update over both views.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from vision_model.training.distributed import (initialize_distributed, unpadded_sampler_indices,
                                               audit_distributed_indices)
from vision_model.training.trajectory import TrajectoryModel, DistributedTrajectoryStep
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA
from vision_model.data.video_dataset import read_manifest


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
    p.add_argument("--resume", action="store_true",
                   help="Continue this output directory from its last completed checkpoint epoch")
    args = p.parse_args()
    if args.epochs < 1 or args.curriculum_epochs < 1:
        p.error("Epoch counts must be positive")
    checkpoint_path = args.output / "checkpoint.pt"
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"No resumable checkpoint at {checkpoint_path}")
    if not args.resume and args.output.exists():
        raise FileExistsError(f"Experiment output already exists: {args.output}")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    context = initialize_distributed()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache_sha256 = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache.get("temporal_context_policy") != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Future prediction requires genuine past-only encoder features")
    if len(set(cache["video_sha256"])) != len(cache["ids"]):
        raise ValueError("Byte-duplicate videos are not permitted")
    manifest = Path(cache["manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != cache["manifest_sha256"]:
        raise ValueError("Cache manifest changed")
    records = read_manifest(manifest)
    by_id = {record.video_id: record for record in records}
    if cache.get("label_schema") != LABEL_SCHEMA or cache.get("label_map") != LABEL_MAP:
        raise ValueError("Cache label schema is missing or obsolete; rebuild features")
    if set(cache["ids"]) != set(by_id) or len(cache["ids"]) != len(records):
        raise ValueError("Cache source IDs do not match the manifest")
    if cache.get("categories") != [by_id[video_id].category for video_id in cache["ids"]]:
        raise ValueError("Cache categories do not match the manifest")
    if cache["labels"].tolist() != [by_id[video_id].label for video_id in cache["ids"]]:
        raise ValueError("Cache labels do not match the manifest")
    x, valid, times, labels = (cache[k] for k in ("features", "valid", "timestamps", "labels"))
    class_counts = torch.bincount(labels, minlength=2)
    if len(class_counts) != 2 or class_counts[0] != class_counts[1] or class_counts[0] < 2:
        raise ValueError("Training needs equal numbers of distinct clean and spill sources; prepare a balanced manifest")
    bank = (torch.cat((x, cache["augmented_features"]), dim=1)
            if "augmented_features" in cache else x)
    if bank.shape[0] != x.shape[0] or bank.shape[2:] != x.shape[2:]:
        raise ValueError("Augmentation bank does not match evaluation trajectories")
    if len(labels) < context.world_size:
        raise ValueError("Every distributed rank needs at least one source video")
    train_sampler = DistributedSampler(range(len(labels)), num_replicas=context.world_size,
                                       rank=context.rank, shuffle=True, seed=args.seed,
                                       drop_last=False)
    eval_sampler = DistributedSampler(range(len(labels)), num_replicas=context.world_size,
                                      rank=context.rank, shuffle=False, drop_last=False)
    eval_sampler.set_epoch(0)
    eval_indices, _ = unpadded_sampler_indices(eval_sampler, len(labels))
    eval_order, eval_rank_counts = audit_distributed_indices(eval_indices, len(labels))
    eval_x, eval_valid, eval_times, eval_labels = (
        tensor[eval_indices].to(context.device) for tensor in (x, valid, times, labels))
    all_eval_labels = labels[eval_order].to(context.device)
    view_exposures = torch.zeros(len(labels), bank.shape[1], dtype=torch.long,
                                 device=context.device)
    rank_source_counts, padding_filtered_counts = [], []
    model = TrajectoryModel(x.shape[-1]).to(context.device)
    step = DistributedTrajectoryStep(model)
    if context.world_size > 1:
        step = DistributedDataParallel(step,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False)
    optimizer = torch.optim.AdamW(step.parameters(), lr=args.learning_rate, weight_decay=0, eps=1e-12)
    weights = dict(curriculum_epochs=args.curriculum_epochs, align_weight=args.align_weight,
                   predict_weight=args.predict_weight, keep_weight=args.keep_weight)
    history = []
    previous_exposures = torch.zeros_like(view_exposures)
    resume_settings = dict(cache_sha256=cache_sha256,
        manifest_sha256=cache["manifest_sha256"], video_ids=cache["ids"],
        world_size=context.world_size, seed=args.seed, learning_rate=args.learning_rate,
        curriculum_epochs=args.curriculum_epochs, align_weight=args.align_weight,
        predict_weight=args.predict_weight, keep_weight=args.keep_weight)
    completed_epoch = 0
    resumed_from_epoch = None
    resume_count = 0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("format") != "trajectory_resumable_v1":
            raise ValueError("Checkpoint predates resumable training; start a new output directory")
        if saved.get("resume_settings") != resume_settings:
            raise ValueError("Checkpoint data, world size, or training settings differ from this run")
        completed_epoch = int(saved["epochs_completed"])
        if args.epochs < completed_epoch:
            raise ValueError(f"--epochs cannot be below completed epoch {completed_epoch}; it is the total target")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        previous_exposures = saved["training_view_exposures"].to(context.device)
        if previous_exposures.shape != view_exposures.shape:
            raise ValueError("Checkpoint view exposure shape differs from the feature bank")
        rank_state = saved["rank_rng_states"][context.rank]
        torch.set_rng_state(rank_state["torch"])
        if context.device.type == "cuda":
            torch.cuda.set_rng_state(rank_state["cuda"], device=context.device)
        rank_source_counts = saved["rank_source_counts_per_epoch"] if context.is_main else []
        padding_filtered_counts = saved["sampler_padding_filtered_per_epoch"] if context.is_main else []
        history = saved["history"] if context.is_main else []
        if context.is_main and (len(history) != completed_epoch + 1 or
                                len(rank_source_counts) != completed_epoch or
                                len(padding_filtered_counts) != completed_epoch):
            raise ValueError("Checkpoint history is incomplete")
        resumed_from_epoch = completed_epoch
        resume_count = int(saved.get("resume_count", 0)) + 1

    def global_metrics(metrics):
        names = list(metrics)
        values = torch.stack([metrics[name].detach().double() for name in names])
        if context.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= context.world_size
        return {name: float(value) for name, value in zip(names, values)}

    @torch.no_grad()
    def evaluate(epoch, *, record=True):
        step.eval()
        # Keep reference weighting fixed across epochs so this curve is comparable.
        _, metrics, out = step(eval_x, eval_valid, eval_times, eval_labels,
                               epoch=args.curriculum_epochs, **weights)
        row = {"epoch": epoch, **global_metrics(metrics)}
        z = out["embedding"].reshape(len(labels), 2, -1)[:, 0]
        prototypes = torch.stack([z[all_eval_labels == c].mean(0) for c in (0, 1)])
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        accuracy = ((z @ prototypes.T).argmax(1) == all_eval_labels).float().mean()
        if context.world_size > 1:
            dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
            accuracy /= context.world_size
        row["training_prototype_accuracy"] = float(accuracy)
        if context.is_main and record:
            history.append(row)
        return row, out

    def save_progress(epoch):
        """Commit one complete update; rank zero replaces the checkpoint atomically."""
        local_rng = {"torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state(context.device)
                             if context.device.type == "cuda" else None}
        if context.world_size > 1:
            rank_rng_states = [None] * context.world_size if context.is_main else None
            dist.gather_object(local_rng, rank_rng_states, dst=0)
            exposure_total = view_exposures.clone()
            dist.all_reduce(exposure_total, op=dist.ReduceOp.SUM)
        else:
            rank_rng_states = [local_rng]
            exposure_total = view_exposures.clone()
        if context.is_main:
            payload = dict(format="trajectory_resumable_v1", model=model.state_dict(),
                optimizer=optimizer.state_dict(), width=x.shape[-1], hidden=128,
                embedding_dim=128, epochs_completed=epoch, optimizer_updates=epoch,
                resume_settings=resume_settings, resume_count=resume_count,
                rank_rng_states=rank_rng_states, history=history,
                rank_source_counts_per_epoch=rank_source_counts,
                sampler_padding_filtered_per_epoch=padding_filtered_counts,
                training_view_exposures=(previous_exposures + exposure_total).cpu(),
                cache_metadata={k: v for k, v in cache.items() if not isinstance(v, torch.Tensor)})
            temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
        if context.world_size > 1:
            dist.barrier()

    if not args.resume:
        if context.is_main:
            args.output.mkdir(parents=True, exist_ok=False)
        if context.world_size > 1:
            dist.barrier()
        evaluate(0)
    elif args.epochs == completed_epoch:
        # The final update may have been committed before report generation was interrupted.
        _, final_outputs = evaluate(completed_epoch, record=False)
        save_progress(completed_epoch)
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_indices, _ = unpadded_sampler_indices(train_sampler, len(labels))
        _, per_rank_counts = audit_distributed_indices(train_indices, len(labels))
        if context.is_main:
            rank_source_counts.append(per_rank_counts)
            padding_filtered_counts.append(sum(len(train_sampler) - count
                                               for count in per_rank_counts))
        if epoch == args.curriculum_epochs + 1:
            # A single documented moment reset after mining curriculum stabilizes.
            optimizer = torch.optim.AdamW(step.parameters(), lr=args.learning_rate,
                                           weight_decay=0, eps=1e-12)
        step.train()
        optimizer.zero_grad(set_to_none=True)
        training_x, chosen_views = (select_training_views(bank, seed=args.seed, epoch=epoch)
                                    if "augmented_features" in cache else (x, torch.arange(2).expand(len(x), -1)))
        indices = torch.as_tensor(train_indices, dtype=torch.long)
        local_views = chosen_views[indices].to(context.device)
        view_exposures.index_put_(
            (indices[:, None].to(context.device).expand_as(local_views), local_views),
            torch.ones_like(local_views), accumulate=True)
        local_x, local_valid, local_times, local_labels = (
            tensor[indices].to(context.device) for tensor in (training_x, valid, times, labels))
        loss, _, _ = step(local_x, local_valid, local_times, local_labels,
                          epoch=epoch, **weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(step.parameters(), 5.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
        optimizer.step()
        row, final_outputs = evaluate(epoch)
        training_statistics = torch.stack((loss.detach().double(), norm.detach().double()))
        if context.world_size > 1:
            dist.all_reduce(training_statistics, op=dist.ReduceOp.SUM)
            training_statistics /= context.world_size
        if context.is_main:
            row["scheduled_loss_before_update"] = float(training_statistics[0])
            row["gradient_norm_before_clipping"] = float(training_statistics[1])
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({k: row[k] for k in ("epoch", "total_loss", "semantic_loss",
                      "prediction_loss", "retention_loss", "minimum_pair_separation")}), flush=True)
        save_progress(epoch)
    final_exposures = view_exposures.clone()
    if context.world_size > 1:
        dist.all_reduce(final_exposures, op=dist.ReduceOp.SUM)
    if context.is_main:
        with (args.output / "history.csv").open("w") as f:
            keys = list(history[-1])
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(history)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        report = dict(settings=settings, protocol="all-unique-video optimization study; no held-out test",
            epochs_completed=args.epochs, optimizer_updates=args.epochs, seed=args.seed,
            resumed_from_epoch=resumed_from_epoch, resume_count=resume_count,
            distributed_world_size=context.world_size, device=str(context.device),
            optimizer_reset_before_update=(args.curriculum_epochs + 1
                                           if args.epochs > args.curriculum_epochs else None),
            count_videos=len(labels), class_counts=torch.bincount(labels).tolist(), views_per_video=2,
            rank_source_counts_per_epoch=rank_source_counts,
            sampler_padding_filtered_per_epoch=padding_filtered_counts,
            evaluation_rank_source_counts=eval_rank_counts,
            evaluation_sampler_padding_filtered=context.world_size * len(eval_sampler) - len(labels),
            distributed_partition_policy="DistributedSampler global rank/world size; padded repeats removed; all sources audited once per epoch",
            loss=f"ViCo-LWIR Eq.10 + {args.align_weight} alignment + {args.predict_weight} prediction + {args.keep_weight} retention",
            curve_policy="post-update; fixed final-curriculum reference weights at every epoch",
            temporal_context_policy=cache["temporal_context_policy"],
            cache_sha256=cache_sha256,
            normalization=cache["normalization"], ids=cache["ids"],
            thermal_augmentation_policy=cache.get("thermal_augmentation_policy"),
            training_views_in_bank=bank.shape[1],
            training_view_exposures=(previous_exposures + final_exposures).cpu().tolist(),
            training_view_policy="two distinct seeded views per source per epoch; finite pre-encoded bank",
            evaluation_views="fixed identity and whole-clip horizontal flip",
            initial=history[0], final=history[-1], history=history)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        order = torch.argsort(torch.as_tensor(eval_order, device=context.device))
        ordered_outputs = {name: value.reshape(len(labels), 2, *value.shape[1:])[order]
                           .reshape(2 * len(labels), *value.shape[1:]).cpu()
                           for name, value in final_outputs.items()}
        torch.save(dict(**ordered_outputs, labels=labels, valid=valid, timestamps=times,
                        ids=cache["ids"]), args.output / "representations.pt")
        plot_history(history, args.output, len(labels), args.epochs)
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def plot_history(history, output, count_videos, count_epochs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_history_pillow(history, output, count_videos, count_epochs)
        return
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


def _plot_history_pillow(history, output, count_videos, count_epochs):
    """Keep the numerical report visible on cluster environments without matplotlib."""
    import math
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 12), f"{count_epochs} epochs | {count_videos} fitted source videos",
              fill="black")
    panels = [
        ("Joint and semantic loss", (("total_loss", "red"), ("semantic_loss", "blue")), True),
        ("Future prediction and persistence", (("prediction_loss", "red"),
                                               ("persistence_loss", "blue")), True),
        ("Alignment and retention", (("alignment_loss", "red"),
                                     ("retention_loss", "blue")), True),
        ("Class cosine bounds", (("minimum_positive_cosine", "blue"),
                                 ("maximum_negative_cosine", "red")), False),
    ]
    for panel, (title, series, logarithmic) in enumerate(panels):
        column, row = panel % 2, panel // 2
        x0, y0 = 65 + column * 590, 65 + row * 380
        x1, y1 = x0 + 505, y0 + 285
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
        draw.text((x0, y0 - 25), title, fill="black")
        plotted = []
        for name, color in series:
            values = [(math.log10(float(item[name])) if logarithmic else float(item[name]))
                      if float(item[name]) > 0 or not logarithmic else float("nan")
                      for item in history]
            plotted.append((name, color, values))
        finite = [value for _, _, values in plotted for value in values if math.isfinite(value)]
        low, high = min(finite), max(finite)
        if high == low:
            high = low + 1
        first, last = history[0]["epoch"], history[-1]["epoch"]
        span = max(1, last - first)
        for name, color, values in plotted:
            points = [(x0 + (item["epoch"] - first) * (x1 - x0) / span,
                       y1 - (value - low) * (y1 - y0) / (high - low))
                      for item, value in zip(history, values) if math.isfinite(value)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            elif points:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text((x0, y1 + 12 + 16 * list(series).index((name, color))),
                      name.replace("_", " "), fill=color)
        draw.text((x1 - 105, y1 + 12), f"epoch {last}", fill="black")
    destination = output / "curves.png"
    image.save(destination)
    print(f"Saved {destination} (Pillow plot; matplotlib unavailable)", flush=True)


if __name__ == "__main__":
    main()
