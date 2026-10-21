#!/usr/bin/env python3
"""Reproducible long-run, frozen-encoder loss ablations on cached real features.

All-data runs diagnose optimization, not generalization. Every variant uses the
same features, head initialization and batch order for each seed. Final-phase
variance calibration freezes the mean branch so its regression targets stop
moving. Nothing subtracts a constant from the optimized loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulate_loss_training import balanced_batches, evaluate, gaussian_output
from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss
from vision_model.training.optimizer import parameter_groups
from vision_model.data.labels import validate_feature_cache


class CachedHeads(nn.Module):
    def __init__(self, hidden: int, latent: int, variance_floor: float):
        super().__init__()
        self.variance_floor = variance_floor
        self.mean_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.variance_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))


def grad_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


@torch.no_grad()
def diagnostics(model, features, valid):
    out = gaussian_output(model, features, valid)
    r2 = (out.temporal_mean - out.video_mean[:, None]).square()[valid]
    v = out.temporal_variance[valid]
    optimum = r2.clamp_min(model.variance_floor)
    regret = r2 / v + v.log() - (r2 / optimum + optimum.log())
    return {
        "uncertainty_regret": float(regret.mean()),
        "uncertainty_optimum": float((r2 / optimum + optimum.log()).mean()),
        "video_mean_norm": float(out.video_mean.norm(dim=-1).mean()),
        "log_variance_optimum_rmse": float((v.log() - optimum.log()).square().mean().sqrt()),
    }


def slope(values):
    x = torch.arange(len(values), dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / x.square().sum())


@torch.no_grad()
def evaluate_fixed_batches(model, objective, features, valid, labels, ids, epoch):
    """End-of-epoch evaluation on fixed six-video batches, including refinement."""
    total, count = 0., 0
    for indices in balanced_batches(labels.cpu(), epoch=0):
        indices = indices.to(features.device)
        result = objective(output=gaussian_output(model, features[indices], valid[indices]),
                           labels=labels[indices], original_ids=[ids[i] for i in indices.tolist()], epoch=epoch)
        total += float(result.loss_sum)
        count += result.sample_count
    return total / count


def run(cache, args, variant, seed):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    features, valid, labels = (cache[k].to(device) for k in ("features", "valid", "labels"))
    ids = cache["ids"]
    latent = cache["mean_head"]["3.bias"].numel()
    model = CachedHeads(features.shape[-1], latent, cache["variance_floor"]).to(device)
    # Reproduce the original initialized heads for seed 239; other seeds are
    # independent initializations, identical across variants within that seed.
    if seed == cache["seed"]:
        model.mean_head.load_state_dict(cache["mean_head"])
        model.variance_head.load_state_dict(cache["variance_head"])
    adaptive = variant == "adaptive_refinement"
    refinement = variant in ("variance_refinement", "adaptive_refinement")
    baseline = variant == "constant_lr" or refinement
    calibrate = variant == "scheduled_calibration" or refinement
    config = GaussianContrastiveLossConfig(total_epochs=args.curriculum_epochs,
                                           pair_weighting=variant != "plain_supcon")
    objective = StableUncertaintyContrastiveLoss(config)
    groups = list(model.parameters()) if baseline else parameter_groups(model, 0.04)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.04)
    mean_params, variance_params = list(model.mean_head.parameters()), list(model.variance_head.parameters())
    joint_epochs = args.epochs - args.calibration_epochs if calibrate else args.epochs
    if adaptive:
        joint_epochs = args.epochs  # Set once the fixed-curriculum fit stabilizes.
    stable_epochs = 0
    history = []
    frozen_mean_state = None
    start_epoch = 0
    if args.resume_dir is not None:
        if variant != "variance_refinement":
            raise ValueError("Resume currently supports variance_refinement only")
        checkpoint = torch.load(args.resume_dir / f"{variant}_seed{seed}.pt", map_location=device, weights_only=True)
        previous = json.loads((args.resume_dir / f"{variant}_seed{seed}.json").read_text())
        old_joint = previous["settings"]["epochs"] - previous["settings"]["calibration_epochs"]
        if (checkpoint["feature_cache_sha256"] != args.cache_sha256 or old_joint != joint_epochs
                or checkpoint["loss_config"] != config.__dict__ or checkpoint["epoch"] < joint_epochs
                or previous["settings"].get("refinement_max_iter", 10) != args.refinement_max_iter
                or previous["settings"].get("refinement_precision", "float32") != args.refinement_precision
                or args.epochs <= checkpoint["epoch"]):
            raise ValueError("Resume requires identical features/loss/joint phase/refinement settings and a longer calibration phase")
        model.load_state_dict(checkpoint["model"])
        model.mean_head.requires_grad_(False)
        frozen_mean_state = {k: v.detach().clone() for k, v in model.mean_head.state_dict().items()}
        optimizer = torch.optim.LBFGS(variance_params, lr=1., max_iter=args.refinement_max_iter, history_size=10,
                                     line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-10)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
        history = previous["history"]
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, args.epochs + 1):
        training = {}
        if epoch:
            e = epoch - 1
            calibration_phase = calibrate and e >= joint_epochs
            if calibration_phase:
                model.mean_head.requires_grad_(False)
                model.zero_grad(set_to_none=True)
                if frozen_mean_state is None:
                    frozen_mean_state = {k: v.detach().clone() for k, v in model.mean_head.state_dict().items()}
            if refinement and e == joint_epochs:
                optimizer = torch.optim.LBFGS(variance_params, lr=1., max_iter=args.refinement_max_iter,
                                             history_size=10, line_search_fn="strong_wolfe",
                                             tolerance_grad=1e-8, tolerance_change=1e-10)
            if baseline:
                lr = args.lr
            else:
                progress = ((e - joint_epochs) / max(args.calibration_epochs - 1, 1)
                            if calibration_phase else e / max(joint_epochs - 1, 1))
                lr = args.final_lr + (args.lr - args.final_lr) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals = dict(loss=0., contrastive=0., entropy=0., uncertainty=0.)
            count = 0
            mean_grads, variance_grads = [], []
            # Epoch/seed dependent ordering, shared between all variants.
            batch_indices = ([torch.arange(len(labels))] if refinement and calibration_phase
                             else balanced_batches(labels.cpu(), epoch=e + seed - 239))
            closure_evaluations = 0
            for indices in batch_indices:
                indices = indices.to(device)
                if refinement and calibration_phase:
                    optimizer.param_groups[0]["lr"] = 1.
                    def closure():
                        nonlocal closure_evaluations
                        optimizer.zero_grad(set_to_none=True)
                        # Means are fixed: this is a true smooth regression objective,
                        # without recomputing stop-gradient pair weights in a line search.
                        out = gaussian_output(model, features, valid)
                        # A float32 scalar near -5 can hide genuine improvements
                        # of order 1e-7 from a line search. Keep network arithmetic
                        # in float32 but evaluate this smooth objective in float64.
                        dtype = getattr(torch, args.refinement_precision)
                        residual = (out.temporal_mean - out.video_mean[:, None]).detach().to(dtype)
                        variance = out.temporal_variance.to(dtype)
                        elements = residual.square() / variance + variance.log()
                        nll = ((elements * valid[..., None]).sum((1, 2)) /
                               (valid.sum(1) * elements.shape[-1])).mean()
                        nll.backward()
                        closure_evaluations += 1
                        return nll
                    optimizer.step(closure)
                output = gaussian_output(model, features[indices], valid[indices])
                result = objective(output=output, labels=labels[indices],
                                   original_ids=[ids[i] for i in indices.tolist()], epoch=e)
                if not (refinement and calibration_phase):
                    optimizer.zero_grad(set_to_none=True)
                    (result.loss_sum / result.sample_count).backward()
                mean_grads.append(grad_norm(mean_params))
                variance_grads.append(grad_norm(variance_params))
                if not (refinement and calibration_phase):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                totals["loss"] += float(result.loss_sum.detach())
                totals["contrastive"] += float(result.contrastive_sum)
                totals["entropy"] += float(result.positive_entropy_sum)
                totals["uncertainty"] += float(result.uncertainty_sum)
                count += result.sample_count
            training = {
                "training_loss": totals["loss"] / count,
                "batch_objective_scope": "full_dataset_post_update" if refinement and calibration_phase else "balanced_minibatches_pre_update",
                "training_contrastive_excess": (totals["contrastive"] - totals["entropy"]) / count,
                "mean_gradient_norm": sum(mean_grads) / len(mean_grads),
                "variance_gradient_norm": sum(variance_grads) / len(variance_grads),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "closure_evaluations": closure_evaluations,
                "phase": "variance_calibration" if calibration_phase else "joint",
            }
        # Keep evaluation curriculum fixed so curves do not mix changing schedules.
        metrics = evaluate(model, objective, features, valid, labels, ids, args.curriculum_epochs)
        metrics["eval_full_dataset_loss"] = metrics["loss"]
        metrics["eval_fixed_minibatch_loss"] = evaluate_fixed_batches(
            model, objective, features, valid, labels, ids, args.curriculum_epochs)
        metrics.update(diagnostics(model, features, valid))
        row = dict(epoch=epoch, **metrics, **training)
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"Nonfinite diagnostic: {variant}, seed {seed}, epoch {epoch}")
        history.append(row)
        if adaptive and joint_epochs == args.epochs and epoch >= args.curriculum_epochs + 1:
            stable_epochs = stable_epochs + 1 if row["contrastive_excess_loss"] < args.refinement_threshold else 0
            if stable_epochs >= args.refinement_patience:
                joint_epochs = epoch
                print(json.dumps({"variant": variant, "seed": seed, "freeze_mean_after_epoch": epoch}), flush=True)
        if epoch % 20 == 0 or epoch == args.epochs:
            print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch,
                              "excess": row["contrastive_excess_loss"],
                              "uncertainty_regret": row["uncertainty_regret"]}), flush=True)
    final = history[-1]
    tail = history[-10:]
    assessment = {
        "contrastive_near_floor": final["contrastive_excess_loss"] < 1e-3,
        "uncertainty_near_fixed_residual_optimum": final["uncertainty_regret"] < 1e-3,
        "mean_unchanged_during_calibration": (all(torch.equal(v, model.mean_head.state_dict()[k])
                                                  for k, v in frozen_mean_state.items())
                                             if frozen_mean_state is not None else None),
        "last_ten_excess_slope": slope([r["contrastive_excess_loss"] for r in tail]),
        "last_ten_uncertainty_regret_slope": slope([r["uncertainty_regret"] for r in tail]),
        "uncertainty_regret_reduction": 1 - final["uncertainty_regret"] / history[0]["uncertainty_regret"],
        "note": "Near-floor does not establish uncertainty optimum or held-out accuracy.",
        "joint_epochs": joint_epochs,
    }
    assessment["training_fit_converged"] = bool(
        assessment["contrastive_near_floor"] and assessment["uncertainty_near_fixed_residual_optimum"]
        and abs(assessment["last_ten_excess_slope"]) < 1e-5
        and abs(assessment["last_ten_uncertainty_regret_slope"]) < 1e-4)
    path = args.output / f"{variant}_seed{seed}.json"
    payload = dict(variant=variant, seed=seed, epochs=args.epochs, loss_config=config.__dict__,
                   label_schema=cache.get("label_schema"), label_map=cache.get("label_map"),
                   class_counts={str(c): int((labels == c).sum()) for c in (0, 1)},
                   feature_cache_sha256=args.cache_sha256, settings=vars(args).copy(),
                   encoder_padding_policy=cache.get("encoder_padding_policy", "unmasked_legacy"),
                   assessment=assessment, history=history)
    payload["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in payload["settings"].items()}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    label_schema=cache.get("label_schema"), label_map=cache.get("label_map"),
                    epoch=args.epochs, loss_config=config.__dict__, seed=seed,
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    feature_cache_sha256=args.cache_sha256, settings=payload["settings"],
                    assessment=assessment), args.output / f"{variant}_seed{seed}.pt")
    return dict(variant=variant, seed=seed, label_schema=cache.get("label_schema"),
                label_map=cache.get("label_map"), class_counts=payload["class_counts"], **assessment, final=final)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "checkpoints/frozen_features.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/convergence_120_epochs")
    parser.add_argument("--resume-dir", type=Path, help="Extend an existing variance-refinement calibration phase")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--calibration-epochs", type=int, default=40)
    parser.add_argument("--curriculum-epochs", type=int, default=19)
    parser.add_argument("--seeds", type=int, nargs="+", default=[239, 240, 241])
    parser.add_argument("--variants", nargs="+", choices=["constant_lr", "scheduled", "scheduled_calibration", "plain_supcon", "variance_refinement", "adaptive_refinement"],
                        default=["constant_lr", "scheduled", "scheduled_calibration", "plain_supcon"])
    parser.add_argument("--lr", type=float, default=5.25e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--refinement-max-iter", type=int, default=10)
    parser.add_argument("--refinement-precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--refinement-threshold", type=float, default=1e-3)
    parser.add_argument("--refinement-patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.epochs < 10 or not 0 < args.calibration_epochs < args.epochs:
        parser.error("Require epochs >= 10 and 0 < calibration-epochs < epochs")
    if args.refinement_max_iter < 1 or args.refinement_patience < 1 or not 0 < args.refinement_threshold < float("inf"):
        parser.error("Refinement iteration count, patience and finite threshold must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.cache_sha256 = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        for variant in args.variants:
            summary.append(run(cache, args, variant, seed))
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
