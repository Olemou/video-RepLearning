#!/usr/bin/env python3
"""Budgeted DDP search of trajectory learning rate and global source batch size.

Run with torch.distributed.run on a causal feature cache. Validation sources are
never used in optimizer updates; the validation set is for model selection only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA
from vision_model.data.video_dataset import read_manifest
from vision_model.training.distributed import (audit_distributed_indices,
                                               initialize_distributed, unpadded_sampler_indices)
from vision_model.training.lr_search import batch_plan, cache_identity, scaled_lr, source_split, warmup_cosine
from vision_model.training.trajectory import DistributedTrajectoryStep, TrajectoryModel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=239)
    p.add_argument("--validation-per-class", type=int, default=16)
    p.add_argument("--accumulation-steps", type=int, default=2)
    p.add_argument("--base-learning-rate", type=float, default=1e-3)
    p.add_argument("--coarse-source-budget", type=int, default=512)
    p.add_argument("--full-source-budget", type=int, default=1024)
    p.add_argument("--max-trials", type=int, default=19)
    p.add_argument("--warmup-fraction", type=float, default=.1)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--single-run", action="store_true",
                   help="Train one fixed batch-32 configuration for the full source budget")
    p.add_argument("--record-every-update", action="store_true",
                   help="Evaluate and record global losses after every optimizer update")
    return p


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new tuning output directory: {args.output}")
    if args.coarse_source_budget < 32 or args.full_source_budget < args.coarse_source_budget:
        raise ValueError("Full budget must be at least the coarse budget, both at least 32")
    if args.max_trials < 1 or args.weight_decay < 0:
        raise ValueError("Invalid trial budget or weight decay")
    torch.set_num_threads(4)
    context = initialize_distributed()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("temporal_context_policy") != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Tuning requires frozen past-only causal prefix features")
    if cache.get("label_map") != LABEL_MAP or cache.get("label_schema") != LABEL_SCHEMA:
        raise ValueError("Cache label mapping is missing or obsolete")
    if len(set(cache["video_sha256"])) != len(cache["ids"]):
        raise ValueError("Byte-identical source videos cannot cross splits")
    if hashlib.sha256(Path(cache["manifest"]).read_bytes()).hexdigest() != cache["manifest_sha256"]:
        raise ValueError("Feature cache manifest changed")
    records_by_id = {record.video_id: record for record in read_manifest(Path(cache["manifest"]))}
    if set(records_by_id) != set(cache["ids"]) or len(records_by_id) != len(cache["ids"]):
        raise ValueError("Feature cache source IDs differ from its manifest")
    if any(records_by_id[video_id].label != int(cache["labels"][index]) or
           records_by_id[video_id].source_sequence != cache["source_sequences"][index]
           for index, video_id in enumerate(cache["ids"])):
        raise ValueError("Feature cache labels or source groups differ from its manifest")
    x, valid, times, labels = (cache[k] for k in ("features", "valid", "timestamps", "labels"))
    if x.ndim != 4 or x.shape[1] != 2 or not torch.isfinite(x).all():
        raise ValueError("Cache must have two finite fixed evaluation views per source")
    bank = (torch.cat((x, cache["augmented_features"]), 1)
            if "augmented_features" in cache else x)
    train, validation = source_split(labels, cache["source_sequences"], seed=args.seed,
                                     validation_per_class=args.validation_per_class)
    train_classes = {label: [i for i in train if int(labels[i]) == label] for label in (0, 1)}
    max_batch = (len(train) // (2 * context.world_size * args.accumulation_steps)
                 * 2 * context.world_size * args.accumulation_steps)
    batches = sorted({size for size in (32, 64, max_batch) if size <= max_batch and
                      size % (2 * context.world_size * args.accumulation_steps) == 0})
    if not batches:
        raise ValueError("No feasible global source batch of at least 32")
    if len(validation) < context.world_size or min(map(len, train_classes.values())) < context.world_size:
        raise ValueError("Too many GPU ranks for the source-disjoint split")
    gpu = (torch.cuda.get_device_name(context.device) if context.device.type == "cuda" else "CPU")
    devices = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(devices, gpu)
    else:
        devices[0] = gpu
    if context.is_main:
        args.output.mkdir(parents=True, exist_ok=False)
    if context.world_size > 1:
        dist.barrier()

    def evaluation(step_module, epoch):
        sampler = DistributedSampler(validation, num_replicas=context.world_size,
                                     rank=context.rank, shuffle=False, drop_last=False)
        sampler.set_epoch(0)
        positions, padded = unpadded_sampler_indices(sampler, len(validation))
        audit_distributed_indices(positions, len(validation))
        selected = torch.as_tensor([validation[i] for i in positions], dtype=torch.long)
        with torch.no_grad():
            step_module.eval()
            _, metrics, _ = step_module(x[selected].to(context.device), valid[selected].to(context.device),
                                        times[selected].to(context.device), labels[selected].to(context.device),
                                        epoch=epoch, curriculum_epochs=epoch)
        # The step gathers all validation sources, so each rank has the same
        # global objective; reduce once to detect any rank disagreement.
        metric_names = ("total_loss", "semantic_loss", "alignment_loss", "prediction_loss",
                        "retention_loss", "persistence_loss")
        values = torch.stack(tuple(metrics[name].double() for name in metric_names))
        if context.world_size > 1:
            dist.all_reduce(values)
            values /= context.world_size
        return {("validation_loss" if name == "total_loss" else "validation_" + name):
                float(value) for name, value in zip(metric_names, values)}, padded

    records = []
    trial_number = 0

    def run_trial(batch, policy, peak, budget, seed, stage):
        nonlocal trial_number
        if trial_number >= args.max_trials:
            return None
        trial_number += 1
        plan = batch_plan(batch, context.world_size, args.accumulation_steps)
        local_batch = batch // context.world_size
        per_class_rank = local_batch // 2
        total_steps = max(1, round(budget / batch))
        curriculum = max(1, math.ceil(total_steps * .2))
        torch.manual_seed(seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        model = TrajectoryModel(x.shape[-1]).to(context.device)
        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        module = DistributedTrajectoryStep(model)
        if context.world_size > 1:
            module = DistributedDataParallel(module,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                broadcast_buffers=False)
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.}],
            lr=peak, eps=1e-12, betas=(.9, .999))
        initial, val_padding = evaluation(module, curriculum)
        history = [dict(update=0, **initial)]
        epoch, updates, samples_seen = 0, 0, 0
        local_padding_excluded = 0
        status, max_norm = "complete", 0.
        next_eval = (set(range(1, total_steps + 1)) if args.record_every_update else
                     {max(1, round(total_steps * fraction)) for fraction in (.25, .5, .75, 1.)})
        while updates < total_steps:
            samplers = [DistributedSampler(train_classes[c], num_replicas=context.world_size,
                                           rank=context.rank, shuffle=True, seed=seed + c * 1009,
                                           drop_last=False) for c in (0, 1)]
            shards = []
            padding = []
            for label, sampler in enumerate(samplers):
                sampler.set_epoch(epoch)
                positions, count = unpadded_sampler_indices(sampler, len(train_classes[label]))
                shards.append([train_classes[label][i] for i in positions])
                padding.append(count)
                local_padding_excluded += count
            available = min(len(shard) // per_class_rank for shard in shards)
            if available < 1:
                raise ValueError("Batch cannot fit at least once per sampler epoch")
            for offset in range(available):
                if updates >= total_steps:
                    break
                selected = [i for label in (0, 1) for i in
                            shards[label][offset * per_class_rank:(offset + 1) * per_class_rank]]
                if context.world_size > 1:
                    gathered = [None] * context.world_size
                    dist.all_gather_object(gathered, selected)
                    merged = [i for part in gathered for i in part]
                else:
                    merged = selected
                if len(merged) != batch or len(set(merged)) != batch or any(i not in train for i in merged):
                    raise RuntimeError("Distributed ranks duplicated or omitted a source in an update")
                counts = torch.bincount(labels[merged], minlength=2)
                if counts.tolist() != [batch // 2, batch // 2]:
                    raise RuntimeError("Update lost class balance")
                updates += 1
                lr = warmup_cosine(updates, total_steps, peak,
                                   warmup_fraction=args.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                # A fixed pre-encoded bank is sampled deterministically; the
                # two views of a source remain on its assigned rank.
                generator = torch.Generator().manual_seed(seed + updates * 104729)
                all_views = torch.stack([torch.randperm(bank.shape[1], generator=generator)[:2]
                                         for _ in range(len(bank))])
                chosen = torch.as_tensor(selected, dtype=torch.long)
                local_x = bank[chosen[:, None], all_views[chosen]].to(context.device)
                optimizer.zero_grad(set_to_none=True)
                module.train()
                loss, train_metrics, _ = module(local_x, valid[chosen].to(context.device),
                                                times[chosen].to(context.device),
                                                labels[chosen].to(context.device),
                                                epoch=updates, curriculum_epochs=curriculum,
                                                microbatch_sources=plan["per_rank_microbatch"])
                finite = torch.isfinite(loss).to(torch.int)
                if context.world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    status = "pruned_nonfinite_loss"
                    break
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 5.)
                norm_value = float(norm)
                norm_max = torch.tensor(norm_value, device=context.device)
                if context.world_size > 1:
                    dist.all_reduce(norm_max, op=dist.ReduceOp.MAX)
                max_norm = max(max_norm, float(norm_max))
                if not math.isfinite(float(norm_max)) or float(norm_max) > 1000:
                    status = "pruned_unstable_gradient"
                    break
                optimizer.step()
                samples_seen += batch
                if updates in next_eval:
                    current, _ = evaluation(module, curriculum)
                    train_names = ("semantic_loss", "alignment_loss", "prediction_loss",
                                   "retention_loss", "persistence_loss")
                    train_values = torch.stack(tuple(train_metrics[name].detach().double()
                                                     for name in train_names))
                    if context.world_size > 1:
                        dist.all_reduce(train_values)
                        train_values /= context.world_size
                    training_terms = {"training_" + name: float(value)
                                      for name, value in zip(train_names, train_values)}
                    row = dict(update=updates, sampler_epoch=epoch + 1,
                               sources_seen=samples_seen, learning_rate=lr,
                               training_loss=float(loss.detach()),
                               gradient_norm_before_clipping=float(norm_max),
                               **training_terms, **current)
                    history.append(row)
                    if not math.isfinite(current["validation_loss"]) or (
                            updates >= total_steps // 2 and
                            current["validation_loss"] > 1.5 * initial["validation_loss"]):
                        status = "pruned_validation_divergence"
                        break
            if status != "complete":
                break
            epoch += 1
        final = history[-1]
        parameter_change = torch.sqrt(sum(
            (parameter.detach() - initial).square().sum()
            for parameter, initial in zip(model.parameters(), initial_parameters)))
        parameter_change_by_rank = [None] * context.world_size
        if context.world_size > 1:
            dist.all_gather_object(parameter_change_by_rank, float(parameter_change))
        else:
            parameter_change_by_rank[0] = float(parameter_change)
        padding_by_rank = [None] * context.world_size
        if context.world_size > 1:
            dist.all_gather_object(padding_by_rank, local_padding_excluded)
        else:
            padding_by_rank[0] = local_padding_excluded
        result = dict(trial=trial_number, stage=stage, status=status, seed=seed,
                      scaling_formula=policy, peak_learning_rate=peak,
                      weight_decay=args.weight_decay, optimizer="AdamW",
                      weight_decay_excludes_bias_and_one_dimensional=True,
                      optimizer_betas=[.9, .999], optimizer_epsilon=1e-12,
                      gradient_clip_norm=5., warmup_fraction=args.warmup_fraction,
                      warmup_steps=(max(1, min(total_steps - 1,
                          math.ceil(total_steps * args.warmup_fraction))) if total_steps > 1 else 0),
                      planned_steps=total_steps, optimizer_updates=updates,
                      planned_source_budget=budget, actual_sources_seen=samples_seen,
                      sampler_padding_excluded_per_rank=padding_by_rank,
                      sampler_padding_excluded_total=sum(padding_by_rank),
                      validation_sampler_padding_excluded=val_padding,
                      distinct_source_updates_verified=updates,
                      parameter_change_l2_by_rank=parameter_change_by_rank,
                      max_gradient_norm=max_norm, initial_validation_loss=initial["validation_loss"],
                      final_validation_loss=final["validation_loss"],
                      final_validation_semantic_loss=final["validation_semantic_loss"],
                      final_validation_prediction_loss=final["validation_prediction_loss"],
                      final_validation_persistence_loss=final["validation_persistence_loss"],
                      convergence=history, **plan)
        records.append(result)
        if context.is_main:
            print(json.dumps({k: result[k] for k in ("trial", "stage", "status",
                "global_source_batch", "peak_learning_rate", "final_validation_loss")}), flush=True)
            (args.output / "trials.json").write_text(json.dumps(records, indent=2) + "\n")
            if args.record_every_update:
                fields = ("update", "sampler_epoch", "sources_seen", "learning_rate",
                          "gradient_norm_before_clipping", "training_loss",
                          "training_semantic_loss", "training_alignment_loss",
                          "training_prediction_loss", "training_retention_loss",
                          "training_persistence_loss", "validation_loss",
                          "validation_semantic_loss", "validation_alignment_loss",
                          "validation_prediction_loss", "validation_retention_loss",
                          "validation_persistence_loss")
                with (args.output / "loss_history.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows({key: row.get(key) for key in fields} for row in history)
                by_epoch = {}
                for row in history[1:]:
                    by_epoch.setdefault(row["sampler_epoch"], []).append(row)
                epoch_fields = ("epoch", "updates", "sources_seen", "mean_training_loss",
                                "end_validation_loss", "end_validation_semantic_loss",
                                "end_validation_prediction_loss",
                                "end_validation_persistence_loss", "end_learning_rate")
                with (args.output / "epoch_loss_history.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=epoch_fields)
                    writer.writeheader()
                    for epoch_number, rows in by_epoch.items():
                        last = rows[-1]
                        writer.writerow(dict(epoch=epoch_number, updates=len(rows),
                            sources_seen=last["sources_seen"],
                            mean_training_loss=statistics.mean(r["training_loss"] for r in rows),
                            end_validation_loss=last["validation_loss"],
                            end_validation_semantic_loss=last["validation_semantic_loss"],
                            end_validation_prediction_loss=last["validation_prediction_loss"],
                            end_validation_persistence_loss=last["validation_persistence_loss"],
                            end_learning_rate=last["learning_rate"]))
            if status == "complete" and stage != "coarse":
                destination = args.output / f"trial_{trial_number:02d}_checkpoint.pt"
                temporary = destination.with_name(destination.name + ".tmp")
                torch.save(dict(format="trajectory_lr_search_v1", model=model.state_dict(),
                    optimizer=optimizer.state_dict(), trial=result, train_source_ids=[cache["ids"][i] for i in train],
                    validation_source_ids=[cache["ids"][i] for i in validation],
                    width=x.shape[-1], hidden=128, embedding_dim=128,
                    cache_metadata={key: value for key, value in cache.items()
                                    if not isinstance(value, torch.Tensor)},
                    cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest()), temporary)
                os.replace(temporary, destination)
        return result

    coarse, full, repeats = [], [], []
    if args.single_run:
        if 32 not in batches:
            raise ValueError("A global source batch of 32 is infeasible for this cache and world size")
        result = run_trial(32, "fixed_selected", args.base_learning_rate,
                           args.full_source_budget, args.seed, "single_run")
        if result is not None:
            full.append(result)
    else:
        for batch in batches:
            for policy in (("constant",) if batch == 32 else ("constant", "sqrt", "linear")):
                result = run_trial(batch, policy, scaled_lr(args.base_learning_rate, batch,
                                   policy=policy), args.coarse_source_budget, args.seed, "coarse")
                if result is not None:
                    coarse.append(result)
        promising = sorted((r for r in coarse if r["status"] == "complete"),
                           key=lambda r: r["final_validation_loss"])[:2]
        seen = set()
        for candidate in promising:
            for factor in (.5, 1., 2.):
                peak = candidate["peak_learning_rate"] * factor
                key = (candidate["global_source_batch"], round(math.log(peak), 10))
                if key in seen:
                    continue
                seen.add(key)
                result = run_trial(candidate["global_source_batch"], candidate["scaling_formula"],
                                   peak, args.full_source_budget, args.seed, "refine")
                if result is not None:
                    full.append(result)
        repeat_candidates = sorted((r for r in full if r["status"] == "complete"),
                                   key=lambda r: r["final_validation_loss"])[:2]
        for offset in (10000, 20000, 30000):
            for candidate in repeat_candidates:
                result = run_trial(candidate["global_source_batch"], candidate["scaling_formula"],
                                   candidate["peak_learning_rate"], args.full_source_budget,
                                   args.seed + offset, "repeat")
                if result is not None:
                    repeats.append(result)
    # Rank repeated configurations by mean across their two seeds; if budget
    # prevents repeats, report the single-run evidence explicitly.
    eligible = [r for r in full if r["status"] == "complete"]
    if repeats:
        eligible = [r for r in eligible if any(
            t["status"] == "complete" and t["global_source_batch"] == r["global_source_batch"] and
            t["peak_learning_rate"] == r["peak_learning_rate"] for t in repeats)]
    ranked = []
    for candidate in eligible:
        matching = [r for r in repeats if r["status"] == "complete" and
                    r["global_source_batch"] == candidate["global_source_batch"] and
                    r["peak_learning_rate"] == candidate["peak_learning_rate"]]
        losses = [candidate["final_validation_loss"]] + [r["final_validation_loss"] for r in matching]
        ranked.append((sum(losses) / len(losses), -len(losses), candidate, losses))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best = ranked[0] if ranked else None
    if context.is_main:
        ranking = [dict(global_source_batch=item[2]["global_source_batch"],
                        peak_learning_rate=item[2]["peak_learning_rate"],
                        scaling_formula=item[2]["scaling_formula"],
                        mean_validation_loss=item[0],
                        validation_loss_std=(statistics.stdev(item[3]) if len(item[3]) > 1 else None),
                        validation_losses=item[3], repetitions=len(item[3])) for item in ranked]
        selected_runs = ([r for r in [best[2], *repeats] if
                          r["global_source_batch"] == best[2]["global_source_batch"] and
                          r["peak_learning_rate"] == best[2]["peak_learning_rate"] and
                          r["status"] == "complete"] if best else [])
        selected_run = (min((r for r in [best[2], *repeats] if
                             r["global_source_batch"] == best[2]["global_source_batch"] and
                             r["peak_learning_rate"] == best[2]["peak_learning_rate"] and
                             r["status"] == "complete"),
                            key=lambda r: r["final_validation_loss"]) if best else None)
        report = dict(protocol=("fixed-configuration source-held-out training; no independent test"
                                if args.single_run else
                                "source-held-out hyperparameter selection; no independent test"),
            selected=(dict(trial=selected_run["trial"], global_source_batch=best[2]["global_source_batch"],
                           peak_learning_rate=best[2]["peak_learning_rate"],
                           scaling_formula=("indistinguishable_at_reference_batch"
                                            if best[2]["global_source_batch"] == 32 else
                                            best[2]["scaling_formula"]),
                           mean_validation_loss=best[0], validation_losses=best[3],
                           validation_loss_std=(statistics.stdev(best[3]) if len(best[3]) > 1 else None),
                           mean_validation_prediction_loss=statistics.mean(
                               r["final_validation_prediction_loss"] for r in selected_runs),
                           validation_persistence_loss=selected_runs[0]["final_validation_persistence_loss"],
                           mean_prediction_below_persistence=(statistics.mean(
                               r["final_validation_prediction_loss"] for r in selected_runs) <
                               selected_runs[0]["final_validation_persistence_loss"]),
                           repetitions=len(best[3])) if best else None),
            configuration_ranking=ranking,
            trial_count=len(records), max_trials=args.max_trials,
            source_exposures_used=sum(r["actual_sources_seen"] for r in records),
            train_source_count=len(train), validation_source_count=len(validation),
            train_source_ids=[cache["ids"][i] for i in train],
            validation_source_ids=[cache["ids"][i] for i in validation],
            source_group_split=True, cache_identity=cache_identity(cache),
            cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
            cache_temporal_context_policy=cache["temporal_context_policy"],
            cache_sampling=cache.get("sampling"), cache_augmentation=cache.get("augmentation"),
            cache_augmented_views=bank.shape[1] - 2, model_id=cache.get("model_id"),
            encoder_checkpoint_sha256=cache.get("checkpoint_sha256"),
            gpu_configuration=devices, distributed_world_size=context.world_size,
            settings={key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
            selected_checkpoint=(f"trial_{selected_run['trial']:02d}_checkpoint.pt" if best else None),
            caveats=["Validation is used for selection and is not an independent final test.",
                     "A finite budget cannot establish a global optimum.",
                     "At the reference batch of 32, constant, square-root, and linear scaling give the same rate.",
                     "Source groups may share a physical scene; scene-held-out generalization is unverified.",
                     "Learning-rate and batch results apply to this cache's clip length and view bank.",
                     "The V-JEPA encoder remains frozen; only the trajectory adapter is tuned."])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if selected_run is not None:
            checkpoint = torch.load(args.output / report["selected_checkpoint"],
                                    map_location="cpu", weights_only=False)
            selected_model = TrajectoryModel(x.shape[-1]).eval()
            selected_model.load_state_dict(checkpoint["model"], strict=True)
            selected_indices = torch.as_tensor(train, dtype=torch.long)
            fixed = x[selected_indices]
            source_valid = valid[selected_indices]
            source_times = times[selected_indices]
            with torch.no_grad():
                output = selected_model(
                    fixed.reshape(-1, fixed.shape[2], fixed.shape[3]),
                    source_valid[:, None].expand(-1, 2, -1).reshape(-1, fixed.shape[2]),
                    source_times[:, None].expand(-1, 2, -1).reshape(-1, fixed.shape[2]))
            torch.save(dict(embedding=output["embedding"].cpu(),
                            labels=labels[selected_indices],
                            ids=[cache["ids"][i] for i in train],
                            source_split="training_only",
                            checkpoint_trial=selected_run["trial"]),
                       args.output / "representations.pt")
        with (args.output / "comparison.csv").open("w", newline="") as stream:
            fields = ("trial", "stage", "status", "seed", "global_source_batch",
                      "per_rank_microbatch", "accumulation_steps", "scaling_formula",
                      "peak_learning_rate", "optimizer_updates", "actual_sources_seen",
                      "initial_validation_loss", "final_validation_loss", "max_gradient_norm")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in records)
        summary = [("# Trajectory fixed-configuration GPU training" if args.single_run else
                    "# Trajectory learning-rate and batch search"), "",
            f"Source-disjoint selection set: {len(validation)} videos; training: {len(train)} videos.",
            f"Trials: {len(records)}; source exposures: {report['source_exposures_used']}.", ""]
        if best:
            summary += [f"Selected peak LR: {best[2]['peak_learning_rate']:.6g}",
                        f"Selected global source batch: {best[2]['global_source_batch']}",
                        f"Scaling formula: {report['selected']['scaling_formula']}",
                        f"Mean source-held-out validation loss: {best[0]:.6g} from {len(best[3])} run(s).",
                        f"Mean prediction loss: {report['selected']['mean_validation_prediction_loss']:.6g}; "
                        f"frozen persistence baseline: {report['selected']['validation_persistence_loss']:.6g}.", ""]
        else:
            summary += ["No stable full-budget configuration was found.", ""]
        summary += ["| Trial | Stage | Sources/update | Peak LR | Rule | Updates | Validation loss | Status |",
                    "|---:|---|---:|---:|---|---:|---:|---|"]
        for row in records:
            summary.append(f"| {row['trial']} | {row['stage']} | {row['global_source_batch']} | "
                           f"{row['peak_learning_rate']:.6g} | {row['scaling_formula']} | "
                           f"{row['optimizer_updates']} | {row['final_validation_loss']:.6g} | "
                           f"{row['status']} |")
        summary += ["", "Each value uses the same fixed source-held-out selection set. "
                    "Full-budget trials are compared with full-budget trials; pruned coarse trials "
                    "are excluded from the winner selection.", "",
                    "The selection set is not an independent test. No spatial localization or "
                    "unseen-scene claim follows from these losses."]
        (args.output / "comparison.md").write_text("\n".join(summary) + "\n")
        print(json.dumps({"selected": report["selected"], "report": str(args.output / "report.json")}), flush=True)
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
