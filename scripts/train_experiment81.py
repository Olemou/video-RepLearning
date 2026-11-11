#!/usr/bin/env python3
"""Isolated Experiment81: matched frozen/full online JEPA + four trajectory losses."""
from __future__ import annotations
import argparse
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel, DistributedTrajectoryStep, trajectory_loss
from vision_model.training.relative_clock import trajectory_model_for_clock
from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from vision_model.training.online_trajectory import (OnlinePrefixEncoder, collect_prefix_features,
    replay_prefix_gradients, parameter_groups, gradient_norm)
from prepare_experiment81 import normalize_bank, digest
from train_trajectory import select_training_views
from diagnose_spill_representation import recognition


def cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("frozen", "full"), required=True)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seed", type=int, default=239)
    p.add_argument("--trajectory-lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--chunk-sources", type=int, default=2)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--clock-policy", choices=("absolute", "relative_prefix"), default="absolute")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.updates, a.chunk_sources, a.save_every, a.eval_every) < 1:
        p.error("Budgets and chunk sizes must be positive")
    torch.set_num_threads(4)
    ctx = initialize_distributed()
    if ctx.device.type != "cuda":
        raise RuntimeError("Real GPU training is required; CPU is for tests only")
    c = torch.load(a.prepared, map_location="cpu", weights_only=False, mmap=True)
    if c["temporal_context_policy"] != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Causal policy mismatch")
    train, selection = c["train_indices"], c["selection_indices"]
    if sorted(train.tolist() + selection.tolist()) != list(range(len(c["ids"]))):
        raise ValueError("Split contains missing/duplicated sources")
    local_train = train[ctx.rank::ctx.world_size]
    local_selection = selection[ctx.rank::ctx.world_size]
    owned = torch.cat((local_train, local_selection)).sort().values
    order, counts = audit_distributed_indices(owned.tolist(), len(c["ids"]))
    if not len(local_train):
        raise ValueError("Each rank needs distinct training sources")
    train_counts = [None] * ctx.world_size
    if ctx.world_size > 1:
        dist.all_gather_object(train_counts, len(local_train))
    else:
        train_counts = [len(local_train)]
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    if ctx.is_main:
        a.output.mkdir(parents=True, exist_ok=a.resume)
    if ctx.world_size > 1:
        dist.barrier()
    settings = dict(mode=a.mode, seed=a.seed, trajectory_lr=a.trajectory_lr, encoder_lr=a.encoder_lr,
        chunk_sources=a.chunk_sources, world_size=ctx.world_size,
        activation_checkpointing=a.activation_checkpointing, prepared_sha256=digest(a.prepared),
        objective="ViCo_episode + 0.1 align + predict + 0.1 keep", curriculum_updates=20,
        optimizer_reset_before_update=21, normalization=c["normalization"], train_rank_counts=train_counts,
        eval_rank_counts=counts, host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        precision="FP32 master parameters/optimizer; BF16 encoder forward; float64 ViCo",
        teacher_metadata=c["metadata"], view_bank_count=c["pixels"].shape[1],
        source_count=len(c["ids"]), train_source_count=len(train), selection_source_count=len(selection),
        manual_annotations=0, mil=False, decoder=False, independent_scene_test=False)
    # Fixed CPU view bank; only the assigned sources are moved/encoded by this rank.
    pixels = {int(i): normalize_bank(c["pixels"][i], c["normalization"]) for i in owned}
    encoder = None
    initial_encoder = None
    if a.mode == "full":
        base = load_pretrained_encoder(c["metadata"]["encoder_dir"], local_files_only=True).encoder
        initial_encoder = {k: v.detach().clone() for k, v in base.state_dict().items()}
        encoder = OnlinePrefixEncoder(base, activation_checkpointing=a.activation_checkpointing).to(ctx.device).eval()
        encoder.requires_grad_(True)
    # Loading the encoder consumes RNG; reset immediately before BOTH trajectory initializations.
    torch.manual_seed(a.seed)
    trajectory = trajectory_model_for_clock(c["teacher"].shape[-1], a.clock_policy).to(ctx.device)
    if a.clock_policy != "absolute":
        settings["clock_policy"] = a.clock_policy
    initial_trajectory_sha = digest_tensor_state(trajectory.state_dict())
    settings["initial_trajectory_sha256"] = initial_trajectory_sha
    step = DistributedTrajectoryStep(trajectory)
    wrapped_step = (DDP(step, device_ids=[ctx.local_rank], broadcast_buffers=False)
                    if ctx.world_size > 1 else step)
    wrapped_encoder = (DDP(encoder, device_ids=[ctx.local_rank], broadcast_buffers=False)
                       if ctx.world_size > 1 and encoder is not None else encoder)

    def optimizer_new():
        groups = [dict(params=trajectory.parameters(), lr=a.trajectory_lr)]
        if encoder is not None:
            groups.append(dict(params=encoder.parameters(), lr=a.encoder_lr))
        return torch.optim.AdamW(groups, weight_decay=0., eps=1e-12)

    optimizer = optimizer_new()
    history, updates, exposures, rng_state = [], 0, torch.zeros(len(c["ids"]), c["pixels"].shape[1], dtype=torch.long), None
    checkpoint_path = a.output / "checkpoint.pt"
    if a.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["settings"].get("clock_policy", "absolute") != a.clock_policy:
            raise ValueError("Resume setting differs: clock_policy")
        for key in ("mode", "seed", "trajectory_lr", "encoder_lr", "chunk_sources", "world_size",
                    "activation_checkpointing", "prepared_sha256", "initial_trajectory_sha256"):
            if saved["settings"][key] != settings[key]:
                raise ValueError(f"Resume setting differs: {key}")
        trajectory.load_state_dict(saved["trajectory"], strict=True)
        if encoder is not None:
            encoder.encoder.load_state_dict(saved["encoder"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        updates, history = saved["optimizer_updates"], saved["history"]
        exposures = saved["view_exposures"]
        rng_state = saved["rng_states"][ctx.rank]
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        if updates > a.updates:
            raise ValueError("--updates is total target, not additional updates")
    if ctx.is_main:
        (a.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    baseline_exposures = exposures.clone()
    local_exposures = torch.zeros_like(exposures, device=ctx.device)
    started = time.monotonic()

    @torch.no_grad()
    def evaluate(update):
        trajectory.eval()
        rows = []
        for start in range(0, len(owned), a.chunk_sources):
            ids = owned[start:start + a.chunk_sources]
            if encoder is None:
                feature = c["teacher"][ids, :2]
            else:
                px = torch.stack([pixels[int(i)][:2] for i in ids])
                feature = collect_prefix_features(encoder, px, c["valid"][ids], ctx.device,
                                                  chunk_sources=a.chunk_sources).cpu()
            rows.extend((int(i), f) for i, f in zip(ids, feature))
        gathered = [None] * ctx.world_size if ctx.is_main else None
        if ctx.world_size > 1:
            dist.gather_object(rows, gathered, dst=0)
        else:
            gathered = [rows]
        result = representation = None
        if ctx.is_main:
            rows = sorted([r for shard in gathered for r in shard], key=lambda r: r[0])
            if [r[0] for r in rows] != list(range(len(c["ids"]))):
                raise RuntimeError("Repeated or missing evaluation sources")
            feature = torch.stack([r[1] for r in rows])
            result = dict(update=update)
            embeddings = torch.zeros(len(c["ids"]), 2, 128)
            for name, ids in (("train", train), ("selection", selection)):
                _, metrics, output = trajectory_loss(trajectory, c["teacher"][ids, :2].to(ctx.device),
                    c["valid"][ids].to(ctx.device), c["timestamps"][ids].to(ctx.device),
                    c["labels"][ids].to(ctx.device), epoch=20, curriculum_epochs=20,
                    student_features=feature[ids].to(ctx.device))
                result[name] = {k: float(v) for k, v in metrics.items()}
                embeddings[ids] = output["embedding"].reshape(len(ids), 2, -1).cpu()
            for name, ids in (("train", train), ("selection", selection)):
                result[name].update(recognition(embeddings[:, 0], c["labels"], train, ids,
                                                exclude_self=name == "train"))
            representation = dict(ids=c["ids"], labels=c["labels"], train_indices=train,
                selection_indices=selection, embeddings=embeddings, features=feature,
                valid=c["valid"], timestamps=c["timestamps"], frame_indices=c["frame_indices"])
        if ctx.world_size > 1:
            dist.barrier()
        return result, representation

    def save(update, representation):
        states = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
        if ctx.world_size > 1:
            states_all = [None] * ctx.world_size if ctx.is_main else None
            dist.gather_object(states, states_all, dst=0)
            seen = local_exposures.clone(); dist.all_reduce(seen)
        else:
            states_all, seen = [states], local_exposures
        if ctx.is_main:
            state = dict(format="experiment81_online_v1", trajectory=cpu_state(trajectory),
                encoder=cpu_state(encoder.encoder) if encoder is not None else None,
                optimizer=optimizer.state_dict(), optimizer_updates=update, settings=settings,
                history=history, view_exposures=(baseline_exposures + seen.cpu()), rng_states=states_all)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(state, temporary); os.replace(temporary, checkpoint_path)
            if representation is not None:
                torch.save(representation, a.output / "representations.pt")
        if ctx.world_size > 1:
            dist.barrier()

    if not a.resume:
        initial, representation = evaluate(0)
        if ctx.is_main:
            history.append(initial)
            torch.save(representation, a.output / "initial_representations.pt")
    for update in range(updates + 1, a.updates + 1):
        trajectory.train()
        if update == 21:
            optimizer = optimizer_new()
        _, views = select_training_views(c["teacher"], seed=a.seed, epoch=update)
        selected = views[local_train]
        teacher = c["teacher"][local_train[:, None], selected].to(ctx.device)
        px = torch.stack([pixels[int(i)][v] for i, v in zip(local_train, selected)])
        valid, times, labels = (c[key][local_train].to(ctx.device)
                               for key in ("valid", "timestamps", "labels"))
        if encoder is None:
            features = teacher
        else:
            features = collect_prefix_features(encoder, px, c["valid"][local_train], ctx.device,
                                               chunk_sources=a.chunk_sources)
        leaf = features.detach().requires_grad_(encoder is not None)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = wrapped_step(teacher, valid, times, labels, epoch=update,
                                        curriculum_epochs=20, student_features=leaf)
        term_gradient_norms = {}
        if encoder is not None and update == 1:
            for name in ("semantic_loss", "alignment_loss", "prediction_loss", "retention_loss"):
                grad = torch.autograd.grad(metrics[name], leaf, retain_graph=True)[0]
                term_gradient_norms[name] = float(grad.norm())
            if ctx.is_main:
                (a.output / "initial_feature_gradient_audit.json").write_text(json.dumps(dict(
                    scope="Full global loss derivatives at pooled student features; rank-local shard",
                    raw_term_gradient_norms=term_gradient_norms), indent=2) + "\n")
        loss.backward()
        replay_error = 0.
        block_norms = {}
        if encoder is not None:
            replay_error = replay_prefix_gradients(wrapped_encoder, px, c["valid"][local_train],
                leaf.grad.detach(), ctx.device, chunk_sources=a.chunk_sources, replay_reference=features)
            block_norms = {k: gradient_norm(v) for k, v in parameter_groups(encoder.encoder).items()}
            if not all(v > 0 and torch.isfinite(torch.tensor(v)) for v in block_norms.values()):
                raise RuntimeError(f"Missing/nonfinite backbone gradient: {block_norms}")
        params = list(trajectory.parameters()) + (list(encoder.parameters()) if encoder is not None else [])
        norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
        optimizer.step()
        for i, v in zip(local_train, selected):
            local_exposures[int(i), v.to(ctx.device)] += 1
        row = dict(update=update, batch_training={k: float(v.detach()) for k, v in metrics.items()},
            gradient_norm_before_clipping=float(norm), block_gradient_norms=block_norms,
            replay_max_feature_error=replay_error, elapsed_seconds=time.monotonic() - started,
            source_exposures_this_update=len(train))
        # Replicated global metrics are identical; max-reduce the replay diagnostic.
        if ctx.world_size > 1:
            error = torch.tensor(replay_error, device=ctx.device); dist.all_reduce(error, op=dist.ReduceOp.MAX)
            row["replay_max_feature_error"] = float(error)
        if encoder is not None and (update == 1 or update == a.updates):
            drift = {}
            for name in parameter_groups(encoder.encoder):
                prefix = (f"blocks.{int(name[-2:])}." if name.startswith("block_") else
                          "norm." if name == "final_norm" else "patch_embed.")
                drift[name] = max(float((value.detach().cpu() - initial_encoder[k]).abs().max())
                                  for k, value in encoder.encoder.state_dict().items() if k.startswith(prefix))
            row["block_parameter_max_change"] = drift
            if not all(v > 0 for v in drift.values()):
                raise RuntimeError(f"An intended backbone group did not change: {drift}")
        representation = None
        if update % a.eval_every == 0 or update == a.updates or update == 1:
            result, representation = evaluate(update)
            if ctx.is_main:
                row.update(train=result["train"], selection=result["selection"])
        if ctx.is_main:
            history.append(row)
            (a.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(dict(mode=a.mode, update=update, seconds=row["elapsed_seconds"],
                training_total=row.get("train", {}).get("total_loss"),
                selection_total=row.get("selection", {}).get("total_loss"),
                selection_accuracy=row.get("selection", {}).get("accuracy"),
                replay_error=row["replay_max_feature_error"])), flush=True)
        if update == 1 or update % a.save_every == 0 or update == a.updates:
            save(update, representation)
    if ctx.is_main:
        (a.output / "completed.json").write_text(json.dumps(dict(optimizer_updates=a.updates,
            seconds=time.monotonic() - started, localization_validated=False,
            checkpoint_sha256=digest(checkpoint_path)), indent=2) + "\n")
    if ctx.world_size > 1:
        dist.destroy_process_group()


def digest_tensor_state(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
