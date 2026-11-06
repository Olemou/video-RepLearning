#!/usr/bin/env python3
"""All-head/all-layer final-decision diagnostics for the two Experiment81 arms."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel
from vision_model.training.relative_clock import trajectory_model_for_clock
from vision_model.training.online_trajectory import OnlinePrefixEncoder, collect_prefix_features
from vision_model.training.distributed import initialize_distributed
from vision_model.interpretability.online_trajectory import explain_prefix_group
from prepare_experiment81 import normalize_bank, digest


def load_case(root, case, device):
    saved = torch.load(root / case / "checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
    reps = torch.load(root / case / "representations.pt", map_location="cpu", weights_only=False)
    encoder = load_pretrained_encoder(saved["settings"]["teacher_metadata"]["encoder_dir"], local_files_only=True).encoder
    if saved["encoder"] is not None:
        encoder.load_state_dict(saved["encoder"], strict=True)
    model = trajectory_model_for_clock(1024, saved["settings"].get("clock_policy", "absolute"))
    model.load_state_dict(saved["trajectory"], strict=True)
    encoder, model = encoder.to(device).eval().requires_grad_(False), model.to(device).eval().requires_grad_(False)
    train = reps["train_indices"]
    prototype = torch.stack([F.normalize(reps["embeddings"][train, 0][reps["labels"][train] == label].mean(0), dim=0)
                             for label in (0, 1)]).to(device)
    settings = saved["settings"]
    del saved
    return encoder, model, reps, prototype, settings


def margins_for_group(model, reps, group_ids, group_features, prototypes, device):
    """Preserve the original split-wide trajectory batch shape during replay."""
    values = []
    for source, _ in enumerate(group_ids):
        source_id = int(group_ids[source])
        split = reps["train_indices"] if source_id in reps["train_indices"] else reps["selection_indices"]
        base = reps["features"][split].to(device).clone()
        for j, index in enumerate(group_ids):
            found = (split == index).nonzero().flatten()
            if len(found):
                # CopySlices remains differentiable in the group student features.
                base[int(found[0])] = group_features[j]
        valid = reps["valid"][split].to(device)
        stamps = reps["timestamps"][split].to(device)
        n, views, time, width = base.shape
        out = model(base.reshape(n * views, time, width), valid[:, None].expand(n, views, time).reshape(-1, time),
                    stamps[:, None].expand(n, views, time).reshape(-1, time), detach_teacher=False)
        at = int((split == source_id).nonzero()[0]) * views
        scores = out["embedding"][at] @ prototypes.T
        values.append(scores[1] - scores[0])
    return torch.stack(values)


def evaluation_groups(reps, settings):
    groups = []
    for rank in range(settings["world_size"]):
        owned = torch.cat((reps["train_indices"][rank::settings["world_size"]],
                           reps["selection_indices"][rank::settings["world_size"]])).sort().values
        for start in range(0, len(owned), settings["chunk_sources"]):
            groups.append(owned[start:start + settings["chunk_sources"]])
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case", choices=("frozen", "full"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(); torch.set_num_threads(4)
    ctx = initialize_distributed()
    if ctx.device.type != "cuda":
        raise RuntimeError("Attribution must run on the allocated GPU")
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    if not c["valid"].all():
        raise ValueError("This all-prefix diagnostic requires a separate variable-tail packing protocol")
    encoder, model, reps, proto, settings = load_case(a.root, a.case, ctx.device)
    checkpoint_sha = digest(a.root / a.case / "checkpoint.pt")
    method_sha = digest(ROOT / "src/vision_model/interpretability/online_trajectory.py")
    if ctx.is_main:
        a.output.mkdir(parents=True, exist_ok=a.resume)
        (a.output / "metadata.json").write_text(json.dumps(dict(case=a.case, checkpoint_sha256=checkpoint_sha,
            source_count=len(c["ids"]), method="final decision margin, all heads, all layers; LeGrad-style adaptation",
            mathematical_target="cosine(z, training_spill_prototype) - cosine(z, training_clean_prototype)",
            explicit_attention_forward=True, original_legrad=False, raw_rollout_class_specific=False,
            attention_query="uniform final tubelet; no CLS token", manual_annotations=0,
            production_precision=settings["precision"], display_scale="determined later from training sources only",
            attribution_replay_tolerance=1e-6, method_source_sha256=method_sha,
            localization_validated=False), indent=2) + "\n")
    if ctx.world_size > 1:
        dist.barrier()
    groups = evaluation_groups(reps, settings)
    if a.only:
        groups = [ids for ids in groups if any(c["ids"][int(i)] in a.only for i in ids)]
    summaries = []
    for round_index in range((len(groups) + ctx.world_size - 1) // ctx.world_size):
        slot = round_index * ctx.world_size + ctx.rank
        row = None
        if slot < len(groups):
            ids = groups[slot]
            group_path = a.output / ("group_" + "_".join(map(str, ids.tolist())) + ".pt")
            if a.resume and group_path.exists():
                old = torch.load(group_path, map_location="cpu", weights_only=False)
                if old["checkpoint_sha256"] != checkpoint_sha:
                    raise ValueError("Refusing to reuse maps from another checkpoint")
                if old.get("method_source_sha256") != method_sha:
                    raise ValueError("Refusing to reuse maps from another method version")
                row = dict(old["summary"], file=group_path.name, ids=ids.tolist(), reused=True)
            else:
                pixels = normalize_bank(c["pixels"][ids, :2].flatten(0, 1), c["normalization"])
                pixels = pixels.reshape(len(ids), 2, *pixels.shape[1:])
                # Normalize per-source explicitly: the helper handles [view,frame,channel,H,W].
                # Flatten above retains the frame axis and broadcasts mean/std correctly.
                with torch.no_grad():
                    production = collect_prefix_features(OnlinePrefixEncoder(encoder), pixels, c["valid"][ids],
                                                          ctx.device, chunk_sources=len(ids))
                    baseline_scores = margins_for_group(model, reps, ids, reps["features"][ids].to(ctx.device), proto, ctx.device)
                    production_scores = margins_for_group(model, reps, ids, production, proto, ctx.device)
                result = explain_prefix_group(encoder, pixels,
                    lambda feature: margins_for_group(model, reps, ids, feature, proto, ctx.device), ctx.device)
                feature_error = float((production.cpu() - reps["features"][ids]).abs().max())
                explicit_feature_error = float((result["explicit_features"] - production.cpu()).abs().max())
                score_error = (result["explicit_scores"] - production_scores.cpu()).abs()
                row = dict(file=group_path.name, ids=ids.tolist(), reused=False,
                    saved_to_production_max_feature_error=feature_error,
                    saved_to_production_margin_error=(production_scores.cpu() - baseline_scores.cpu()).abs().tolist(),
                    production_to_explicit_max_feature_error=explicit_feature_error,
                    production_to_explicit_margin_error=score_error.tolist(),
                    production_margin=production_scores.cpu().tolist(), explicit_margin=result["explicit_scores"].tolist(),
                    prediction_agreement=bool(((production_scores.cpu() >= 0) == (result["explicit_scores"] >= 0)).all()),
                    explicit_self_replay_error=result["replay_max_feature_error"])
                result.update(checkpoint_sha256=checkpoint_sha, method_source_sha256=method_sha,
                              ids=ids, summary=row)
                row["payload"] = result
                print(json.dumps({k: v for k, v in row.items() if k != "payload"}), flush=True)
        gathered = [None] * ctx.world_size if ctx.is_main else None
        if ctx.world_size > 1:
            dist.gather_object(row, gathered, dst=0)
        else:
            gathered = [row]
        if ctx.is_main:
            for entry in gathered:
                if entry is None:
                    continue
                payload = entry.pop("payload", None)
                if payload is not None:
                    # Avoid a cyclic payload->summary->payload reference.
                    payload["summary"] = dict(entry)
                    torch.save(payload, a.output / entry["file"])
                summaries.append(entry)
            (a.output / "progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
        if ctx.world_size > 1:
            dist.barrier()
    if ctx.is_main:
        (a.output / "completed.json").write_text(json.dumps(dict(groups=len(summaries),
            sources=sum(len(x["ids"]) for x in summaries), checkpoint_sha256=checkpoint_sha,
            localization_validated=False), indent=2) + "\n")
    if ctx.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
