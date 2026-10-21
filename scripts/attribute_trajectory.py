#!/usr/bin/env python3
"""Attribute a trained causal trajectory checkpoint to source video timestamps."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig
from vision_model.data.video_dataset import read_manifest
from vision_model.interpretability.temporal_attribution import (
    AttributionTarget, attribute_trajectory, perturbation_check)
from vision_model.interpretability.video_visualization import save_temporal_figures
from vision_model.training.distributed import initialize_distributed
from vision_model.training.trajectory import TrajectoryModel


def _frame_indices(cache: dict, records: dict, index: int) -> torch.Tensor:
    """Use cached indices; verify/reconstruct legacy deterministic caches."""
    if "frame_indices" in cache:
        indices = cache["frame_indices"][index]
    else:
        config = cache.get("sampling")
        if not isinstance(config, dict) or config.get("training") is not False:
            raise ValueError("Cache lacks frame_indices and deterministic sampling metadata; rebuild it")
        record = records[cache["ids"][index]]
        sampled = TemporalSampler(TemporalSamplingConfig(**config)).sample(
            num_frames=record.num_frames, source_fps=record.fps, sample_index=index)
        indices = torch.as_tensor(sampled[1::2], dtype=torch.long)
        if len(indices) < cache["timestamps"].shape[1]:
            raise ValueError("Legacy sampling/frame count mismatch; rebuild the cache")
        indices = indices[:cache["timestamps"].shape[1]]
    length = int(cache["valid"][index].sum())
    record = records[cache["ids"][index]]
    if (indices.shape != cache["valid"][index].shape or
            not torch.allclose(indices[:length].float() / record.fps,
                               cache["timestamps"][index, :length], atol=1e-4, rtol=0)):
        raise ValueError("Frame indices and timestamps disagree; rebuild the cache")
    return indices


def _reference(args, model, cache, device):
    if args.target != "semantic_similarity":
        return None
    if args.reference is not None:
        value = torch.load(args.reference, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value["embedding"]
        return torch.as_tensor(value, device=device).flatten()
    if args.prototype_class is None:
        raise ValueError("Semantic similarity requires --reference or --prototype-class")
    # A prototype built from fitted sources is explicitly a training-fit reference.
    representations = args.checkpoint.parent / "representations.pt"
    saved = torch.load(representations, map_location="cpu", weights_only=False)
    embeddings = saved["embedding"].reshape(len(cache["ids"]), 2, -1)[:, 0]
    chosen = embeddings[cache["labels"] == args.prototype_class]
    if chosen.numel() == 0:
        raise ValueError("No prototype examples in chosen class")
    return torch.nn.functional.normalize(chosen.mean(0), dim=0).to(device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--video-id")
    choice.add_argument("--all-videos", action="store_true")
    p.add_argument("--target", required=True, choices=["state", "dynamics", "semantic_similarity",
                                                      "future_prediction", "future_error"])
    p.add_argument("--timestep", type=int)
    p.add_argument("--component", type=int)
    p.add_argument("--future-timestep", type=int)
    p.add_argument("--reference", type=Path)
    p.add_argument("--prototype-class", type=int, choices=[0, 1])
    p.add_argument("--view", type=int, default=0, choices=[0, 1])
    p.add_argument("--method", choices=["positive_gradient", "attention_gradient"],
                   default="positive_gradient")
    p.add_argument("--normalization", choices=["none", "sum", "max"], default="sum")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--heads", type=int, nargs="+")
    p.add_argument("--full-maps", action="store_true", help="Save quadratic maps for one selected video")
    p.add_argument("--max-attention-elements", type=int, default=8_000_000)
    p.add_argument("--storyboard-count", type=int, default=8)
    p.add_argument("--faithfulness", action="store_true")
    args = p.parse_args(argv)
    if args.full_maps and args.all_videos:
        p.error("--full-maps requires a single --video-id to bound distributed transfer")
    context = initialize_distributed()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "trajectory_resumable_v1":
        raise ValueError("Expected a resumable trajectory checkpoint")
    if cache.get("temporal_context_policy") != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Cache does not guarantee past-only prefix features")
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if checkpoint["resume_settings"]["cache_sha256"] != digest:
        raise ValueError("Checkpoint was trained with a different feature cache")
    manifest = Path(cache["manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != cache["manifest_sha256"]:
        raise ValueError("Manifest changed since feature extraction")
    records = {record.video_id: record for record in read_manifest(manifest)}
    if set(records) != set(cache["ids"]):
        raise ValueError("Manifest and cache video IDs disagree")
    model = TrajectoryModel(checkpoint["width"], checkpoint["hidden"],
                            checkpoint["embedding_dim"]).to(context.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    reference = _reference(args, model, cache, context.device)
    selected = list(range(len(cache["ids"]))) if args.all_videos else [cache["ids"].index(args.video_id)]
    sampler = DistributedSampler(selected, num_replicas=context.world_size, rank=context.rank,
                                 shuffle=False, drop_last=False)
    sampler.set_epoch(0)
    # DistributedSampler pads by repeating positions at the end. Remove those
    # positions using their global slot, not the repeated sample's identity.
    local_positions = [position for slot, position in enumerate(sampler)
                       if context.rank + slot * context.world_size < len(selected)]
    local = []
    for position in local_positions:
        index = selected[position]
        video_id = cache["ids"][index]
        indices = _frame_indices(cache, records, index)
        features = cache["features"][index, args.view].to(context.device)
        valid = cache["valid"][index].to(context.device)
        times = cache["timestamps"][index].to(context.device)
        target = AttributionTarget(args.target, args.timestep, args.component,
                                   reference, args.future_timestep)
        result = attribute_trajectory(model, features, valid, times, target,
            video_id=video_id, sample_id=("identity", "mirror")[args.view],
            frame_indices=indices, checkpoint_id=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            method=args.method, normalization=args.normalization,
            layers=None if args.layers is None else tuple(args.layers),
            heads=None if args.heads is None else tuple(args.heads),
            include_maps=args.full_maps, max_attention_elements=args.max_attention_elements)
        faithfulness = perturbation_check(model, features, valid, times, target,
                                         result.relevance, count=1) if args.faithfulness else None
        local.append((index, result, faithfulness))
    if context.world_size > 1:
        shards = [None] * context.world_size if context.is_main else None
        dist.gather_object(local, shards, dst=0)
        gathered = [item for shard in shards for item in shard] if context.is_main else []
    else:
        gathered = local
    if context.is_main:
        if sorted(index for index, _, _ in gathered) != sorted(selected):
            raise RuntimeError("Distributed attribution omitted or duplicated a source")
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_rows = []
        for index, result, faithfulness in sorted(gathered, key=lambda row: row[0]):
            folder = args.output / f"{index:05d}"
            folder.mkdir()
            record = records[result.video_id]
            video_path = Path(cache["dataset_root"]) / record.video_path
            figures = save_temporal_figures(result, folder, video_path=video_path,
                                            storyboard_count=args.storyboard_count)
            payload = result.metadata()
            payload.update(source_video=str(video_path), figures=[str(path.name) for path in figures],
                           faithfulness=faithfulness)
            (folder / "attribution.json").write_text(json.dumps(payload, indent=2))
            if args.full_maps:
                torch.save(dict(raw_attention=result.raw_attention,
                                signed_gradients=result.signed_gradients,
                                relevance_maps=result.relevance_maps), folder / "attention_maps.pt")
            manifest_rows.append(dict(index=index, video_id=result.video_id, folder=folder.name,
                                      target_value=result.target_value))
        (args.output / "manifest.json").write_text(json.dumps(dict(
            checkpoint=str(args.checkpoint.resolve()), cache=str(args.cache.resolve()),
            world_size=context.world_size, sample_count=len(gathered),
            prototype_source="training_fit" if args.prototype_class is not None else None,
            results=manifest_rows), indent=2))
    if context.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
