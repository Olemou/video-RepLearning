#!/usr/bin/env python3
"""Prepare exact video views and immutable FP32-master/BF16-forward teachers."""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.distributed as dist
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform, ThermalAugmentor
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from vision_model.training.online_trajectory import OnlinePrefixEncoder, collect_prefix_features


DEFAULT_CACHE = ROOT / "reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/causal_features.pt"
DEFAULT_SPLIT = ROOT / "reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/training/report.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_bank(pixels, normalization):
    mean = torch.tensor(normalization["mean"]).reshape(1, 1, 3, 1, 1)
    std = torch.tensor(normalization["std"]).reshape(1, 1, 3, 1, 1)
    return ((pixels.float() * (1.0 / 255) - mean) / std).bfloat16()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--encoder-dir", type=Path, default=Path("/home/olemou/hf"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-sources", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(4)
    ctx = initialize_distributed()
    if ctx.device.type != "cuda":
        raise RuntimeError("Experiment81 preparation requires the actual allocated GPU")
    if a.output.exists():
        raise FileExistsError(a.output)
    c = torch.load(a.teacher_cache, map_location="cpu", weights_only=False)
    split = json.loads(a.split.read_text())
    if c["temporal_context_policy"] != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Not a causal teacher cache")
    if digest(a.encoder_dir / "model.safetensors") != c["checkpoint_sha256"]:
        raise ValueError("Encoder checkpoint changed")
    if digest(c["manifest"]) != c["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if len(set(c["video_sha256"])) != len(c["ids"]):
        raise ValueError("Duplicate sources")
    train = [c["ids"].index(i) for i in split["train_source_ids"]]
    selection = [c["ids"].index(i) for i in split["validation_source_ids"]]
    if sorted(train + selection) != list(range(len(c["ids"]))) or set(train) & set(selection):
        raise ValueError("Source split mismatch")
    transform = OfficialVJEPA2Transform(str(a.encoder_dir), preserve_field_of_view=True)
    dataset = VideoManifestDataset(c["manifest"], dataset_root=c["dataset_root"],
        sampling=TemporalSamplingConfig(**c["sampling"]), transform=None)
    lookup = {r.video_id: i for i, r in enumerate(dataset.records)}
    encoder = load_pretrained_encoder(str(a.encoder_dir), local_files_only=True).encoder
    encoder = OnlinePrefixEncoder(encoder).to(ctx.device).eval().requires_grad_(False)
    owned = list(range(ctx.rank, len(c["ids"]), ctx.world_size))
    _, counts = audit_distributed_indices(owned, len(c["ids"]))
    rows = []
    max_transform_error = 0.0
    for i in owned:
        item = dataset[lookup[c["ids"][i]]]
        record = dataset.records[lookup[c["ids"][i]]]
        if digest(Path(c["dataset_root"]) / record.video_path) != c["video_sha256"][i]:
            raise ValueError(f"Source changed: {c['ids'][i]}")
        valid = item["temporal_valid_mask"].reshape(-1, 2).all(1)
        if not torch.equal(valid, c["valid"][i]) or not torch.equal(item["frame_indices"][1::2][valid], c["frame_indices"][i][valid]):
            raise ValueError(f"Sampling mismatch: {c['ids'][i]}")
        frames = item["video"]
        views = [frames, frames.flip(-1)]
        seeds = c.get("augmentation_seeds", [[]] * len(c["ids"]))[i]
        augmentor = ThermalAugmentor()
        for seed in seeds:
            views.append(augmentor(frames, generator=torch.Generator().manual_seed(seed)))
        mean = torch.tensor(transform.normalization["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(transform.normalization["std"]).view(1, 3, 1, 1)
        official = torch.stack([transform(view) for view in views])
        # Store the exact letterboxed uint8 boundary, avoiding a 2x larger
        # normalized cache. Verify reconstruction against the official transform.
        pixels = ((official * std + mean) * 255).round().clamp(0, 255).byte()
        reconstructed = (pixels.float() * (1.0 / 255) - mean) / std
        error = float((official - reconstructed).abs().max())
        max_transform_error = max(max_transform_error, error)
        if error > 1e-6:
            raise RuntimeError(f"Official preprocessing reconstruction changed: {error}")
        normalized = normalize_bank(pixels, transform.normalization)
        # This execution boundary is shared by both arms: FP32 master weights,
        # BF16 inputs/autocast, unlike the historical BF16-weight extraction.
        teacher = collect_prefix_features(encoder, normalized[None], valid[None],
                                           ctx.device, chunk_sources=a.chunk_sources)[0].cpu()
        historical = torch.cat((c["features"][i], c["augmented_features"][i]), 0) if seeds else c["features"][i]
        difference = float((teacher[:, valid] - historical[:, valid]).abs().max())
        rows.append(dict(index=i, pixels=pixels, rgb=frames, teacher=teacher,
                         frame_indices=item["frame_indices"], historical_feature_max_error=difference))
        print(json.dumps(dict(stage="prepare", rank=ctx.rank, id=c["ids"][i],
                              views=len(views), historical_feature_max_error=difference)), flush=True)
    gathered = [None] * ctx.world_size if ctx.is_main else None
    if ctx.world_size > 1:
        dist.gather_object(rows, gathered, dst=0)
    else:
        gathered = [rows]
    if ctx.is_main:
        rows = sorted([r for shard in gathered for r in shard], key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(len(c["ids"]))):
            raise RuntimeError("Missing/duplicated sources")
        payload = dict(ids=c["ids"], labels=c["labels"], valid=c["valid"], timestamps=c["timestamps"],
            train_indices=torch.tensor(train), selection_indices=torch.tensor(selection),
            pixels=torch.stack([r["pixels"] for r in rows]), rgb=[r["rgb"] for r in rows],
            teacher=torch.stack([r["teacher"] for r in rows]),
            frame_indices=torch.stack([r["frame_indices"] for r in rows]),
            normalization=transform.normalization, source_sha256=c["video_sha256"],
            categories=c["categories"], augmentation_seeds=c.get("augmentation_seeds"),
            temporal_context_policy=c["temporal_context_policy"],
            metadata=dict(original_teacher_cache=str(a.teacher_cache.resolve()),
                original_teacher_sha256=digest(a.teacher_cache), encoder_dir=str(a.encoder_dir.resolve()),
                encoder_sha256=digest(a.encoder_dir / "model.safetensors"), split_sha256=digest(a.split),
                rank_source_counts=counts, world_size=ctx.world_size, host=socket.gethostname(),
                precision="FP32 immutable teacher weights; BF16 forward/input; FP32 pooled storage",
                historical_precision="BF16 weights; frozen historical targets differ numerically",
                historical_feature_max_error=max(r["historical_feature_max_error"] for r in rows),
                official_transform_reconstruction_max_error=max_transform_error,
                sampling=c["sampling"], manual_annotations=0, mil=False, decoder=False))
        a.output.mkdir(parents=True, exist_ok=False)
        torch.save(payload, a.output / "prepared.pt")
        (a.output / "metadata.json").write_text(json.dumps(payload["metadata"], indent=2) + "\n")
    if ctx.world_size > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
