#!/usr/bin/env python3
"""Cache genuine frozen past-only prefix features for both real video views."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.distributed as dist
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform, ThermalAugmentor
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.distributed import initialize_distributed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thermal-views", type=int, default=0,
                   help="Additional independently seeded full-frame thermal views per video")
    p.add_argument("--clip-length", type=int, default=64,
                   help="Even number of sampled frames; 64 in the standard experiment")
    p.add_argument("--target-fps", type=float, default=4.,
                   help="Sampling FPS; 4 in the standard experiment")
    p.add_argument("--checkpoint-dir", type=Path,
                   help="Local files for facebook/vjepa2-vitl-fpc64-256")
    args = p.parse_args()
    if args.thermal_views < 0 or args.clip_length < 4 or args.clip_length % 2 or args.target_fps <= 0:
        p.error("Need nonnegative thermal views, even clip length >= 4, and positive target FPS")
    torch.set_num_threads(4)
    context = initialize_distributed()
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    model_source = str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else model_id
    if args.checkpoint_dir:
        for filename in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
            if not (args.checkpoint_dir / filename).is_file():
                raise FileNotFoundError(args.checkpoint_dir / filename)
    transform = OfficialVJEPA2Transform(model_source, preserve_field_of_view=True)
    augmentor = ThermalAugmentor()
    augmented_transform = OfficialVJEPA2Transform(model_source, preserve_field_of_view=True,
                                                 augmentation=augmentor) if args.thermal_views else None
    dataset = VideoManifestDataset(args.manifest, dataset_root=args.dataset_root,
        sampling=TemporalSamplingConfig(clip_length=args.clip_length, target_fps=args.target_fps, training=False,
                                         short_video_policy="repeat_last"), transform=transform)
    device = context.device.type
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_pretrained_encoder(model_source, local_files_only=True)
    model.freeze_encoder(True)
    encoder = model.encoder.to(device=context.device, dtype=dtype).eval()
    rows, seen = [], {}
    # Fail before expensive encoding; a balanced manifest must contain unique sources.
    for record in dataset.records:
        digest = hashlib.sha256((args.dataset_root / record.video_path).read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Byte-identical sources: {seen[digest]} and {record.video_id}")
        seen[digest] = record.video_id
    for i, record in enumerate(dataset.records):
        if i % context.world_size != context.rank:
            continue
        digest = hashlib.sha256((args.dataset_root / record.video_path).read_bytes()).hexdigest()
        item = dataset[i]
        length = int(item["temporal_valid_mask"].sum()) // 2
        if length < 2:
            raise ValueError("Each video needs at least two observed tubelets")
        views = [item["pixel_values_videos"][:2 * length],
                 transform(item["video"].flip(-1))[:2 * length]]
        augmentation_seeds = [239 + i * 1000 + v for v in range(args.thermal_views)]
        for seed in augmentation_seeds:
            views.append(augmented_transform(item["video"][:2 * length],
                         generator=torch.Generator().manual_seed(seed)))
        extracted = []
        for pixels in views:
            features = []
            for end in range(1, length + 1):
                clip = pixels[:end * 2].permute(1, 0, 2, 3)[None].to(device=context.device, dtype=dtype)
                with torch.inference_mode(), torch.autocast(device_type=device,
                        dtype=dtype, enabled=device == "cuda"):
                    tokens = encoder(clip)
                    spatial = (clip.shape[-2] // 16) * (clip.shape[-1] // 16)
                    features.append(tokens[:, -spatial:].mean(1).float().cpu()[0])
            extracted.append(torch.stack(features))
        ends = item["frame_indices"][:2 * length][1::2].float()
        rows.append(dict(features=torch.stack(extracted), timestamps=ends / record.fps,
                         frame_indices=ends.long(),
                         label=record.label, id=record.video_id, sha256=digest,
                         augmentation_seeds=augmentation_seeds,
                         source_sequence=record.source_sequence))
    if context.world_size > 1:
        gathered = [None] * context.world_size if context.is_main else None
        dist.gather_object(rows, gathered, dst=0)
        if not context.is_main:
            dist.destroy_process_group()
            return
        rows = [row for shard in gathered for row in shard]
    rows.sort(key=lambda r: r["id"])
    print(json.dumps({"encoded_sources": len(rows), "world_size": context.world_size}), flush=True)
    max_t = max(r["features"].shape[1] for r in rows)
    features = torch.zeros(len(rows), 2 + args.thermal_views, max_t, rows[0]["features"].shape[-1])
    valid = torch.zeros(len(rows), max_t, dtype=torch.bool)
    timestamps = torch.zeros(len(rows), max_t)
    frame_indices = torch.full((len(rows), max_t), -1, dtype=torch.long)
    for i, row in enumerate(rows):
        t = row["features"].shape[1]
        features[i, :, :t] = row["features"]
        valid[i, :t] = True
        timestamps[i, :t] = row["timestamps"]
        frame_indices[i, :t] = row["frame_indices"]
    payload = dict(features=features[:, :2], valid=valid, timestamps=timestamps,
        frame_indices=frame_indices,
        labels=torch.tensor([r["label"] for r in rows]), ids=[r["id"] for r in rows],
        video_sha256=[r["sha256"] for r in rows], source_sequences=[r["source_sequence"] for r in rows],
        temporal_context_policy="past_only_prefix_last_tubelet_v1", model_id=model_id,
        checkpoint_source=model_source,
        checkpoint_sha256=(hashlib.sha256((args.checkpoint_dir / "model.safetensors").read_bytes()).hexdigest()
                           if args.checkpoint_dir else None),
        label_schema=LABEL_SCHEMA, label_map=LABEL_MAP,
        categories=[next(record.category for record in dataset.records if record.video_id == row["id"])
                    for row in rows],
        distributed_world_size=context.world_size,
        normalization=transform.normalization, spatial_policy="mean_letterbox_full_frame",
        augmentation="identity_and_whole_clip_horizontal_flip", sampling=dataset.sampler.config.__dict__,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        manifest=str(args.manifest.resolve()), dataset_root=str(args.dataset_root.resolve()))
    if args.thermal_views:
        payload.update(augmented_features=features[:, 2:],
                       thermal_augmentation_policy=augmentor.policy,
                       augmentation_seeds=[r["augmentation_seeds"] for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Feature cache already exists: {args.output}")
    torch.save(payload, args.output)
    print(json.dumps(dict(output=str(args.output), shape=list(features.shape))), flush=True)
    if context.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
