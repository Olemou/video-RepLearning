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
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform, ThermalAugmentor
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.encoder import load_pretrained_encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thermal-views", type=int, default=0,
                   help="Additional independently seeded full-frame thermal views per video")
    args = p.parse_args()
    if args.thermal_views < 0:
        p.error("--thermal-views cannot be negative")
    torch.set_num_threads(4)
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    transform = OfficialVJEPA2Transform(model_id, preserve_field_of_view=True)
    augmentor = ThermalAugmentor()
    augmented_transform = OfficialVJEPA2Transform(model_id, preserve_field_of_view=True,
                                                 augmentation=augmentor) if args.thermal_views else None
    dataset = VideoManifestDataset(args.manifest, dataset_root=args.dataset_root,
        sampling=TemporalSamplingConfig(clip_length=64, target_fps=4, training=False,
                                         short_video_policy="repeat_last"), transform=transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_pretrained_encoder(model_id, local_files_only=True)
    model.freeze_encoder(True)
    encoder = model.encoder.to(device=device, dtype=dtype).eval()
    rows, seen = [], {}
    for i, record in enumerate(dataset.records):
        digest = hashlib.sha256((args.dataset_root / record.video_path).read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != record.label:
                raise ValueError("Identical videos have conflicting labels")
            continue
        seen[digest] = record.label
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
                clip = pixels[:end * 2].permute(1, 0, 2, 3)[None].to(device=device, dtype=dtype)
                with torch.inference_mode(), torch.autocast(device_type=device,
                        dtype=dtype, enabled=device == "cuda"):
                    tokens = encoder(clip)
                    spatial = (clip.shape[-2] // 16) * (clip.shape[-1] // 16)
                    features.append(tokens[:, -spatial:].mean(1).float().cpu()[0])
            extracted.append(torch.stack(features))
        ends = item["frame_indices"][:2 * length][1::2].float()
        rows.append(dict(features=torch.stack(extracted), timestamps=ends / record.fps,
                         label=record.label, id=record.video_id, sha256=digest,
                         augmentation_seeds=augmentation_seeds,
                         source_sequence=record.source_sequence))
        print(f"encoded {len(rows)} unique videos: {record.video_id}, {length} causal prefixes x {len(views)} views", flush=True)
    rows.sort(key=lambda r: r["id"])
    max_t = max(r["features"].shape[1] for r in rows)
    features = torch.zeros(len(rows), 2 + args.thermal_views, max_t, rows[0]["features"].shape[-1])
    valid = torch.zeros(len(rows), max_t, dtype=torch.bool)
    timestamps = torch.zeros(len(rows), max_t)
    for i, row in enumerate(rows):
        t = row["features"].shape[1]
        features[i, :, :t] = row["features"]
        valid[i, :t] = True
        timestamps[i, :t] = row["timestamps"]
    payload = dict(features=features[:, :2], valid=valid, timestamps=timestamps,
        labels=torch.tensor([r["label"] for r in rows]), ids=[r["id"] for r in rows],
        video_sha256=[r["sha256"] for r in rows], source_sequences=[r["source_sequence"] for r in rows],
        temporal_context_policy="past_only_prefix_last_tubelet_v1", model_id=model_id,
        normalization=transform.normalization, spatial_policy="mean_letterbox_full_frame",
        augmentation="identity_and_whole_clip_horizontal_flip", sampling=dataset.sampler.config.__dict__,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        manifest=str(args.manifest.resolve()), dataset_root=str(args.dataset_root.resolve()))
    if args.thermal_views:
        payload.update(augmented_features=features[:, 2:],
                       thermal_augmentation_policy=augmentor.policy,
                       augmentation_seeds=[r["augmentation_seeds"] for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(dict(output=str(args.output), shape=list(features.shape))), flush=True)


if __name__ == "__main__":
    main()
