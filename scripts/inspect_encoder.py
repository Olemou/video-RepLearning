#!/usr/bin/env python3
"""Load the exact pretrained checkpoint and run one video forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from vision_model.data.sampling import TemporalSamplingConfig  # noqa: E402
from vision_model.data.transforms import OfficialVJEPA2Transform  # noqa: E402
from vision_model.data.video_dataset import VideoManifestDataset  # noqa: E402
from vision_model.models.encoder import load_pretrained_encoder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset" / "manifest.jsonl")
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args(argv)
    dtype = getattr(torch, args.dtype)
    # Encoder parity is measured against the original checkpoint preprocessing.
    # Thermal normalization is the project default everywhere else.
    transform = OfficialVJEPA2Transform(args.model_id, thermal_config=None)
    dataset = VideoManifestDataset(
        args.manifest,
        sampling=TemporalSamplingConfig(training=False, short_video_policy="repeat_last"),
        transform=transform,
    )
    item = dataset[0]
    model = load_pretrained_encoder(args.model_id)
    model.to(device=args.device, dtype=dtype).eval()
    values = item["pixel_values_videos"].unsqueeze(0).to(args.device)
    with torch.inference_mode(), torch.autocast(
        device_type=torch.device(args.device).type,
        dtype=dtype,
        enabled=dtype != torch.float32,
    ):
        valid_mask = item["temporal_valid_mask"].unsqueeze(0).to(args.device)
        output = model(values, temporal_valid_mask=valid_mask)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "video_id": item["video_id"],
                "input_shape": list(values.shape),
                "token_shape": list(output.tokens.shape),
                "embedding_shape": list(output.embedding.shape),
                "temporal_mean_shape": list(output.temporal_mean.shape),
                "temporal_variance_shape": list(output.temporal_variance.shape),
                "video_variance_shape": list(output.video_variance.shape),
                "valid_tubelets": int(output.temporal_valid_mask.sum()),
                "label": item["label"],
                "dtype": str(output.tokens.dtype),
                "weight_dtype": str(next(model.parameters()).dtype),
                "finite": bool(torch.isfinite(output.tokens).all()),
                "embedding_norm": float(output.embedding.float().norm()),
                "mean": float(output.tokens.float().mean()),
                "std": float(output.tokens.float().std()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
