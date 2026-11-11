#!/usr/bin/env python3
"""Synthetic attribution smoke example; figures do not represent trained spill evidence."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vision_model.interpretability.temporal_attribution import AttributionTarget, attribute_trajectory
from vision_model.interpretability.video_visualization import save_temporal_figures
from vision_model.training.trajectory import TrajectoryModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(239)
    model = TrajectoryModel(16, 16, 8).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.up.weight, std=.04)
    features = torch.randn(12, 16)
    valid = torch.tensor([True] * 9 + [False] * 3)
    timestamps = torch.tensor([0., .3, .7, 1., 1.5, 2.2, 2.8, 3.1, 3.9, 0., 0., 0.])
    result = attribute_trajectory(model, features, valid, timestamps,
        AttributionTarget("state", timestep=7, component=0), video_id="synthetic_example",
        sample_id="synthetic", frame_indices=torch.arange(12),
        checkpoint_id="random_synthetic_weights", include_maps=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    save_temporal_figures(result, args.output)
    (args.output / "attribution.json").write_text(json.dumps(result.metadata(), indent=2))
    print(f"Synthetic figures written to {args.output}")


if __name__ == "__main__":
    main()
