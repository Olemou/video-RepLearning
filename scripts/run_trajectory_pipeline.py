#!/usr/bin/env python3
"""Run balanced causal V-JEPA feature extraction, DDP training, and video attribution.

The pretrained V-JEPA 2 encoder stays frozen. Two ranks encode disjoint source
videos and train the trajectory adapter with DDP; one process renders the final
single-video attribution from the trained checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "extract", "train", "visualize")


def commands(args: argparse.Namespace) -> dict[str, list[str]]:
    python = sys.executable
    distributed = [python, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={args.nproc_per_node}"]
    output = args.output_root
    prepare = [python, str(ROOT / "scripts/prepare_balanced_videos.py"),
                    "--manifest", str(args.manifest), "--dataset-root", str(args.dataset_root),
                    "--output-dir", str(output / "balanced"), "--seed", str(args.seed)]
    if args.videos_per_class is not None:
        prepare += ["--per-class", str(args.videos_per_class)]
    return {
        "prepare": prepare,
        "extract": distributed + [str(ROOT / "scripts/extract_causal_video_features.py"),
                    "--manifest", str(output / "balanced/manifest.jsonl"),
                    "--dataset-root", str(args.dataset_root),
                    "--checkpoint-dir", str(args.encoder_dir),
                    "--output", str(output / "causal_features.pt"),
                    "--thermal-views", str(args.thermal_views),
                    "--clip-length", str(args.clip_length),
                    "--target-fps", str(args.target_fps)],
        "train": distributed + [str(ROOT / "scripts/train_trajectory.py"),
                    "--cache", str(output / "causal_features.pt"),
                    "--output", str(output / "training"),
                    "--epochs", str(args.epochs), "--seed", str(args.seed),
                    "--learning-rate", str(args.learning_rate), "--curriculum-epochs", "20",
                    "--align-weight", "0.1", "--predict-weight", "1.0",
                    "--keep-weight", "0.1"],
        "visualize": [python, str(ROOT / "scripts/visualize_trajectory_video.py"),
                      "--video", str(args.visualize_video),
                      "--model", str(output / "training/checkpoint.pt"),
                      "--encoder-dir", str(args.encoder_dir),
                      "--target", "semantic_similarity",
                      "--prototype-class", str(args.prototype_class),
                      "--output", str(output / "visualization")],
    }


def contract(args: argparse.Namespace) -> dict:
    """Settings that must remain fixed when continuing an interrupted run."""
    return dict(manifest=str(args.manifest.resolve()),
                dataset_root=str(args.dataset_root.resolve()),
                encoder_dir=str(args.encoder_dir.resolve()),
                visualize_video=str(args.visualize_video.resolve()),
                seed=args.seed, thermal_views=args.thermal_views,
                videos_per_class=args.videos_per_class,
                learning_rate=args.learning_rate,
                clip_length=args.clip_length, target_fps=args.target_fps,
                nproc_per_node=args.nproc_per_node,
                prototype_class=args.prototype_class)


def _completed_epochs(checkpoint: Path) -> int:
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "trajectory_resumable_v1":
        raise ValueError(f"Checkpoint is not resumable: {checkpoint}")
    return int(saved["epochs_completed"])


def _preflight(args: argparse.Namespace, *, need_gpu: bool) -> None:
    for path in (args.manifest, args.visualize_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("config.json", "model.safetensors", "video_preprocessor_config.json"):
        if not (args.encoder_dir / name).is_file():
            raise FileNotFoundError(args.encoder_dir / name)
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError(f"{name} is required on PATH")
    if need_gpu:
        import torch
        available = torch.cuda.device_count()
        if available < args.nproc_per_node:
            raise RuntimeError(f"Requested {args.nproc_per_node} GPUs but PyTorch sees {available}. "
                               "Run on the allocated GPU node with the torchgpu environment.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/home/olemou/videos/manifest.jsonl"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/olemou/videos"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("/home/olemou/hf"))
    parser.add_argument("--visualize-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New experiment directory; use --resume to continue it")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total target epochs, including any completed before --resume")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Fixed AdamW learning rate for the trajectory adapter")
    parser.add_argument("--thermal-views", type=int, default=8)
    parser.add_argument("--videos-per-class", type=int,
                        help="Use exactly this many byte-unique sources per class")
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--target-fps", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--prototype-class", type=int, choices=(0, 1), default=1)
    parser.add_argument("--stop-after", choices=STAGES, default="visualize",
                        help="Finish an earlier stage, then continue with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without checking GPUs or writing files")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.seed < 0 or args.thermal_views < 0 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            (args.videos_per_class is not None and args.videos_per_class < 2) or
            args.nproc_per_node < 1 or args.clip_length < 4 or args.clip_length % 2 or
            args.target_fps <= 0):
        parser.error("Invalid epochs, GPU count, seed, thermal views, clip length, or FPS")
    args.output_root = args.output_root.resolve()
    planned = commands(args)
    stages = STAGES[:STAGES.index(args.stop_after) + 1]
    if args.dry_run:
        for stage in stages:
            print(f"[{stage}] {shlex.join(planned[stage])}", flush=True)
        return
    settings_path = args.output_root / "pipeline_settings.json"
    expected = contract(args)
    if args.resume:
        if not settings_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {settings_path}")
        if json.loads(settings_path.read_text()) != expected:
            raise ValueError("Pipeline settings differ from the existing run")
    elif args.output_root.exists():
        raise FileExistsError(f"Use a new --output-root or --resume: {args.output_root}")
    _preflight(args, need_gpu=any(stage != "prepare" for stage in stages))
    if not args.resume:
        args.output_root.mkdir(parents=True)
        settings_path.write_text(json.dumps(expected, indent=2) + "\n")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"]
                                             if env.get("PYTHONPATH") else "")
    outputs = dict(prepare=args.output_root / "balanced/manifest.jsonl",
                   extract=args.output_root / "causal_features.pt",
                   train=args.output_root / "training/report.json",
                   visualize=args.output_root / "visualization/attribution.json")
    for stage in stages:
        if args.resume and outputs[stage].is_file():
            if stage != "train":
                print(f"[{stage}] existing output: {outputs[stage]}", flush=True)
                continue
            completed = _completed_epochs(args.output_root / "training/checkpoint.pt")
            if completed >= args.epochs:
                print(f"[train] already completed {completed} epochs", flush=True)
                continue
        command = planned[stage].copy()
        if stage == "train" and args.resume and (args.output_root / "training/checkpoint.pt").is_file():
            command.append("--resume")
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if not outputs[stage].is_file():
            raise RuntimeError(f"{stage} finished without expected artifact: {outputs[stage]}")
    print(json.dumps(dict(output_root=str(args.output_root), last_stage=stages[-1],
                          expected_outputs={name: str(path) for name, path in outputs.items()})),
          flush=True)


if __name__ == "__main__":
    main()
