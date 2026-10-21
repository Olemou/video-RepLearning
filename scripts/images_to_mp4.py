#!/usr/bin/env python3
"""Convert categorized or spill/no-spill image-sequence folders into H.264 MP4s."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model.data.conversion import ConversionError, convert_dataset  # noqa: E402
from vision_model.data.labels import LABEL_MAP  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--fps", type=float, default=30.0)
    result.add_argument("--codec", default="libx264")
    result.add_argument(
        "--categories",
        nargs="+",
        default=["clean", "cold", "warm"],
        help=(
            "Source category directory names (default: clean cold warm). If these "
            "directories are absent, flat sequence folders are discovered and "
            "labeled from spill_/no_spill_ image filenames."
        ),
    )
    result.add_argument(
        "--label-map", nargs="+", default=[f"{category}={label}" for category, label in LABEL_MAP.items()],
        help="Binary labels as category=integer pairs.",
    )
    result.add_argument(
        "--resize-policy",
        choices=("error", "stretch", "letterbox", "crop"),
        default="letterbox",
        help="How frames are normalized to one even-sized canvas (default: letterbox).",
    )
    result.add_argument("--width", type=int)
    result.add_argument("--height", type=int)
    result.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--fail-fast", action="store_true")
    result.add_argument("--manifest", type=Path)
    result.add_argument("--errors", type=Path)
    result.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        label_map = {name: int(value) for name, value in (item.split("=", 1) for item in args.label_map)}
        records, failures = convert_dataset(
            args.input_root,
            args.output_root,
            fps=args.fps,
            codec=args.codec,
            recursive=args.recursive,
            resize_policy=args.resize_policy,
            width=args.width,
            height=args.height,
            overwrite=args.overwrite,
            validate=args.validate,
            manifest_path=args.manifest,
            errors_path=args.errors,
            fail_fast=args.fail_fast,
            categories=args.categories,
            label_map=label_map,
        )
    except ConversionError as exc:
        logging.error("Conversion aborted: %s", exc)
        return 2
    logging.info("Converted/validated %d sequences; %d failed", len(records), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
