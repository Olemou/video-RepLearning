#!/usr/bin/env python3
"""Validate a generated MP4 dataset against its JSONL manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model.data.conversion import ConversionError  # noqa: E402
from vision_model.data.validation import render_report, validate_dataset, write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--short-threshold", type=int, default=64)
    parser.add_argument(
        "--categories", nargs="+", default=["clean", "cold", "warm"]
    )
    parser.add_argument("--decode-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_validation.md",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = args.manifest or args.dataset_root / "manifest.jsonl"
    try:
        summary = validate_dataset(
            args.dataset_root,
            manifest,
            source_root=args.source_root,
            short_threshold=args.short_threshold,
            decode_all=args.decode_all,
            categories=tuple(args.categories),
        )
    except (OSError, ConversionError) as exc:
        logging.error("Validation aborted: %s", exc)
        return 2
    report = render_report(summary, args.dataset_root, manifest)
    write_report(args.report, report)
    print(report)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
