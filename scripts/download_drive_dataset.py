#!/usr/bin/env python3
"""Download the configured public Google Drive image-sequence dataset."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_model.data.google_drive import DriveDownloadError, download_public_tree  # noqa: E402


def folder_id(value: str) -> str:
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", value)
    candidate = match.group(1) if match else value
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise argparse.ArgumentTypeError("Expected a Drive folder URL or folder ID")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=folder_id, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["clean", "cold", "warm"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        records = download_public_tree(
            args.folder,
            args.output_root,
            categories=tuple(args.categories),
            overwrite=args.overwrite,
            workers=args.workers,
        )
    except DriveDownloadError as exc:
        logging.error("Download failed: %s", exc)
        return 2
    logging.info("Downloaded/verified %d files", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
