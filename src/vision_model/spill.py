"""Annotate spill masks and train the spill segmenter.

Commands:
    python -m vision_model.spill annotate --path dataset/spill_annotations
    python -m vision_model.spill train --path dataset/spill_annotations --epochs 100 --device cuda
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


HELP = __doc__.strip()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(HELP)
        return
    command = sys.argv.pop(1)
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    if command == "annotate":
        if "--annotate" not in sys.argv:
            sys.argv.append("--annotate")
        script = scripts / "annotate_spill_masks.py"
    elif command == "train":
        script = scripts / "train_spill_segmenter.py"
    else:
        raise SystemExit(f"Unknown command {command!r}. Use 'annotate' or 'train'.\n\n{HELP}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
