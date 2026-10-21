"""User-friendly entry point for the browser spill-mask editor.

Example:
    python -m vision_model.annotate_spill --path dataset/spill_annotations
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    # Module usage means "open the editor" by default. The original script
    # retains --annotate so it can still be used only to export frames.
    if "--annotate" not in sys.argv:
        sys.argv.append("--annotate")
    runpy.run_path(str(scripts / "annotate_spill_masks.py"), run_name="__main__")


if __name__ == "__main__":
    main()
