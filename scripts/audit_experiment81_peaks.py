#!/usr/bin/env python3
"""Automatic peak locations, independent of heatmap display scaling."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from PIL import Image, ImageDraw, ImageFont
from control_experiment81 import content_cells
from report_experiment81_spatial import maps_from
from vision_model.interpretability.map_geometry import letterbox_bounds


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    out = a.root / "native_peak_review"; out.mkdir(exist_ok=False)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    maps = {case: maps_from(a.root / "spatial" / case)[0] for case in ("frozen", "full")}
    rows = []
    for case in maps:
        for i, name in enumerate(c["ids"]):
            h, w = c["rgb"][i].shape[-2:]
            support = content_cells(h, w).flatten().nonzero().flatten()
            top, left, rh, rw = letterbox_bounds(h, w)
            for method in ("final_attention_gradient", "multilayer_positive"):
                for time, field in enumerate(maps[case][i][method]):
                    native = field.flatten()
                    pos = int(support[native[support].argmax()])
                    x = ((pos % 16 + .5) * 16 - left) * w / rw
                    y = ((pos // 16 + .5) * 16 - top) * h / rh
                    rows.append(dict(case=case, id=name, tubelet=time, method=method,
                        native_peak=[pos // 16, pos % 16], source_xy=[x, y], peak_value=float(native[pos]),
                        content_median_value=float(native[support].median()),
                        peak_to_content_median=float(native[pos] / native[support].median().clamp_min(1e-30))))
    (out / "all80_native_peaks.json").write_text(json.dumps(dict(rows=rows,
        scope="Automatic maxima of raw all-head conditional sensitivity over valid content; no color normalization",
        warning="Markers are automatic attribution maxima, not manual labels or localization ground truth",
        manual_spatial_annotations=0, localization_accuracy_available=False), indent=2) + "\n")
    font = ImageFont.load_default(size=16)
    for name in ("cold_folderImage56", "cold_folderImage10", "cold_folderImage13"):
        i = c["ids"].index(name)
        frame = c["rgb"][i][-1].permute(1, 2, 0).numpy()
        h, w = frame.shape[:2]
        canvas = Image.new("RGB", (3 * w, h + 75), "white"); draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), name + " | Automatic strongest response; markers are NOT training annotations", fill="black", font=font)
        for col, method in enumerate((None, "final_attention_gradient", "multilayer_positive")):
            canvas.paste(Image.fromarray(frame), (col * w, 75))
            draw.text((col * w + 4, 32), "Original" if method is None else "Full: " + method, fill="black", font=font)
            if method is not None:
                r = next(r for r in rows if r["case"] == "full" and r["id"] == name and r["tubelet"] == 5 and r["method"] == method)
                x, y = r["source_xy"]; x += col * w; y += 75
                draw.line((x - 10, y, x + 10, y), fill="yellow", width=3)
                draw.line((x, y - 10, x, y + 10), fill="yellow", width=3)
                draw.text((col * w + 4, 55), f"peak {r['peak_value']:.3g}; peak/median {r['peak_to_content_median']:.2f}", fill="black", font=font)
        canvas.save(out / f"{name}.png")


if __name__ == "__main__":
    main()
