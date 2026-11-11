#!/usr/bin/env python3
"""All 24 blocks, all 80 sources and all six tubelets, without layer selection."""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from PIL import Image, ImageDraw, ImageFont
from control_experiment81 import content_cells
from prepare_experiment81 import digest
from report_experiment81_spatial import maps_from
from vision_model.interpretability.map_geometry import spatial_evidence_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    out = a.root / "all_blocks"; out.mkdir(exist_ok=False)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    maps = {case: maps_from(a.root / "spatial" / case)[0] for case in ("frozen", "full")}
    if any(sorted(m) != list(range(80)) for m in maps.values()):
        raise ValueError("Require all80 unique sources in both arms")
    values = []
    for rows in maps.values():
        for i in c["train_indices"].tolist():
            support = content_cells(*c["rgb"][i].shape[-2:])
            values.append(rows[i]["positive_gradients"][:, :, support].flatten())
    scale = max(float(torch.quantile(torch.cat(values), .995)), 1e-20)
    font = ImageFont.load_default(size=14)
    records, gallery = [], []
    known = {"cold_folderImage56", "cold_folderImage10", "cold_folderImage13"}
    for case, rows in maps.items():
        for i, name in enumerate(c["ids"]):
            rgb = c["rgb"][i].permute(0, 2, 3, 1)
            h, w = rgb.shape[1:3]
            support = content_cells(h, w)
            allowed = support.flatten().nonzero().flatten()
            field = rows[i]["positive_gradients"]
            if field.shape != (24, 6, 16, 16):
                raise ValueError(f"Unexpected all-block shape: {field.shape}")
            for t in range(6):
                for layer in range(24):
                    native = field[layer, t].flatten()
                    v = native[allowed]
                    pos = int(allowed[v.argmax()])
                    probability = v / v.sum().clamp_min(1e-30)
                    records.append(dict(case=case, id=name, label=int(c["labels"][i]),
                        split="train" if i in c["train_indices"] else "selection",
                        block_zero_based=layer, block_one_based=layer + 1, tubelet=t,
                        native_peak=[pos // 16, pos % 16], peak=float(v.max()), median=float(v.median()),
                        peak_to_median=float(v.max() / v.median().clamp_min(1e-30)),
                        top5percent_mass=float(v.topk(max(1, round(len(v) * .05))).values.sum() / v.sum().clamp_min(1e-30)),
                        normalized_spatial_entropy=float(-(probability * probability.clamp_min(1e-30).log()).sum() / torch.tensor(float(len(v))).log())))
                if t != 5 and name not in known:
                    continue
                canvas = Image.new("RGB", (4 * 320, 78 + 6 * 285), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text((5, 3), f"{name} | {case} | tubelet {t} | all24 blocks | fixed scale {scale:.4g}", fill="black", font=font)
                draw.text((5, 25), "All-head positive gradient of FINAL spill-minus-clean margin w.r.t. block attention", fill="black", font=font)
                draw.text((5, 47), "Yellow cross = automatic raw maximum; NOT a liquid label. Blue = low sensitivity, not verified background.", fill="black", font=font)
                for layer in range(24):
                    overlay = spatial_evidence_overlay(rgb[2 * t + 1], field[layer, t], scale=scale,
                        scope="conditional_attention_key_gradient", signed=False, style="blue_red", alpha=.4)
                    tile = Image.fromarray(overlay.numpy()).resize((320, 240))
                    r = records[-24 + layer]
                    # Both cohorts have 4:3 content: rows 2..13 of the checkpoint grid.
                    from vision_model.interpretability.map_geometry import letterbox_bounds
                    top, left, rh, rw = letterbox_bounds(h, w)
                    py, px = r["native_peak"]
                    x = ((px + .5) * 16 - left) * 320 / rw
                    y = ((py + .5) * 16 - top) * 240 / rh
                    marker = ImageDraw.Draw(tile)
                    marker.line((x-5, y, x+5, y), fill="yellow", width=2)
                    marker.line((x, y-5, x, y+5), fill="yellow", width=2)
                    col, row = layer % 4, layer // 4
                    sx, sy = col * 320, 78 + row * 285
                    draw.text((sx+3, sy), f"block {layer:02d} (layer {layer+1}) peak {r['peak']:.2g}", fill="black", font=font)
                    draw.text((sx+3, sy+20), f"peak/median {r['peak_to_median']:.2f}", fill="black", font=font)
                    canvas.paste(tile, (sx, sy+43))
                filename = f"{name}_{case}_tubelet{t}.png"
                canvas.save(out / filename)
                gallery.append(dict(id=name, case=case, tubelet=t, file=filename))
    (out / "measurements.json").write_text(json.dumps(dict(records=records,
        sources_per_arm=80, blocks=24, tubelets=6, manual_annotations=0,
        scope="Conditional attention-key sensitivity, not pixel probabilities or original layerwise LeGrad",
        localization_metric=False), indent=2) + "\n")
    (out / "calibration.json").write_text(json.dumps(dict(scale=scale,
        policy="Training-only q99.5 pooled across BOTH arms, ALL24 blocks, ALL six times, valid content",
        per_frame_normalization=False, per_layer_normalization=False, manual_roi=False,
        checkpoint_sha256={case: digest(a.root / case / "checkpoint.pt") for case in maps}), indent=2) + "\n")
    (out / "inventory.json").write_text(json.dumps(gallery, indent=2) + "\n")
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>All24 blocks</title>'
        '<h1>Every encoder block: frozen and full fine-tuning</h1><p>All80 sources at final sampled tubelet; '
        'all six tubelets for cold56, cold10, cold13. One shared training-only scale. '
        'Automatic maxima are not annotations or localization accuracy.</p>' + ''.join(
            f'<p><a href="{e["file"]}">{html.escape(e["id"])} / {e["case"]} / tubelet {e["tubelet"]}</a></p>' for e in gallery))
    print(json.dumps(dict(measurements=len(records), figures=len(gallery), scale=scale)), flush=True)


if __name__ == "__main__":
    main()
