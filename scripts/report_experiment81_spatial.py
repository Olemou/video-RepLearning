#!/usr/bin/env python3
"""Shared-scale spatial gallery and honest removal summaries for Experiment81."""
from __future__ import annotations
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from vision_model.interpretability.map_geometry import (inverse_letterbox_map,
    spatial_evidence_overlay, nonnegative_evidence_colors)
from control_experiment81 import content_cells
from prepare_experiment81 import digest


def maps_from(path):
    rows, summaries = {}, []
    for file in sorted(path.glob("group_*.pt")):
        g = torch.load(file, map_location="cpu", weights_only=False)
        summaries.append(g["summary"])
        for j, index in enumerate(g["ids"].tolist()):
            if index in rows:
                raise RuntimeError("Repeated source maps")
            rows[index] = {k: g[k][j] for k in ("positive_gradients", "signed_gradients", "multilayer_positive",
                                               "final_attention_gradient", "raw_final_attention", "rollout")}
    return rows, summaries


def fixed_scale(all_maps, train, c, key):
    values = []
    for rows in all_maps.values():
        for i in train:
            h, w = c["rgb"][i].shape[-2:]
            support = content_cells(h, w)
            values.append(rows[i][key][:, support].flatten())
    values = torch.cat(values)
    return max(float(torch.quantile(values, .995)), 1e-20)


def removal_summary(root):
    rows = []
    for case in ("frozen", "full"):
        path = root / "input_controls" / case / "results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for label, label_name in ((1, "cold"), (0, "clean")):
            selected = [r for r in data["sources"] if r["split"] == "selection" and r["label"] == label]
            for fraction in data["fractions"]:
                for fill in ("mean", "blur33"):
                    deltas = []
                    for source in selected:
                        changes = [p for p in source["perturbations"] if p["fraction"] == fraction and p["fill"] == fill]
                        top = next(p["margin_drop"] for p in changes if p["kind"] == "top")
                        random = np.mean([p["margin_drop"] for p in changes if p["kind"].startswith("random_")])
                        deltas.append(float(top - random))
                    generator = np.random.default_rng(239)
                    samples = generator.choice(deltas, size=(2000, len(deltas)), replace=True).mean(1)
                    rows.append(dict(case=case, category=label_name, fraction=fraction, fill=fill,
                        n=len(deltas), mean_high_minus_random_drop=float(np.mean(deltas)),
                        exploratory_bootstrap95=np.quantile(samples, [.025, .975]).tolist(),
                        source_differences=deltas, localization_metric=False))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    out = a.root / "spatial_gallery"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    all_maps, summaries = {}, {}
    for case in ("frozen", "full"):
        all_maps[case], summaries[case] = maps_from(a.root / "spatial" / case)
        if sorted(all_maps[case]) != list(range(len(c["ids"]))):
            raise ValueError("The spatial gallery must contain all 80 unique sources")
    train = c["train_indices"].tolist()
    scales = {key: fixed_scale(all_maps, train, c, key) for key in
              ("final_attention_gradient", "multilayer_positive", "rollout", "raw_final_attention")}
    # Selected-layer figures use one scale pooled over the same prespecified
    # layers and all training sources/arms, never a favorable head/frame.
    chosen = [0, 5, 11, 17, 23]
    layer_values = []
    for rows in all_maps.values():
        for i in train:
            support = content_cells(*c["rgb"][i].shape[-2:])
            layer_values.append(rows[i]["positive_gradients"][chosen][:, :, support].flatten())
    layer_scale = max(float(torch.quantile(torch.cat(layer_values), .995)), 1e-20)
    metadata = dict(scales=scales, layer_scale=layer_scale, selected_layers_zero_based=chosen,
        calibration="Training sources only; pooled across frozen/full arms; q99.5 magnitude scale",
        display_threshold=0, per_frame_normalization=False, smoothing=False, manual_roi=False,
        probability_maps=False, raw_rollout_class_specific=False, tubelet_frames=2,
        attribution_summaries=summaries)
    (out / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    font = ImageFont.load_default(size=15)
    entries = []
    for i, name in enumerate(c["ids"]):
        folder = out / name; folder.mkdir()
        rgb = c["rgb"][i].permute(0, 2, 3, 1)
        h, w = rgb.shape[1:3]
        images = []
        storyboard = Image.new("RGB", (5 * w, 65 + 6 * (h + 30)), "white")
        draw = ImageDraw.Draw(storyboard)
        split = "TRAIN" if i in train else "SOURCE SELECTION"
        draw.text((5, 2), f"{name} | {split} | conditional attention-key gradients; not masks/probabilities", fill="black", font=font)
        titles = ["Original", "Frozen: final A*gradient", "Full: final A*gradient", "Full: all-layer gradient", "Full: rollout (class-agnostic)"]
        for j, title in enumerate(titles):
            draw.text((j * w + 4, 29), title, fill="black", font=font)
        for frame in range(len(rgb)):
            t = frame // 2
            panels = [rgb[frame].numpy()]
            for case, key in (("frozen", "final_attention_gradient"), ("full", "final_attention_gradient"),
                              ("full", "multilayer_positive")):
                overlay = spatial_evidence_overlay(rgb[frame], all_maps[case][i][key][t], scale=scales[key],
                    scope="conditional_attention_key_gradient", signed=False, style="blue_red", alpha=.4)
                panels.append(overlay.numpy())
            roll = inverse_letterbox_map(all_maps["full"][i]["rollout"][t], h, w, interpolation="nearest")
            panels.append(nonnegative_evidence_colors(roll, scale=scales["rollout"]).numpy())
            image = np.concatenate(panels, 1)
            images.append(image)
            if frame % 2:
                y = 65 + t * (h + 30)
                draw.text((4, y), f"source frame {int(c['frame_indices'][i, frame])}; end {float(c['timestamps'][i,t]):.3f}s; tubelet {t}",
                          fill="black", font=font)
                storyboard.paste(Image.fromarray(image), (0, y + 25))
        storyboard.save(folder / "comparison.png")
        Image.fromarray(images[-1]).save(folder / "endpoint.png")
        command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{5*w}x{h}",
                   "-framerate", "4", "-i", "-", "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(folder / "comparison.mp4")]
        subprocess.run(command, input=np.stack(images).tobytes(), check=True)
        layer_image = Image.new("RGB", (6 * w, h + 32), "white"); ld = ImageDraw.Draw(layer_image)
        layer_image.paste(Image.fromarray(rgb[-1].numpy()), (0, 32)); ld.text((3, 4), "Original; last tubelet", fill="black", font=font)
        for column, layer in enumerate(chosen, 1):
            field = all_maps["full"][i]["positive_gradients"][layer, -1]
            overlay = spatial_evidence_overlay(rgb[-1], field, scale=layer_scale,
                scope="conditional_attention_key_gradient", signed=False, style="blue_red", alpha=.4)
            layer_image.paste(Image.fromarray(overlay.numpy()), (column * w, 32))
            ld.text((column * w + 3, 4), f"Full block {layer}, all heads", fill="black", font=font)
        layer_image.save(folder / "selected_layers.png")
        entries.append(dict(id=name, label=int(c["labels"][i]), split=split, image=f"{name}/comparison.png",
                            endpoint=f"{name}/endpoint.png", video=f"{name}/comparison.mp4", layers=f"{name}/selected_layers.png"))
    overview = Image.new("RGB", (4 * 480, 20 * 145), "white")
    for slot, entry in enumerate(entries):
        tile = Image.new("RGB", (480, 145), "white"); td = ImageDraw.Draw(tile)
        td.text((3, 1), entry["id"] + " | " + entry["split"], fill="black", font=ImageFont.load_default(size=12))
        endpoint = Image.open(out / entry["endpoint"])
        # Original and trained final-layer map, side-by-side, all sources.
        ow = endpoint.width // 5
        left = endpoint.crop((0, 0, ow, endpoint.height)).resize((240, 120))
        right = endpoint.crop((2 * ow, 0, 3 * ow, endpoint.height)).resize((240, 120))
        tile.paste(left, (0, 22)); tile.paste(right, (240, 22))
        overview.paste(tile, ((slot % 4) * 480, (slot // 4) * 145))
    overview.save(out / "all80_overview.png")
    controls = removal_summary(a.root)
    (out / "input_control_summary.json").write_text(json.dumps(controls, indent=2) + "\n")
    links = []
    for e in entries:
        links.append(f'<article><h2>{html.escape(e["id"])} — {e["split"]}</h2>'
            f'<a href="{e["image"]}"><img loading="lazy" src="{e["endpoint"]}" width="100%"></a>'
            f'<p><a href="{e["image"]}">All times</a> · <a href="{e["video"]}">Video</a> · '
            f'<a href="{e["layers"]}">Intermediate layers</a></p></article>')
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Experiment81 spatial audit</title>'
        '<style>body{font:16px sans-serif;max-width:1600px;margin:auto;padding:20px}article{margin:35px 0}</style>'
        '<h1>Experiment81: all 80 real sources</h1><p>Fixed training-only scales; all heads. '
        'Gradients are conditional attention sensitivity, not segmentation or pixel probability. '
        'Rollout is class-agnostic. Each map spans a two-frame tubelet. Blue means low displayed magnitude.</p>'
        '<p><a href="calibration.json">Scales and replay errors</a> · <a href="all80_overview.png">All-source overview</a></p>' + ''.join(links))
    (out / "inventory.json").write_text(json.dumps(entries, indent=2) + "\n")
    report = a.root / "report.md"
    if report.exists():
        extra = ["", "## Spatial outputs and input controls", "",
            "[Complete 80-source spatial gallery](spatial_gallery/index.html) · "
            "[All-source overview](spatial_gallery/all80_overview.png) · "
            "[Replay/calibration record](spatial_gallery/calibration.json)", "",
            "Known failures: [cold56](spatial_gallery/cold_folderImage56/comparison.png), "
            "[cold10](spatial_gallery/cold_folderImage10/comparison.png), "
            "[cold13](spatial_gallery/cold_folderImage13/comparison.png).", "",
            "These maps explain the instrumented explicit-attention forward. Its own gradient replay is checked, "
            "and its measured discrepancy from production SDPA is published rather than hidden. "
            "Raw attention/rollout and conditional attention-key sensitivity are not pixel-level spill predictions.", "",
            "| Arm | Selection category | Removed | Fill | Mean high-minus-random margin drop | Exploratory 95% interval |",
            "|---|---|---:|---|---:|---|"]
        for r in controls:
            ci = r["exploratory_bootstrap95"]
            extra.append(f"| {r['case']} | {r['category']} (n={r['n']}) | {r['fraction']:.0%} | {r['fill']} | "
                         f"{r['mean_high_minus_random_drop']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |")
        extra += ["", "Controls recompute the original-input path and keep training prototypes fixed. "
            "Positive high-minus-random drops support decision sensitivity, not liquid identity. "
            "The small source-bootstrap intervals are exploratory and unadjusted across variants. "
            "Clean-video responses measure spill-margin sensitivity; absence of a highlighted spill is not a localization failure.", "",
            "Qualitative spill acceptance must be recorded after inspecting the actual original/map pairs. "
            "No real IoU or pointing accuracy is available without independent ground truth.", ""]
        report.write_text(report.read_text() + "\n".join(extra))


if __name__ == "__main__":
    main()
