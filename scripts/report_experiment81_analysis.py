#!/usr/bin/env python3
"""Report measured nuisance controls and gradient-free occlusion evidence."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    results = {case: json.loads((a.root / "shortcut_controls" / case / "results.json").read_text())
               for case in ("frozen", "full")}
    report = ["# Experiment81: deeper fixed-model analysis", "",
        "No additional optimizer updates or manual annotations were used for these controls. "
        "All classifier prototypes remain fixed at their original training-source values.", "",
        "## What the inputs actually cover", "",
        "The original sampler uses a **centered 12-frame window** at 4 FPS, not the first three seconds. "
        "Cold56 spans source frames 263–304 (about 17.5–20.3 seconds). "
        "The earlier chat wording about the first three seconds was incorrect and was corrected. "
        "Future-prefix causality holds within the observed window; centered offline sampling itself "
        "must not be described as a full streaming deployment protocol.", "",
        "Native resolutions are perfectly coupled with these source classes: all 40 clean videos "
        "are 384×288 and all 40 cold videos are 640×480. Both become the same full-field "
        "256×256 letterboxed encoder input; original dimensions are not supplied as a model feature. "
        "This establishes an acquisition-cohort confound, not proof that the network uses resolution. "
        "Resampling, scene composition and sensor/encoding characteristics remain possible cues.", "",
        "A train-only scalar timestamp probe obtained 62.5% selection accuracy and AUROC 0.5547. "
        "Absolute timing is not supported as the principal shortcut by that probe.", "",
        "## Fixed-model counterfactual results", "",
        "| Arm | Control | Selection correct | Selection AUROC |",
        "|---|---|---:|---:|"]
    for case, result in results.items():
        controls = {"original": result["original"], **result["image_controls"], **result["clock_controls"]}
        for kind, metrics in controls.items():
            m = metrics["selection"]
            report.append(f"| {case} | {kind} | {round(m['accuracy']*m['n'])}/{m['n']} | {m['roc_auc']:.4f} |")
    report += ["", "Spatial-mean inputs retain per-frame channel means and temporal mean changes but have no "
        "within-frame liquid shape. Their failure cannot rule out nuisance dependence: a classifier can "
        "need scene texture while still ignoring liquid. Fixed-gain and mean-matched controls share "
        "positive gain 0.5 and do not clip; mean matching removes the per-frame mean while preserving "
        "relative spatial contrast. Patch shuffling uses a single permutation throughout each clip, "
        "preserves the intensity histogram and local patches, and disrupts coherent region arrangement. "
        "Remaining patch texture can still include liquid; survival is not proof that only background is used.", "",
        "Relative/common/random-origin clock controls preserve image features, source order and real "
        "timestamp differences. Separate prediction/persistence measurements are in the raw control JSONs. "
        "Image-intervention tables report recognition rather than pretending unchanged teacher targets "
        "are ground truth for a changed physical video.", "",
        "The native-resolution shuffle diagnostic initially assumed 384×288 and stopped on cold 640×480 "
        "sources. It was corrected to use a geometry-preserving common-divisor grid, tested at multiple "
        "resolutions, and resumed without rerunning the three completed image controls. The original "
        "failed log and source snapshot are preserved. The training pipeline did not fail or change.", "",
        "## Gradient-free local decision controls", "",
        "Each of 48 fixed full-content regions is removed across the complete sampled clip, using "
        "mean and blur fills separately. Maps contain the actual change in the trained spill-minus-clean "
        "cosine margin. They bypass attention capture and local gradient saturation. These are "
        "episode-level counterfactual sensitivities, not framewise segmentation or material labels.", ""]
    gallery = a.root / "occlusion_gallery"; gallery.mkdir(exist_ok=False)
    scale = .01  # Predeclared common cosine-margin change, not fitted to a source.
    font = ImageFont.load_default(size=14)
    entries = []
    for i in sorted(c["selection_indices"].tolist() + [c["ids"].index("cold_folderImage56")]):
        name = c["ids"][i]
        rgb = c["rgb"][i][-1].permute(1, 2, 0)
        h, w = rgb.shape[:2]
        canvas = Image.new("RGB", (5 * w, h + 70), "white"); draw = ImageDraw.Draw(canvas)
        draw.text((3, 2), f"{name} | Episode region removal | red: positive drop; blue: negative; common scale ±{scale}", fill="black", font=font)
        canvas.paste(Image.fromarray(rgb.numpy()), (0, 70)); draw.text((3, 35), "Original final sampled frame", fill="black", font=font)
        for column, (case, fill) in enumerate((("frozen", "mean"), ("full", "mean"), ("frozen", "blur33"), ("full", "blur33")), 1):
            row = torch.load(a.root / "occlusion" / case / f"{name}.pt", map_location="cpu", weights_only=False)
            field = F.interpolate(row["maps"][fill][None, None], size=(h, w), mode="nearest")[0, 0]
            magnitude = (field.abs() / scale).clamp(0, 1)[..., None] * .65
            color = torch.zeros_like(rgb).float(); color[..., 0] = (field >= 0) * 255; color[..., 2] = (field < 0) * 255
            overlay = ((1 - magnitude) * rgb.float() + magnitude * color).round().clamp(0, 255).byte()
            canvas.paste(Image.fromarray(overlay.numpy()), (column * w, 70))
            draw.text((column * w + 3, 35), f"{case} / {fill}; max {float(field.max()):.4g}", fill="black", font=font)
        canvas.save(gallery / f"{name}.png"); entries.append(name)
    (gallery / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Actual episode margin changes</h1>'
        '<p>48 fixed whole-content regions; mean and blur controls. Fixed scale ±0.01; no per-image normalization. '
        'These maps are not spill masks. Regions are constant over the sampled clip.</p>' + ''.join(
            f'<h2>{name}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png" width="100%"></a>' for name in entries))
    report += ["[Gradient-free occlusion gallery](occlusion_gallery/index.html). "
        "The display uses a fixed ±0.01 cosine-margin scale, without per-image rescaling. "
        "Native signed matrices are preserved even when differences are too small to appear strongly colored.", "",
        "## Gradient competition and saturation", ""]
    gradients = json.loads((a.root / "final_gradient_audit.json").read_text())
    report += ["Weighted full-loss derivatives at the pooled student features:", "",
        "| Term | Weighted feature-gradient norm |", "|---|---:|"]
    for name, value in gradients["pooled_feature_gradients"]["weighted_gradient_norms"].items():
        report.append(f"| {name} | {value:.7g} |")
    report += ["", "[Per-term gradient agreement and representative early/middle/final-block contributions](final_gradient_audit.json). "
        "The parameter contributions are explicitly limited to one clean and one cold source's final prefix, "
        "rather than mislabeled as full global parameter gradients.", "",
        "Cosine-margin alignment/tangent norms are stored with each occlusion result. A small tangent norm "
        "can suppress local gradients at a confident decision. Region removal provides an independent "
        "check; neither gradient saturation nor a colorful counterfactual map establishes spill identity.", "",
        "## Implication for the four losses", "",
        "For each independently encoded prefix, zero-sum changes in final patch outputs leave their spatial "
        "mean unchanged. All four current losses operate downstream of those means. Therefore they cannot "
        "distinguish every spatial rearrangement or enforce liquid identity/extent at that output interface. "
        "The backbone can still learn regional features, but unfreezing it supplies no unique region target.", "",
        "Original scene/acquisition correlations, the absence of region identity supervision and decision "
        "explanation limitations must be assessed separately. Classification, perturbation sensitivity "
        "and qualitative spill correspondence are different claims. There are still no real masks, "
        "trusted scene groups or untouched final scene test; no IoU/pointing accuracy is invented.", ""]
    (a.root / "deep_analysis.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
