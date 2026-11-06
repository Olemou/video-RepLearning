#!/usr/bin/env python3
"""Measure whether global fine-tuning changed patch relationships, not just means."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F
from attribute_experiment81 import load_case, evaluation_groups
from prepare_experiment81 import normalize_bank, digest
from control_experiment81 import content_cells
from vision_model.training.online_trajectory import autocast_for


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Use allocated GPU")
    out = a.root / "spatial_geometry.json"
    if out.exists():
        raise FileExistsError(out)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    device = torch.device("cuda", 0)
    layer_ids = [0, 5, 11, 17, 23]
    reference_grams, rows = {}, []
    for case in ("frozen", "full"):
        encoder, model, reps, _, settings = load_case(a.root, case, device)
        del model
        captured = {}
        handles = [encoder.blocks[layer].register_forward_hook(
            lambda module, inputs, output, key=layer: captured.__setitem__(key, output)) for layer in layer_ids]
        for ids in evaluation_groups(reps, settings):
            px = torch.stack([normalize_bank(c["pixels"][int(i), :2], c["normalization"]) for i in ids])
            clip = px.flatten(0, 1).permute(0, 2, 1, 3, 4).to(device)
            with autocast_for(device):
                tokens = encoder(clip)
            pooled = tokens[:, -256:].mean(1).float().reshape(len(ids), 2, -1).cpu()
            if float((pooled - reps["features"][ids, :, -1]).abs().max()) > 1e-6:
                raise RuntimeError("Patch geometry forward does not match saved features")
            for j, index in enumerate(ids.tolist()):
                support = content_cells(*c["rgb"][index].shape[-2:]).flatten().to(device)
                for layer in layer_ids + ["final_norm"]:
                    values = tokens if layer == "final_norm" else captured[layer]
                    patches = values[j * 2, -256:][support].float()
                    unit = F.normalize(patches, dim=-1)
                    gram = unit @ unit.T
                    centered = patches - patches.mean(0, keepdim=True)
                    covariance = centered @ centered.T
                    effective_rank = covariance.trace().square() / covariance.square().sum().clamp_min(1e-30)
                    n = len(unit)
                    record = dict(case=case, id=c["ids"][index], label=int(c["labels"][index]), layer=layer,
                        split="train" if index in reps["train_indices"] else "selection",
                        average_offdiagonal_patch_cosine=float((gram.sum() - gram.trace()) / (n * (n - 1))),
                        spatial_variance_energy_fraction=float(centered.square().sum() / patches.square().sum()),
                        spatial_variation_participation_rank=float(effective_rank))
                    key = (index, layer)
                    if case == "frozen":
                        reference_grams[key] = gram.cpu()
                    else:
                        record["frozen_to_full_gram_rms_change"] = float((gram.cpu() - reference_grams[key]).square().mean().sqrt())
                    rows.append(record)
            captured.clear()
        for handle in handles:
            handle.remove()
        del encoder
        torch.cuda.empty_cache()
    out.write_text(json.dumps(dict(scope="All80 sources, identity view, last observed tubelet; all valid content patches",
        selected_layers=layer_ids + ["final_norm"], measurements=rows,
        interpretation="Measures spatial feature geometry and its change; not a liquid-localization metric",
        checkpoint_sha256={case: digest(a.root / case / "checkpoint.pt") for case in ("frozen", "full")},
        manual_spatial_annotations=0, localization_validated=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
