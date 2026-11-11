#!/usr/bin/env python3
"""All-block token/attention capture plus independent production input gradients."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from attribute_experiment81 import load_case, evaluation_groups, margins_for_group
from control_experiment81 import content_cells, transform_bank
from prepare_experiment81 import normalize_bank, digest
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.online_trajectory import OnlinePrefixEncoder, autocast_for, explicit_prefix_forward
from vision_model.interpretability.prefix_input_gradients import prefix_input_gradient, trapezoid_attributions
from vision_model.interpretability.map_geometry import (
    inverse_letterbox_map, nonnegative_evidence_colors, blend_rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["frozen", "full"])
    a = p.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This bounded diagnostic requires the single allocated GPU; not a DDP training launch")
    a.output.mkdir(exist_ok=False)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    device = torch.device("cuda", 0)
    clean = next(i for i in c["selection_indices"].tolist() if c["labels"][i] == 0)
    selected = [c["ids"].index(n) for n in ("cold_folderImage56", "cold_folderImage10", "cold_folderImage13")] + [clean]
    summaries = []
    font = ImageFont.load_default(size=14)
    for case in a.cases:
        encoder, model, reps, proto, settings = load_case(a.root, case, device)
        wrapper = OnlinePrefixEncoder(encoder)
        transform = OfficialVJEPA2Transform(settings["teacher_metadata"]["encoder_dir"], preserve_field_of_view=True)
        groups = evaluation_groups(reps, settings)
        for i in selected:
            ids = next(g for g in groups if i in g)
            local = int((ids == i).nonzero()[0])
            px = torch.stack([normalize_bank(c["pixels"][int(j), :2], c["normalization"]) for j in ids]).float()
            score = lambda feature: margins_for_group(model, reps, ids, feature, proto, device)[local]
            endpoint = prefix_input_gradient(wrapper, px, c["valid"][ids], score, device)
            error = float((endpoint["features"] - reps["features"][ids]).abs().max())
            if error > 1e-6:
                raise RuntimeError(f"Original input features failed replay: {error}")
            unrelated = endpoint["gradient"].clone(); unrelated[local, 0] = 0
            if unrelated.abs().max() > 1e-12:
                raise RuntimeError("Identity decision unexpectedly depends on partner/mirror")
            captured = {}
            hooks = [block.register_forward_hook(lambda module, args, output, key=l:
                captured.__setitem__(key, output[local*2, -256:].detach().cpu().half()))
                for l, block in enumerate(encoder.blocks)]
            clip = px.flatten(0, 1).permute(0, 2, 1, 3, 4).to(device)
            with torch.no_grad(), autocast_for(device):
                tokens = encoder(clip)
            for hook in hooks:
                hook.remove()
            pooled = tokens[:, -256:].mean(1).float().reshape(len(ids), 2, -1).cpu()
            if float((pooled - reps["features"][ids, :, -1]).abs().max()) > 1e-6:
                raise RuntimeError("Block capture changed production output")
            with torch.no_grad(), autocast_for(device):
                explicit, attention, _ = explicit_prefix_forward(encoder, clip)
            raw_keys = torch.stack([att[local*2, :, -256:].mean((0, 1)).reshape(6, 16, 16).cpu() for att in attention])
            patch_attention = torch.stack([att[local*2, :, -256:, -256:].mean(0).cpu().half() for att in attention])
            patches = torch.stack([captured[l] for l in range(24)])
            support = content_cells(*c["rgb"][i].shape[-2:]).flatten()
            unit = F.normalize(patches[:, support].float(), dim=-1)
            similarities = unit @ unit.transpose(-1, -2)
            payload = dict(id=c["ids"][i], case=case, block_tokens=patches,
                patch_cosine_similarity=similarities.half(), valid_patch_positions=support.nonzero().flatten(),
                raw_last_query_all_key_attention=raw_keys, last_tubelet_patch_attention=patch_attention,
                input_gradient=endpoint["gradient"][local, 0], input_gradient_score=endpoint["score"],
                input_gradient_replay_error=endpoint["replay_max_error"], production_saved_feature_error=error,
                explicit_feature_error=float((explicit.cpu().reshape_as(pooled) - pooled).abs().max()),
                checkpoint_sha256=digest(a.root / case / "checkpoint.pt"),
                scope="All24 blocks; tokens/raw attention at 12-frame prefix; gradients through all six real prefixes",
                probabilities_are_class_specific=False, localization_validated=False)
            del attention, tokens, clip, explicit, captured, endpoint
            if case == "full" and c["ids"][i] == "cold_folderImage56":
                rgb = c["rgb"][i]
                fills = dict(mean=rgb.float().mean((-2, -1), keepdim=True).expand_as(rgb).round().byte(),
                    blur33=F.avg_pool2d(F.pad(rgb.float(), (16,16,16,16), mode="reflect"), 33, 1).round().clamp(0,255).byte())
                ig_records = {}
                for fill_name, fill in fills.items():
                    baseline = px.clone()
                    baseline[local] = transform_bank(fill, [], transform, include_augmentations=False).float()
                    nodes = {1.: (payload["input_gradient_score"], payload["input_gradient"])}
                    for steps in (32, 64):
                        for j in range(steps + 1):
                            alpha = j / steps
                            if alpha not in nodes:
                                path = torch.lerp(baseline, px, alpha)
                                r = prefix_input_gradient(wrapper, path, c["valid"][ids], score, device)
                                nodes[alpha] = (r["score"], r["gradient"][local, 0])
                        ig = trapezoid_attributions(baseline[local, 0], px[local, 0],
                            [nodes[j/steps][1] for j in range(steps + 1)])
                        delta = nodes[1.][0] - nodes[0.][0]
                        residual = abs(float(ig.sum()) - delta)
                        threshold = max(.005, .05 * abs(delta))
                        summary = dict(fill=fill_name, steps=steps, endpoint_score=nodes[1.][0], baseline_score=nodes[0.][0],
                            score_delta=delta, attribution_sum=float(ig.sum()), completeness_abs_error=residual,
                            completeness_tolerance=threshold, completeness_passed=residual <= threshold)
                        print(json.dumps(dict(id=c["ids"][i], case=case, integrated_gradients=summary)), flush=True)
                        if residual <= threshold:
                            break
                    ig_records[fill_name] = dict(attribution=ig, summary=summary,
                        caveat="Reference is not certified clean; mixed-precision casts can limit completeness")
                payload["integrated_gradients"] = ig_records
            torch.save(payload, a.output / f"{c['ids'][i]}_{case}.pt")
            # Fixed probability scale, shared by all blocks, arms, cases. No gradient interpretation.
            rgb = c["rgb"][i][-1].permute(1,2,0)
            canvas = Image.new("RGB", (1280, 78 + 6 * 270), "white"); draw = ImageDraw.Draw(canvas)
            draw.text((4,3), f"{c['ids'][i]} | {case} | ALL24 blocks RAW attention (NOT class-specific)", fill="black", font=font)
            draw.text((4,26), "All heads; uniform last-tubelet queries -> last-tubelet keys; shared probability scale 0.02", fill="black", font=font)
            draw.text((4,49), "Block tokens and patch-to-patch attention matrices are retained in the .pt file.", fill="black", font=font)
            for layer in range(24):
                value = inverse_letterbox_map(raw_keys[layer,-1], *rgb.shape[:2], interpolation="nearest")
                overlay = blend_rgb(rgb, nonnegative_evidence_colors(value, scale=.02), alpha=.4)
                x,y = (layer%4)*320, 78+(layer//4)*270
                draw.text((x+3,y), f"block {layer:02d} / layer {layer+1}", fill="black", font=font)
                canvas.paste(Image.fromarray(overlay.numpy()).resize((320,240)), (x,y+25))
            canvas.save(a.output / f"{c['ids'][i]}_{case}_raw24.png")
            # Production input-gradient magnitude, not attention or a segmentation map.
            grad = payload["input_gradient"].abs().mean(1)
            content = content_cells(*rgb.shape[:2])
            pooled_grad = F.avg_pool2d(grad[:,None],16,16)[:,0]
            peak = pooled_grad[-1].masked_fill(~content, -1).argmax().item()
            summary = dict(id=c["ids"][i], case=case, production_feature_error=error,
                input_gradient_replay_error=payload["input_gradient_replay_error"], input_gradient_peak=[peak//16,peak%16],
                input_gradient_max=float(pooled_grad[-1][content].max()), input_gradient_median=float(pooled_grad[-1][content].median()))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            del payload, px, patches, similarities, unit
            torch.cuda.empty_cache()
        del encoder, wrapper, model
        torch.cuda.empty_cache()
    (a.output / "completed.json").write_text(json.dumps(dict(sources=selected, results=summaries,
        host=os.uname().nodename, gpu=torch.cuda.get_device_name(), allocation=settings["job_id"],
        raw_attention_probability_scale=.02, blocks=24, training_updates_added=0,
        manual_annotations=0, localization_validated=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
