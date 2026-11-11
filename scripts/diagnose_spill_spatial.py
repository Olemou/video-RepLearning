#!/usr/bin/env python3
"""Distributed, causal, patch-preserving diagnostic extraction on a fixed split.

Both ranks own distinct source videos, including every derived view/prefix.
Only rank zero writes consolidated artifacts. No fitted prototype sees a
validation source. Intermediate grids are block outputs, not final-norm tokens.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from vision_model.training.trajectory import TrajectoryModel
from vision_model.interpretability.spatial_layers import capture_spatial_attention
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest, recognition

LAYERS = (5, 11, 17, 23)


def gather_rows(row, context):
    if context.world_size == 1:
        return [row]
    gathered = [None] * context.world_size if context.is_main else None
    dist.gather_object(row, gathered, dst=0)
    return gathered


def heat_panel(frames, maps, path, title, *, signed=False):
    """Shared scale across layers and times, no per-frame renormalization."""
    nrows, ncols = maps.shape[:2]
    maximum = max(float(maps.abs().max()), 1e-30)
    canvas = Image.new("RGB", (256 * ncols, 225 * nrows + 45), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), title, fill="black")
    draw.text((5, 22), f"Shared magnitude scale: {maximum:.5g}; not spill probability", fill="black")
    for row in range(nrows):
        for col in range(ncols):
            grid = maps[row, col].numpy() / maximum
            # Diverging signed scale; sequential positive scale, shared globally.
            if signed:
                rgb = np.stack((np.maximum(grid, 0), 1 - np.abs(grid), np.maximum(-grid, 0)), -1)
            else:
                rgb = np.stack((grid, np.zeros_like(grid), 1 - grid), -1)
            hm = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((256, 256))
            source = Image.fromarray(frames[col]).convert("RGB")
            h, w = frames[col].shape[:2]
            scale = min(256 / h, 256 / w)
            rh, rw = round(h * scale), round(w * scale)
            top, left = (256 - rh) // 2, (256 - rw) // 2
            hm = hm.crop((left, top, left + rw, top + rh)).resize((256, 192))
            source = source.resize((256, 192))
            canvas.paste(Image.blend(source, hm, .5), (col * 256, row * 225 + 45))
            draw.text((col * 256 + 3, row * 225 + 240), f"block {LAYERS[row]} | tubelet {col}", fill="black")
    canvas.save(path)


def layer_attribution(encoder, model, pixels, features, times, proto, context):
    """Replay suffixes for the trained spill-vs-clean cosine margin.

    Captures all attention heads and all queries. Gradients of the same final
    scalar are propagated through later blocks, rather than applying the
    trajectory head out of distribution to intermediate features.
    """
    device = context.device
    prefix_inputs, encoded = [], []
    handle = encoder.blocks[LAYERS[0]].register_forward_pre_hook(
        lambda _m, inputs: prefix_inputs.append(inputs[0].detach().clone()))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for end in range(1, len(features) + 1):
            out = encoder(pixels[:2 * end].permute(1, 0, 2, 3)[None].to(device=device, dtype=torch.bfloat16))
            encoded.append(out[0, -256:].mean(0).float())
    handle.remove()
    encoded = torch.stack(encoded)
    replayed = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16), capture_spatial_attention(encoder, LAYERS):
        for end, hidden in enumerate(prefix_inputs, 1):
            for block in encoder.blocks[LAYERS[0]:]:
                hidden = block(hidden, frames=end, height=16, width=16)
            replayed.append(encoder.norm(hidden)[0, -256:].mean(0).float())
    replayed = torch.stack(replayed)
    source = replayed.detach().clone().requires_grad_(True)
    mask = torch.ones(1, len(source), device=device, dtype=torch.bool)
    embedding = model(source[None], mask, times.to(device)[None], detach_teacher=False)["embedding"][0]
    score = embedding @ (proto[1] - proto[0]).to(device)
    upstream = torch.autograd.grad(score, source)[0]
    positive = torch.zeros(4, len(features), 16, 16)
    signed = torch.zeros_like(positive)
    product = torch.zeros_like(positive)
    raw = torch.zeros_like(positive)
    patch_to_patch = []
    for end, original in enumerate(prefix_inputs, 1):
        hidden = original.detach().clone().requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16), capture_spatial_attention(encoder, LAYERS) as attentions:
            for block in encoder.blocks[LAYERS[0]:]:
                hidden = block(hidden, frames=end, height=16, width=16)
            feature = encoder.norm(hidden)[0, -256:].mean(0).float()
            grads = torch.autograd.grad(feature, [attentions[i] for i in LAYERS], grad_outputs=upstream[end - 1])
            for slot, (layer, grad) in enumerate(zip(LAYERS, grads)):
                att = attentions[layer].detach()[0]
                grad = grad.detach()[0]
                positive[slot, :end] += grad.clamp_min(0).mean((0, 1)).reshape(end, 16, 16).float().cpu()
                signed[slot, :end] += grad.mean((0, 1)).reshape(end, 16, 16).float().cpu()
                product[slot, :end] += (att * grad).clamp_min(0).mean((0, 1)).reshape(end, 16, 16).float().cpu()
                raw[slot, :end] += att[:, -256:].mean((0, 1)).reshape(end, 16, 16).float().cpu()
                if end == len(features):
                    patch_to_patch.append(att[:, -256:].reshape(att.shape[0], 256, end, 256).sum(2).mean(0).cpu())
    with torch.no_grad():
        replay_score = float(score)
        fused_score = float(model(encoded[None], mask, times.to(device)[None])["embedding"][0] @ (proto[1] - proto[0]).to(device))
    error = float((replayed - encoded).abs().max())
    if error > .15 or abs(replay_score - fused_score) > .02:
        raise RuntimeError(f"Layer replay mismatch: feature {error}, scalar {abs(replay_score - fused_score)}")
    return dict(positive=positive, signed=signed, attention_gradient=product, raw=raw,
                patch_to_patch=torch.stack(patch_to_patch), margin=replay_score,
                replay_feature_max_error=error, replay_margin_error=abs(replay_score - fused_score),
                cache_feature_max_error=float((encoded.cpu() - features).abs().max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder-dir", type=Path, default=Path("/home/olemou/hf"))
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(239)
    ctx = initialize_distributed()
    if ctx.device.type != "cuda":
        raise RuntimeError("Use the allocated GPU node for bounded real-encoder diagnostics")
    if ctx.is_main:
        a.output.mkdir(parents=True, exist_ok=False)
    if ctx.world_size > 1:
        dist.barrier()
    c = torch.load(a.cache, map_location="cpu", weights_only=False)
    ck = torch.load(a.experiment / "training/trial_01_checkpoint.pt", map_location="cpu", weights_only=False)
    if digest(a.encoder_dir / "model.safetensors") != c["checkpoint_sha256"]:
        raise ValueError("Encoder hash mismatch")
    ids, y = c["ids"], c["labels"]
    assigned = list(range(ctx.rank, len(ids), ctx.world_size))
    _, counts = audit_distributed_indices(assigned, len(ids))
    if len(set(counts)) != 1:
        raise ValueError("Diagnostic gather currently requires equal unpadded rank lengths")
    manifest = {r["video_id"]: r for r in (json.loads(line) for line in Path(c["manifest"]).read_text().splitlines())}
    transform = OfficialVJEPA2Transform(str(a.encoder_dir), preserve_field_of_view=True)
    loaded = load_pretrained_encoder(str(a.encoder_dir), local_files_only=True)
    loaded.freeze_encoder(True)
    encoder = loaded.encoder.to(device=ctx.device, dtype=torch.bfloat16).eval()
    model = TrajectoryModel(ck["width"], ck["hidden"], ck["embedding_dim"]).to(ctx.device).eval()
    model.load_state_dict(ck["model"], strict=True)
    with torch.no_grad():
        z = model(c["features"][:, 0].to(ctx.device), c["valid"].to(ctx.device), c["timestamps"].to(ctx.device))["embedding"].cpu()
    train = torch.tensor([ids.index(v) for v in ck["train_source_ids"]])
    val = torch.tensor([ids.index(v) for v in ck["validation_source_ids"]])
    proto = torch.stack([F.normalize(z[train][y[train] == label].mean(0), dim=0) for label in (0, 1)])
    selected = {"cold_folderImage13", "cold_folderImage10", "clean_videoFrame121", "clean_videoFrame124"}
    captured, handles = {}, []
    for layer in LAYERS:
        def hook(_m, _inputs, output, layer=layer):
            captured[layer] = output[:, -256:].detach().clone()
        handles.append(encoder.blocks[layer].register_forward_hook(hook))
    all_rows = []
    for local_i, index in enumerate(assigned):
        video_id = ids[index]
        record = manifest[video_id]
        ends = c["frame_indices"][index][c["valid"][index]].tolist()
        # Sampled frame starts are reconstructed with the authoritative sampler.
        from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig
        sampled, sample_valid = TemporalSampler(TemporalSamplingConfig(**c["sampling"])).sample_with_validity(
            num_frames=record["num_frames"], source_fps=record["fps"])
        n = len(ends)
        indices = sampled[:2 * n].tolist()
        if indices[1::2] != ends:
            raise ValueError("Frame/timestamp alignment mismatch")
        path = Path(c["dataset_root"]) / record["video_path"]
        if digest(path) != c["video_sha256"][index]:
            raise ValueError("Source video changed")
        frames_by_index = decode_selected_frames(path, indices, frame_count=record["num_frames"], width=record["width"], height=record["height"])
        frames = torch.stack([torch.from_numpy(frames_by_index[i]).permute(2, 0, 1) for i in indices])
        bank = torch.zeros(2, c["features"].shape[2], 4, 256, 1024, dtype=torch.float16)
        final_raw = torch.zeros(2, c["features"].shape[2], 256, 1024, dtype=torch.float16)
        cache_error = 0.
        for view in range(2):
            pixels = transform(frames if view == 0 else frames.flip(-1))
            with torch.no_grad(), torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16):
                for end in range(1, n + 1):
                    tokens = encoder(pixels[:2 * end].permute(1, 0, 2, 3)[None].to(device=ctx.device, dtype=torch.bfloat16))
                    for slot, layer in enumerate(LAYERS):
                        value = encoder.norm(captured[layer]) if layer == 23 else captured[layer]
                        bank[view, end - 1, slot] = value[0].cpu().half()
                    final_raw[view, end - 1] = captured[23][0].cpu().half()
                    cache_error = max(cache_error, float((tokens[0, -256:].mean(0).float().cpu() - c["features"][index, view, end - 1]).abs().max()))
        if cache_error > .08:
            raise RuntimeError(f"Extraction does not reproduce the teacher cache for {video_id}: {cache_error:g}")
        # Small pixel-only controls: no encoder or representation fitting.
        gray = frames.float().mean(1) / 255
        pixel_stats = torch.tensor([float(gray.mean()), float(gray.std()), float(gray.quantile(.1)), float(gray.quantile(.9))])
        thumbnail = Image.fromarray(frames[len(frames)//2].permute(1, 2, 0).numpy()).resize((192, 144))
        # Hooks are harmless during suffix replay; only frozen copies are saved.
        attribution = None
        if video_id in selected:
            pixels = transform(frames)
            attribution = layer_attribution(encoder, model, pixels, c["features"][index, 0, :n], c["timestamps"][index, :n], proto, ctx)
            # Equal-area, attribution-selected input removal controls. Affected
            # pixels are replaced in normalized input, then EVERY prefix is encoded.
            spatial_score = attribution["attention_gradient"][-1].sum(0)
            scale = min(256 / record["height"], 256 / record["width"])
            rh, rw = round(record["height"] * scale), round(record["width"] * scale)
            top, left = (256 - rh) // 2, (256 - rw) // 2
            content = torch.zeros(256, 256)
            content[top:top + rh, left:left + rw] = 1
            candidates = (F.avg_pool2d(content[None, None], 16, 16)[0, 0].flatten() == 1).nonzero().flatten()
            rank = candidates[spatial_score.flatten()[candidates].argsort()]
            chosen = {"low_content26": rank[:26], "high_content26": rank[-26:],
                      "random_content26": candidates[torch.randperm(len(candidates), generator=torch.Generator().manual_seed(239))[:26]]}
            conditions = {}
            for name, cells in chosen.items():
                mask = torch.zeros(256, dtype=torch.bool)
                mask[cells] = True
                mask = mask.reshape(16, 16).repeat_interleave(16, 0).repeat_interleave(16, 1)
                changed = pixels.masked_fill(mask[None, None], 0)
                feats = []
                with torch.no_grad(), torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16):
                    for end in range(1, n + 1):
                        tokens = encoder(changed[:2 * end].permute(1, 0, 2, 3)[None].to(device=ctx.device, dtype=torch.bfloat16))
                        feats.append(tokens[0, -256:].mean(0).float())
                with torch.no_grad():
                    emb = model(torch.stack(feats)[None], torch.ones(1, n, dtype=torch.bool, device=ctx.device), c["timestamps"][index:index + 1, :n].to(ctx.device))["embedding"][0]
                    conditions[name] = float(emb.cpu() @ (proto[1] - proto[0]))
            attribution["input_removal_margins"] = conditions
            attribution["frames"] = [frames_by_index[i] for i in ends]
        row = dict(index=index, tokens=bank, final_raw=final_raw, pixel_stats=pixel_stats,
                   thumbnail=np.asarray(thumbnail), attribution=attribution, cache_error=cache_error)
        gathered = gather_rows(row, ctx)
        if ctx.is_main:
            all_rows.extend(gathered)
            print(json.dumps(dict(encoded_sources=len(all_rows), total=len(ids), max_cache_error=max(r["cache_error"] for r in gathered))), flush=True)
    for handle in handles:
        handle.remove()
    if ctx.is_main:
        all_rows.sort(key=lambda r:r["index"])
        payload = dict(tokens=torch.stack([r["tokens"] for r in all_rows]),
            final_raw=torch.stack([r["final_raw"] for r in all_rows]),
            norm_state={k:v.cpu().float() for k,v in encoder.norm.state_dict().items()},
            layers=list(LAYERS), ids=ids, labels=y, valid=c["valid"], timestamps=c["timestamps"],
            train_indices=train, validation_indices=val, checkpoint_sha256=digest(a.experiment / "training/trial_01_checkpoint.pt"),
            encoder_sha256=c["checkpoint_sha256"], cache_sha256=digest(a.cache), world_size=ctx.world_size,
            source_counts_per_rank=counts, views=["identity", "mirror"],
            temporal_context_policy=c["temporal_context_policy"], normalization=c["normalization"],
            cache_feature_max_error=max(r["cache_error"] for r in all_rows),
            patch_supervision="none", intermediate_features="post-block unnormalized; final block includes encoder.norm")
        torch.save(payload, a.output / "spatial_tokens.pt")
        contact = Image.new("RGB", (192 * 8, 174 * 10), "white")
        draw = ImageDraw.Draw(contact)
        for i, row in enumerate(all_rows):
            x, yy = (i % 8) * 192, (i // 8) * 174
            contact.paste(Image.fromarray(row["thumbnail"]), (x, yy))
            draw.text((x + 2, yy + 144), ids[i], fill="black")
            draw.text((x + 2, yy + 158), "TRAIN" if i in train else "VALIDATION", fill="blue" if i in train else "red")
        contact.save(a.output / "source_contact_sheet.png")
        stats = torch.stack([r["pixel_stats"] for r in all_rows])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., random_state=239))
        classifier.fit(stats[train].numpy(), y[train].numpy())
        scores = classifier.predict_proba(stats.numpy())[:, 1]
        pixel_report = dict(mean_std_quantile_logistic_validation_accuracy=float((classifier.predict(stats[val].numpy()) == y[val].numpy()).mean()),
                            descriptor_names=["mean", "std", "p10", "p90"], ids=ids, stats=stats.tolist(), probabilities=scores.tolist(),
                            caveat="Pixel moments contain spill and background; success alone does not identify a background shortcut.")
        for layer_slot, layer in enumerate(LAYERS):
            pooled = payload["tokens"][:, 0, :, layer_slot].float().mean(2).mean(1)
            pixel_report[f"block{layer}_frozen_mean_prototype"] = recognition(pooled, y, train, val)
        for row in all_rows:
            attr = row["attribution"]
            if attr is None:
                continue
            video_id = ids[row["index"]]
            torch.save({k:v for k,v in attr.items() if k!="frames"}, a.output / f"{video_id}_layer_maps.pt")
            for name in ("positive", "signed", "attention_gradient", "raw"):
                heat_panel(attr["frames"], attr[name], a.output / f"{video_id}_{name}.png", f"{video_id}: {name}; same trained cosine-margin target", signed=name=="signed")
            # Patch-to-patch attention has one all-head mean matrix per block.
            grid = attr["patch_to_patch"].numpy()
            image = Image.new("RGB", (256 * 4, 280), "white")
            maximum = float(grid.max())
            for slot, layer in enumerate(LAYERS):
                gray = (grid[slot] / maximum * 255).astype(np.uint8)
                image.paste(Image.fromarray(gray).convert("RGB"), (slot * 256, 0))
                ImageDraw.Draw(image).text((slot * 256 + 3, 259), f"block {layer}; all heads", fill="black")
            image.save(a.output / f"{video_id}_patch_to_patch.png")
            pixel_report[video_id] = {k:v for k,v in attr.items() if not isinstance(v, (torch.Tensor, list))}
        (a.output / "spatial_diagnostics.json").write_text(json.dumps(pixel_report, indent=2) + "\n")
        print(json.dumps(dict(output=str(a.output), world_size=ctx.world_size, rank_source_counts=counts,
                              spatial_tokens_shape=list(payload["tokens"].shape))), flush=True)
    if ctx.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
