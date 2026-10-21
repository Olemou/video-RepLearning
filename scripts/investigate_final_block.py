#!/usr/bin/env python3
"""Compare frozen final-block attention and class-conditioned final-block flow.

All 16 heads and all consumed last-tubelet queries are retained. Keys span the
actual past-only prefix. No CLS token, head selection, retraining or ROI guidance.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from vision_model.training.trajectory import TrajectoryModel
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.models.final_block_maps import replay_last_tubelet
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.data.sampling import TemporalSamplingConfig


def correlation(a, b):
    result = float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).statistic)
    return result if np.isfinite(result) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "reports/trajectory_thermal_aug_100")
    p.add_argument("--video-ids", nargs="+", default=["warm_scene_1_hot", "cold_scene3", "clean_videoFrame99"])
    args = p.parse_args()
    torch.set_num_threads(4)
    cache = torch.load(args.run / "causal_features.pt", map_location="cpu", weights_only=False)
    saved = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    trained = TrajectoryModel(saved["width"], saved["hidden"], saved["embedding_dim"])
    trained.load_state_dict(saved["model"])
    torch.manual_seed(239)
    initial = TrajectoryModel(saved["width"], saved["hidden"], saved["embedding_dim"])
    heads = {"trained": trained.eval(), "initial": initial.eval()}
    for model in heads.values():
        model.requires_grad_(False)
    n, v, time, width = cache["features"].shape
    valid = cache["valid"][:, None].expand(n, v, time).reshape(n*v, time)
    times = cache["timestamps"][:, None].expand(n, v, time).reshape(n*v, time)
    embedding_bank = {}
    with torch.no_grad():
        for name, model in heads.items():
            embedding_bank[name] = model(cache["features"].reshape(n*v, time, width), valid, times)["embedding"].reshape(n, v, -1)
    transform = OfficialVJEPA2Transform(cache["model_id"], preserve_field_of_view=True)
    dataset = VideoManifestDataset(cache["manifest"], dataset_root=cache["dataset_root"], transform=transform,
        sampling=TemporalSamplingConfig(clip_length=64, target_fps=4, training=False, short_video_policy="repeat_last"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = load_pretrained_encoder(cache["model_id"], local_files_only=True)
    teacher.freeze_encoder(True)
    encoder = teacher.encoder.to(device=device, dtype=dtype).eval()
    assert not any(p.requires_grad for p in encoder.parameters())
    final_weight = encoder.blocks[-1].attn.qkv.weight
    fingerprint = hashlib.sha256(final_weight.detach().float().cpu().numpy().tobytes()).hexdigest()
    captured = {}
    def hook(module, inputs):
        captured["hidden"] = inputs[0].detach().cpu()
    handle = encoder.blocks[-1].register_forward_pre_hook(hook)
    out = args.run / "final_block"
    out.mkdir(parents=True, exist_ok=True)
    collected = []
    for video_id in args.video_ids:
        i = cache["ids"].index(video_id)
        di = [r.video_id for r in dataset.records].index(video_id)
        item = dataset[di]
        count = int(cache["valid"][i].sum())
        pixels = item["pixel_values_videos"][:2*count]
        hidden_bank, fused_bank, replay_bank = [], [], []
        no_attention_bank, no_block_bank = [], []
        for end in range(1, count + 1):
            clip = pixels[:2*end].permute(1, 0, 2, 3)[None].to(device=device, dtype=dtype)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                encoded = encoder(clip)
                fused_bank.append(encoded[0, -256:].mean(0).float().cpu())
                h = captured.pop("hidden")
                hidden_bank.append(h)
                replay, _ = replay_last_tubelet(encoder, h.to(device), frames=end, height=16, width=16)
                replay_bank.append(replay.mean(1).float().cpu()[0])
                # Measure final attention's necessity separately from its residual
                # bypass. Earlier blocks and input images remain unmodified.
                base = h.to(device)[:, -256:]
                block = encoder.blocks[-1]
                without_attention = base + block.mlp(block.norm2(base))
                no_attention_bank.append(encoder.norm(without_attention).mean(1).float().cpu()[0])
                no_block_bank.append(encoder.norm(base).mean(1).float().cpu()[0])
        fused_features, replay_features = torch.stack(fused_bank), torch.stack(replay_bank)
        cache_error = float((fused_features - cache["features"][i, 0, :count]).abs().max())
        replay_error = float((replay_features - fused_features).abs().max())
        gradients, margins, score_error, branch_ablations = {}, {}, {}, {}
        for name, model in heads.items():
            bank = embedding_bank[name]
            keep = torch.arange(n) != i
            proto = torch.stack([F.normalize(bank[keep & (cache["labels"] == label)].mean((0, 1)), dim=0)
                                  for label in (0, 1)])
            features = cache["features"][i:i+1, 0].clone()
            features[0, :count] = replay_features
            features.requires_grad_(True)
            result = model(features, cache["valid"][i:i+1], cache["timestamps"][i:i+1], detach_teacher=False)
            score = result["embedding"][0] @ (proto[1] - proto[0])
            gradients[name] = torch.autograd.grad(score, features)[0][0, :count]
            with torch.no_grad():
                original = model(cache["features"][i:i+1, 0], cache["valid"][i:i+1], cache["timestamps"][i:i+1])["embedding"][0]
                original_score = original @ (proto[1] - proto[0])
                if name == "trained":
                    for condition, bank_features in (("remove_final_attention_branch", no_attention_bank),
                                                      ("skip_entire_final_block", no_block_bank)):
                        altered = cache["features"][i:i+1, 0].clone()
                        altered[0, :count] = torch.stack(bank_features)
                        z = model(altered, cache["valid"][i:i+1], cache["timestamps"][i:i+1])["embedding"][0]
                        changed_score = float(z @ (proto[1] - proto[0]))
                        branch_ablations[condition] = dict(margin=changed_score,
                            predicted_label=int(changed_score > 0), margin_change=changed_score-float(original_score))
            margins[name] = float(original_score)
            score_error[name] = abs(float(score.detach()) - float(original_score))
        raw = torch.zeros(count, count, 16, 16)
        weighted = {name: torch.zeros_like(raw) for name in heads}
        signed_input = torch.zeros_like(raw)
        final_heads = None
        for t, prehidden in enumerate(hidden_bank):
            hidden = prehidden.to(device).clone().requires_grad_(True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                tokens, attention = replay_last_tubelet(encoder, hidden, frames=t+1, height=16, width=16)
                feature = tokens.mean(1).float()
            raw[t, :t+1] = attention.detach().mean((0, 1, 2)).reshape(t+1, 16, 16).cpu()
            if t == count - 1:
                final_heads = attention.detach()[0].mean(1).reshape(16, count, 16, 16).sum(1).cpu()
            for j, (name, upstream) in enumerate(gradients.items()):
                da, dh = torch.autograd.grad(feature, (attention, hidden),
                    grad_outputs=upstream[t:t+1].to(device), retain_graph=j < len(heads)-1)
                relevance = (da.float() * attention.detach()).clamp_min(0).mean((0, 1)).sum(0)
                weighted[name][t, :t+1] = relevance.reshape(t+1, 16, 16).detach().cpu()
                if name == "trained":
                    signed_input[t, :t+1] = (dh.float() * hidden.detach().float()).sum(-1).reshape(t+1, 16, 16).detach().cpu()
            print(f"{video_id}: block 23 gradient replay {t+1}/{count}", flush=True)
        # Retain key-time identity: a past key may affect several later prefix
        # states, so sum its contributions over the prefixes that consume it.
        maps = dict(raw=raw.sum(0) / count, trained=weighted["trained"].sum(0),
                    initial=weighted["initial"].sum(0), signed_activation=signed_input.sum(0))
        height, source_width = item["video"].shape[-2:]
        scale = min(256 / height, 256 / source_width)
        rh, rw = round(height*scale), round(source_width*scale)
        top, left = (256-rh)//2, (256-rw)//2
        def content_grid(grid):
            square = cv2.resize(np.asarray(grid, dtype=np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
            return square[top:top+rh, left:left+rw]
        floor = torch.zeros(256, 256)
        floor[top+round(.65*rh):top+rh, left:left+rw] = 1
        content = torch.zeros_like(floor)
        content[top:top+rh, left:left+rw] = 1
        floor = F.avg_pool2d(floor[None, None], 16, 16)[0, 0]
        content = F.avg_pool2d(content[None, None], 16, 16)[0, 0]
        summary = dict(video_id=video_id, label=int(cache["labels"][i]), margins=margins,
            original_cache_max_error=cache_error, replay_feature_max_error=replay_error,
            replay_score_absolute_error=score_error,
            final_block_branch_ablations=branch_ablations,
            per_prefix_feature_gradient_norm={name: grad.norm(dim=-1).tolist() for name, grad in gradients.items()},
            raw_attention_frozen=True, prefix_query_selection="all 256 last-tubelet queries; all 16 heads",
            grad_map_spearman_initial_vs_trained=correlation(maps["initial"], maps["trained"]),
            floor_proxy="bottom 35% of source image, not an annotated spill mask")
        for name in ("raw", "trained", "initial"):
            m = maps[name].sum(0)
            summary[name] = dict(total_mass=float(m.sum()),
                padding_mass_fraction=float((m*(1-content)).sum() / m.sum().clamp_min(1e-20)),
                floor_fraction_of_content_mass=float((m*floor).sum() / (m*content).sum().clamp_min(1e-20)))
        if video_id == "warm_scene_1_hot":
            prior_path = args.run / "localization_audit/report.json"
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                input_grid = np.asarray(prior["actual_input_occlusion_grid"])
                summary["correlation_with_actual_input_removal"] = {}
                for name in ("raw", "trained", "initial"):
                    m = content_grid(maps[name].sum(0).numpy())
                    grid = np.asarray([[m[round(r*rh/4):round((r+1)*rh/4),
                                          round(c*rw/4):round((c+1)*rw/4)].sum()
                                       for c in range(4)] for r in range(4)])
                    summary["correlation_with_actual_input_removal"][name] = correlation(grid, input_grid)
        folder = out / video_id
        folder.mkdir(exist_ok=True)
        torch.save(dict(maps=maps, raw_by_prefix_key=raw, weighted_by_prefix_key=weighted,
                        final_head_maps=final_heads, summary=summary), folder / "maps.pt")
        (folder / "report.json").write_text(json.dumps(summary, indent=2))
        collected.append(dict(id=video_id, item=item, maps=maps, heads=final_heads,
                              summary=summary, folder=folder, bounds=(top,left,rh,rw)))
    handle.remove()
    assert fingerprint == hashlib.sha256(final_weight.detach().float().cpu().numpy().tobytes()).hexdigest()
    limits = {}
    for name in ("raw", "trained", "initial"):
        values = np.concatenate([e["maps"][name].numpy().ravel() for e in collected])
        # Same color scale across ALL selected videos and frames, never per frame.
        limits[name] = max(float(np.quantile(values, .995)), 1e-20)
    # Compare trained and initial class-conditioned maps in the same units too.
    limits["trained"] = limits["initial"] = max(limits["trained"], limits["initial"])
    for entry in collected:
        maps, item = entry["maps"], entry["item"]
        top, left, rh, rw = entry["bounds"]
        count = len(maps["raw"])
        raw_frames = item["video"][:2*count].permute(0, 2, 3, 1).numpy()
        height, width = raw_frames.shape[1:3]
        writer = cv2.VideoWriter(str(entry["folder"] / "final_block.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4,
                                (width*4, height))
        panels = []
        for frame_index, rgb in enumerate(raw_frames):
            t = frame_index // 2
            parts = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
            for name in ("raw", "trained", "initial"):
                grid = maps[name][t].numpy() / limits[name]
                square = cv2.resize(grid, (256,256), interpolation=cv2.INTER_NEAREST)
                heat = cv2.resize(square[top:top+rh,left:left+rw], (width,height), interpolation=cv2.INTER_NEAREST)
                color = cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255), cv2.COLORMAP_JET)
                parts.append(color)
            panel = np.concatenate(parts, axis=1)
            for j, title in enumerate(("original", "frozen block 23", "trained score x attn grad", "initial score x attn grad")):
                cv2.putText(panel, title, (j*width+5,20), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            label = "spill" if entry["summary"]["label"] else "clean"
            cv2.putText(panel, f"{entry['id']} | label={label} | margin={entry['summary']['margins']['trained']:+.3f}",
                        (5,height-10),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            writer.write(panel)
            panels.append(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB))
        writer.release()
        indices = np.linspace(0, len(panels)-1, 4).round().astype(int)
        fig, axes = plt.subplots(4,1,figsize=(15,9),constrained_layout=True)
        for ax,index in zip(axes,indices):
            ax.imshow(panels[index]); ax.axis("off")
        fig.suptitle("Final-block investigation: generic attention versus class-score gradient flow; not spill probability")
        fig.savefig(entry["folder"] / "comparison.png",dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(4,4,figsize=(10,9),constrained_layout=True)
        all_heads = entry["heads"].numpy()
        vmax = max(float(np.quantile(all_heads,.995)),1e-20)
        for h, ax in enumerate(axes.flat):
            ax.imshow(all_heads[h],cmap="viridis",vmin=0,vmax=vmax,interpolation="nearest")
            ax.set_title(f"Head {h}",fontsize=9); ax.axis("off")
        fig.suptitle("All final-block heads: last-prefix queries, summed key times; common color scale")
        fig.savefig(entry["folder"] / "all_16_heads.png",dpi=150)
        plt.close(fig)
    report = dict(run=str(args.run), encoder_block_index=23, attention_heads=16,
        final_block_qkv_sha256=fingerprint, final_block_frozen_unchanged=True,
        method="one-block ReLU(attention * class-margin gradient), all heads, query rows consumed by actual pooling",
        initial_control="seed239 initialized trajectory model with its own source-excluded prototypes",
        classifier_feature_bridge="detach_teacher=False for attribution only; default training detach unchanged",
        score="cosine spill prototype minus cosine clean prototype; source excluded from prototypes only",
        color_limits=limits, shared_scale_over_videos_and_frames=True,
        limitations=["These are contextual-key/token flow maps, not pixel masks or probabilities.",
          "No CLS token exists; only last-tubelet query rows are used by the real feature extractor.",
          "This is not full Chefer relevance propagation or multi-layer attention rollout.",
          "Raw attention ignores residual paths; score gradients include the final-block residual and MLP.",
          "Float32 explicit attention differs slightly from fused bf16 inference; fidelity errors are recorded.",
          "Floor measurements use an unannotated geometric proxy. Videos were used in representation training."],
        videos=[e["summary"] for e in collected])
    (out / "report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
