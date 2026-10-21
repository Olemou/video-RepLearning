#!/usr/bin/env python3
"""t-SNE and per-video inference with padding-aware transformer attention maps."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence_study import CachedHeads
from simulate_loss_training import gaussian_output
from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA, validate_feature_cache
from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.config import ThermalAugConfig
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.models.attention_maps import received_attention


@torch.no_grad()
def embeddings(heads, features, valid):
    return F.normalize(gaussian_output(heads, features, valid).video_mean, dim=-1)


def separation(z, labels):
    from sklearn.metrics import silhouette_score
    similarity = z @ z.T
    diagonal = np.eye(len(z), dtype=bool)
    positives = (labels[:, None] == labels[None, :]) & ~diagonal
    negatives = labels[:, None] != labels[None, :]
    nearest = np.where(diagonal, -np.inf, similarity).argmax(1)
    return dict(mean_positive_cosine=float(similarity[positives].mean()),
                mean_negative_cosine=float(similarity[negatives].mean()),
                cosine_silhouette=float(silhouette_score(z, labels, metric="cosine")),
                training_neighbor_agreement_excluding_self=float((labels[nearest] == labels).mean()))


def plot_tsne(cache, heads, output, seed):
    from sklearn.manifold import TSNE
    torch.manual_seed(seed)
    initial = CachedHeads(cache["features"].shape[-1], cache["mean_head"]["3.bias"].numel(), cache["variance_floor"])
    if seed == cache["seed"]:
        initial.mean_head.load_state_dict(cache["mean_head"])
        initial.variance_head.load_state_dict(cache["variance_head"])
    arrays = [embeddings(initial, cache["features"], cache["valid"]).numpy(),
              embeddings(heads.cpu(), cache["features"], cache["valid"]).numpy()]
    labels = cache["labels"].numpy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    points = []
    for row, (name, z) in enumerate(zip(("Initial heads", "Trained heads"), arrays)):
        for col, perplexity in enumerate((5, 10, 15)):
            if perplexity >= len(z):
                raise ValueError("This comparison needs at least 16 reference videos")
            xy = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="random",
                      learning_rate="auto", max_iter=1500, random_state=239).fit_transform(z)
            ax = axes[row, col]
            for category, marker in (("clean", "o"), ("cold", "^"), ("warm", "s")):
                idx = np.array(cache["categories"]) == category
                ax.scatter(*xy[idx].T, c="#2185c5" if category == "clean" else "#d65b36",
                           marker=marker, s=65, label=f"{category} (class {LABEL_MAP[category]})")
            for i, (x, y) in enumerate(xy):
                ax.annotate(str(i + 1), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
                points.append(dict(stage=name, perplexity=perplexity, number=i + 1,
                                   video_id=cache["ids"][i], label=int(labels[i]), category=cache["categories"][i],
                                   x=float(x), y=float(y)))
            ax.set_title(f"{name}; perplexity={perplexity}")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("24 training videos | t-SNE of normalized Gaussian mean embeddings\nSeparate projections: axis scales and inter-cluster distances are not comparable")
    fig.savefig(output / "tsne_classes.png", dpi=180)
    plt.close(fig)
    with (output / "tsne_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=points[0].keys()); writer.writeheader(); writer.writerows(points)
    metrics = {name: separation(z, labels) for name, z in zip(("initial", "trained"), arrays)}
    metrics["note"] = "Original embedding-space statistics on training videos; neither t-SNE nor neighbor agreement is held-out accuracy. Initial heads reproduce the checkpoint seed (cached initialization for the cache seed)."
    (output / "separation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output / "embeddings.npz", initial=arrays[0], trained=arrays[1], labels=labels,
                        ids=np.array(cache["ids"]))
    print(json.dumps(metrics, indent=2), flush=True)


def external_item(path, sampler, transform):
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(path), dimension_order="NCHW", device="cpu", num_ffmpeg_threads=1)
    fps = float(decoder.metadata.average_fps)
    indices, valid = sampler.sample_with_validity(num_frames=len(decoder), source_fps=fps, sample_index=0)
    frames = decoder.get_frames_at(indices=indices.tolist()).data
    return dict(video_id=path.stem, video_path=str(path.resolve()), label=None, category=None,
                pixel_values_videos=transform(frames), frame_indices=torch.from_numpy(indices.copy()),
                temporal_valid_mask=torch.from_numpy(valid.copy()), source_fps=fps)


def export_maps(item, maps, valid, normalization, folder, prediction):
    folder.mkdir(parents=True, exist_ok=True)
    pixels = item["pixel_values_videos"].float()
    mean = torch.tensor(normalization["mean"])[None, :, None, None]
    std = torch.tensor(normalization["std"])[None, :, None, None]
    rgb = (pixels * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    frame_valid = valid.repeat_interleave(2).numpy()
    selected = np.flatnonzero(frame_valid)
    # Relative to uniform incoming attention; fixed scale across all frames of this clip.
    relative = maps * int(valid.sum()) * maps.shape[-1] * maps.shape[-2]
    heat = F.interpolate(relative[:, None], size=rgb.shape[1:3], mode="bilinear", align_corners=False)[:, 0].repeat_interleave(2, 0).numpy()
    vmax = max(float(np.percentile(heat[selected], 99)), 1e-8)
    color = plt.get_cmap("inferno")(np.clip(heat / vmax, 0, 1))[..., :3]
    overlay = (.55 * rgb + .45 * color).clip(0, 1)
    sampled = selected[np.linspace(0, len(selected) - 1, min(6, len(selected))).astype(int)]
    fig, axes = plt.subplots(2, len(sampled), figsize=(3 * len(sampled), 6), squeeze=False, constrained_layout=True)
    for column, i in enumerate(sampled):
        axes[0, column].imshow(rgb[i]); axes[1, column].imshow(overlay[i])
        axes[0, column].set_title(f"source frame {int(item['frame_indices'][i])}\n{float(item['frame_indices'][i]) / item['source_fps']:.2f}s")
        for ax in axes[:, column]: ax.axis("off")
    fig.suptitle(f"{item['video_id']} | prototype prediction: {prediction}\nLast-layer incoming attention (not class attribution); shared clip color scale 0–{vmax:.2f} × uniform")
    fig.savefig(folder / "attention.png", dpi=120); plt.close(fig)
    # Sampled model-view clip: not a full-resolution explanation of unseen frames.
    pairs = np.concatenate((rgb[selected], overlay[selected]), axis=2)
    process = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{pairs.shape[2]}x{pairs.shape[1]}", "-r", "4", "-i", "pipe:0",
                              "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(folder / "attention.mp4")],
                             input=(pairs * 255).round().astype(np.uint8).tobytes(), capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode())
    np.savez_compressed(folder / "attention.npz", attention=maps.numpy(), tubelet_valid=valid.numpy(),
                        sampled_frame_indices=item["frame_indices"].numpy(), displayed_sample_positions=selected)
    return dict(overlay_color_max_relative_to_uniform=vmax, attention_mass=float(maps.sum()),
                displayed_frames=len(selected), sampling_fps=4)


def infer_attention(args, cache, heads):
    device = torch.device(args.device)
    z = embeddings(heads.cpu(), cache["features"], cache["valid"])
    prototypes = F.normalize(torch.stack([z[cache["labels"] == c].mean(0) for c in (0, 1)]), dim=-1).to(device)
    heads.to(device).eval()
    thermal = ThermalAugConfig(mean=tuple(cache["normalization"]["mean"]), std=tuple(cache["normalization"]["std"]))
    local_cache = ROOT / ".cache/huggingface/hub" / ("models--" + cache["model_id"].replace("/", "--"))
    reference = local_cache / "refs/main"
    local_snapshot = local_cache / "snapshots" / reference.read_text().strip() if reference.exists() else None
    model_source = str(args.model_source or (local_snapshot if local_snapshot and local_snapshot.exists() else cache["model_id"]))
    transform = OfficialVJEPA2Transform(model_source, thermal_config=thermal)
    sampling = TemporalSamplingConfig(**cache["sampling"])
    if sampling.training or sampling.clip_length != 64 or sampling.target_fps != 4:
        raise ValueError("Attention export currently expects deterministic 64-frame / 4 FPS sampling")
    dataset = VideoManifestDataset(args.manifest, sampling=sampling, transform=transform)
    model = load_pretrained_encoder(model_source, local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = model.encoder.to(device=device, dtype=dtype).eval()
    capture = {}
    def hook(module, inputs, kwargs):
        capture["map"] = received_attention(module, inputs[0], **kwargs).cpu()
    handle = encoder.blocks[-1].attn.register_forward_pre_hook(hook, with_kwargs=True)
    count = len(args.video) if args.video else len(dataset)
    if args.limit is not None: count = min(count, args.limit)
    results = []
    try:
        for index in range(count):
            item = (external_item(args.video[index], TemporalSampler(sampling), transform) if args.video else dataset[index])
            pixels = item["pixel_values_videos"].unsqueeze(0).to(device)
            valid = item["temporal_valid_mask"].reshape(1, 32, 2).all(-1).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                tokens = encoder(pixels.permute(0, 2, 1, 3, 4).contiguous(), temporal_valid_mask=valid)
                features = tokens.reshape(1, 32, 256, 1024).mean(2).float()
            with torch.no_grad():
                out = gaussian_output(heads, features, valid)
                embedding = F.normalize(out.video_mean, dim=-1)
                scores = (embedding @ prototypes.T)[0].cpu()
            predicted = int(scores.argmax())
            folder = args.output / "videos" / f"video_{index + 1:03d}"
            details = export_maps(item, capture["map"], valid[0].cpu(), cache["normalization"], folder,
                                  "clean (0)" if predicted == 0 else "warm/cold (1)")
            result = dict(video_id=item["video_id"], video_path=item["video_path"], true_label=item["label"],
                          predicted_class=predicted, cosine_to_prototypes=scores.tolist(),
                          mean_video_variance=float(out.video_variance.mean()),
                          attention_layer=23, directory=str(folder.relative_to(args.output)), **details)
            if not args.video:
                result["cached_embedding_max_abs_difference"] = float((embedding.cpu()[0] - z[index]).abs().max())
            torch.save(dict(embedding=embedding.cpu(), video_variance=out.video_variance.cpu()), folder / "embedding.pt")
            (folder / "inference.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            (args.output / "inference_results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"inferred {index + 1}/{count}: {item['video_id']} -> class {predicted}", flush=True)
    finally:
        handle.remove()
    entries = ["<!doctype html><meta charset='utf-8'><title>Video attention</title>",
               "<h1>Per-video inference and attention</h1><p>Last-layer attention averaged over heads and valid queries. "
               "Class-independent; not proof of causal importance. Predictions use cosine class prototypes from training videos; scores are not probabilities. "
               "Maps cover sampled, cropped model inputs. Each tubelet map covers two sampled frames.</p>"]
    for r in results:
        directory = r["directory"]
        entries.append(f"<h2>{html.escape(r['video_id'])} — predicted class {r['predicted_class']}</h2>"
                       f"<img style='max-width:100%' src='{directory}/attention.png'>"
                       f"<p><a href='{directory}/attention.mp4'>Overlay video</a> · "
                       f"<a href='{directory}/inference.json'>Inference details</a></p>")
    (args.output / "index.html").write_text("\n".join(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "tsne", "attention"), default="all")
    parser.add_argument("--cache", type=Path, default=ROOT / "checkpoints/masked_video_features.pt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "reports/real_video_100_epochs/investigation/corrected/adaptive_refinement_seed239.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/real_video_100_epochs/visualizations")
    parser.add_argument("--video", type=Path, nargs="+", help="Infer new video file(s); reference cache supplies training class prototypes")
    parser.add_argument("--model-source", help="Local pretrained encoder snapshot; defaults to the project's cached snapshot")
    parser.add_argument("--limit", type=int, help="Limit inference videos for a smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    validate_feature_cache(cache, args.manifest, encoder_padding_policy="masked_attention_v1")
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if state["feature_cache_sha256"] != digest or state["label_schema"] != LABEL_SCHEMA or state["label_map"] != LABEL_MAP:
        raise ValueError("Checkpoint/cache/label mismatch")
    heads = CachedHeads(cache["features"].shape[-1], cache["mean_head"]["3.bias"].numel(), cache["variance_floor"]).eval()
    heads.load_state_dict(state["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "provenance.json").write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), feature_cache_sha256=digest,
        label_map=LABEL_MAP, settings={k: [str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}), indent=2) + "\n")
    if args.mode in ("all", "tsne"): plot_tsne(cache, heads, args.output, state["seed"])
    if args.mode in ("all", "attention"): infer_attention(args, cache, heads)


if __name__ == "__main__":
    main()
