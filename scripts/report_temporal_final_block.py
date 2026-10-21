#!/usr/bin/env python3
"""Inspect the trainable final temporal block separately from frozen V-JEPA."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vision_model.training.trajectory import TrajectoryModel, standardized
from vision_model.models.final_block_maps import temporal_layer_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "reports/trajectory_thermal_aug_100")
    args = p.parse_args()
    torch.set_num_threads(4)
    cache = torch.load(args.run / "causal_features.pt", map_location="cpu", weights_only=False)
    ckpt = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    trained = TrajectoryModel(ckpt["width"], ckpt["hidden"], ckpt["embedding_dim"])
    trained.load_state_dict(ckpt["model"])
    torch.manual_seed(239)
    initial = TrajectoryModel(ckpt["width"], ckpt["hidden"], ckpt["embedding_dim"])
    models = dict(initial=initial.eval(), trained=trained.eval())
    out = args.run / "final_block"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, video_id in enumerate(("warm_scene_1_hot", "cold_scene3", "clean_videoFrame99")):
        i = cache["ids"].index(video_id)
        valid, times = cache["valid"][i:i+1], cache["timestamps"][i:i+1]
        count = int(valid.sum())
        record = dict(video_id=video_id)
        for col, (name, model) in enumerate(models.items()):
            captured = {}
            def hook(module, values):
                captured["hidden"] = values[0].detach()
            handle = model.temporal.layers[-1].register_forward_pre_hook(hook)
            with torch.no_grad():
                output = model(cache["features"][i:i+1, 0], valid, times)
                attention = temporal_layer_attention(model.temporal.layers[-1], captured["hidden"], valid)[0, :, :count, :count]
                score = model.event_score(standardized(output["states"])).squeeze(-1).masked_fill(~valid, -torch.inf)
                beta = score.softmax(1)[0, :count]
            handle.remove()
            record[name] = dict(final_temporal_attention=attention.tolist(), event_weights=beta.tolist(),
                                actual_direct_pooling_coefficients=(.5/count + .5*beta).tolist())
            ax = axes[row, col]
            im = ax.imshow(attention.mean(0), vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
            ax.set_title(f"{video_id}\n{name}: final temporal block, 4-head mean", fontsize=9)
            ax.set_xlabel("Past key tubelet"); ax.set_ylabel("Query tubelet")
            fig.colorbar(im, ax=ax, shrink=.7)
        ax = axes[row, 2]
        trained_record = record["trained"]
        ax.plot(range(1,count+1), trained_record["event_weights"], marker="o", label="Displayed event beta")
        ax.plot(range(1,count+1), trained_record["actual_direct_pooling_coefficients"], marker=".", label="0.5/T + 0.5 beta")
        ax.axhline(1/count, linestyle="--", color="gray", label="Uniform")
        ax.set_title("Trained final pooling: mean + event weights", fontsize=9)
        ax.set_xlabel("Tubelet"); ax.grid(alpha=.2); ax.legend(fontsize=7)
        results.append(record)
    fig.suptitle("The trained final adapter block attends over TIME; it has no spatial patch axis")
    fig.savefig(out / "temporal_final_block.png", dpi=160)
    plt.close(fig)
    delta = float((trained.temporal.layers[-1].self_attn.in_proj_weight - initial.temporal.layers[-1].self_attn.in_proj_weight).abs().max())
    (out / "temporal_report.json").write_text(json.dumps(dict(final_temporal_qkv_max_weight_change=delta,
         caveat="Causal row 0 must attend to itself with probability 1; this is not event evidence.", videos=results), indent=2))
    print(f"Final temporal QKV maximum weight change: {delta:.8f}; saved {out / 'temporal_final_block.png'}")


if __name__ == "__main__":
    main()
