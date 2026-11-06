#!/usr/bin/env python3
"""Final four-loss gradient audit, without taking any optimizer update."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F
from vision_model.training.trajectory import trajectory_loss
from vision_model.training.online_trajectory import OnlinePrefixEncoder, autocast_for
from attribute_experiment81 import load_case, evaluation_groups
from prepare_experiment81 import normalize_bank, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires the allocated GPU")
    device = torch.device("cuda", 0)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    encoder, model, reps, _, settings = load_case(a.root, "full", device)
    train = reps["train_indices"]
    leaf = reps["features"][train].to(device).detach().requires_grad_()
    loss, detail, _ = trajectory_loss(model, c["teacher"][train, :2].to(device),
        c["valid"][train].to(device), c["timestamps"][train].to(device), c["labels"][train].to(device),
        epoch=20, curriculum_epochs=20, student_features=leaf)
    names = ["semantic_loss", "alignment_loss", "prediction_loss", "retention_loss"]
    weights = [1., .1, 1., .1]
    gradients = [torch.autograd.grad(detail[k], leaf, retain_graph=j < 3)[0].detach() * weights[j]
                 for j, k in enumerate(names)]

    def summaries(vectors):
        norms = [float(v.norm()) for v in vectors]
        cosine = [[float(F.cosine_similarity(x.flatten(), y.flatten(), dim=0, eps=1e-30))
                   for y in vectors] for x in vectors]
        return dict(weighted_gradient_norms=dict(zip(names, norms)), cosine=cosine)

    result = dict(scope="Global 64-source fixed-view/final-curriculum objective; no optimizer steps",
        term_order=names, weights=weights, pooled_feature_gradients=summaries(gradients), prefix_contributions=[],
        checkpoint_sha256=digest(a.root / "full/checkpoint.pt"), localization_validated=False)
    parameters = {
        "patch_embed_weight": encoder.patch_embed.proj.weight,
        "block_00_qkv_weight": encoder.blocks[0].attn.qkv.weight,
        "block_11_qkv_weight": encoder.blocks[11].attn.qkv.weight,
        "block_23_qkv_weight": encoder.blocks[23].attn.qkv.weight,
        "final_norm_weight": encoder.norm.weight}
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    wrapper = OnlinePrefixEncoder(encoder)
    chosen = [int(train[0]), c["ids"].index("cold_folderImage56")]
    groups = evaluation_groups(reps, settings)
    for source in chosen:
        ids = next(g for g in groups if source in g)
        px = torch.stack([normalize_bank(c["pixels"][int(i), :2], c["normalization"]) for i in ids])
        time = int(c["valid"][source].sum()) - 1
        clip = px[:, :, :2 * (time + 1)].flatten(0, 1).permute(0, 2, 1, 3, 4).to(device)
        with autocast_for(device):
            feature = wrapper(clip).reshape(len(ids), 2, -1)
        covectors = []
        local = int((ids == source).nonzero()[0]); at = int((train == source).nonzero()[0])
        for gradient in gradients:
            cotangent = torch.zeros_like(feature)
            cotangent[local] = gradient[at, :, time]
            covectors.append(cotangent)
        raw = []
        for j, cotangent in enumerate(covectors):
            g = torch.autograd.grad((feature * cotangent).sum(), list(parameters.values()), retain_graph=j < 3)
            raw.append([v.detach().cpu() for v in g])
        scope = {name: summaries([raw[t][j] for t in range(4)]) for j, name in enumerate(parameters)}
        feature_error = float((feature.detach().cpu() - reps["features"][ids, :, time]).abs().max())
        result["prefix_contributions"].append(dict(id=c["ids"][source], tubelet=time,
            scope="One source/final-prefix contribution to the global loss, not its complete parameter gradient",
            feature_replay_max_error=feature_error, parameter_gradients=scope))
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
