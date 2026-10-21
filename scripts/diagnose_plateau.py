#!/usr/bin/env python3
"""Numerical audit of the real 100-epoch checkpoint, without changing it."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from convergence_study import CachedHeads, grad_norm
from simulate_loss_training import gaussian_output
from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss


def main():
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = torch.load(ROOT / "checkpoints/real_video_100_features.pt", weights_only=True)
    checkpoint = torch.load(ROOT / "reports/real_video_100_epochs/constant_lr_seed239.pt", weights_only=True)
    features, valid, labels = [cache[k].to(device) for k in ("features", "valid", "labels")]
    model = CachedHeads(1024, 256, cache["variance_floor"]).to(device)
    model.load_state_dict(checkpoint["model"])
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(**checkpoint["loss_config"]))
    output = gaussian_output(model, features, valid)
    result = objective(output=output, labels=labels, original_ids=cache["ids"], epoch=19)
    (result.loss_sum / result.sample_count).backward()
    diagnostics = dict(mean_head_gradient=grad_norm(model.mean_head.parameters()),
                       variance_head_gradient=grad_norm(model.variance_head.parameters()))
    with torch.no_grad():
        delta = ((output.video_mean[:, None] - output.video_mean[None, :]).square() /
                 (output.video_variance[:, None] + output.video_variance[None, :] + 1e-6)).mean(-1)
        negative = labels[:, None].ne(labels[None, :])
        probabilities = torch.softmax((-delta / .1).masked_fill(~negative, -torch.inf), 1)
        entropy = -(probabilities * probabilities.clamp_min(1e-38).log()).sum(1)
        r2 = (output.temporal_mean - output.video_mean[:, None]).square()
        optimum = r2.clamp_min(model.variance_floor)
        lower_uncertainty = (((r2 / optimum + optimum.log()) * valid[..., None]).sum((1, 2)) /
                             (valid.sum(1) * r2.shape[-1])).mean()
        lower_bound = result.positive_entropy_sum / result.sample_count + .01 * lower_uncertainty
        diagnostics.update(
            observed_total=float(result.loss_sum / result.sample_count),
            fixed_residual_and_positive_entropy_lower_bound=float(lower_bound),
            maximum_improvement_with_those_quantities_fixed=float(result.loss_sum / result.sample_count - lower_bound),
            padded_tubelet_fraction=float((~valid).float().mean()),
            effective_negative_count_mean=float(entropy.exp().mean()),
            available_negative_count_mean=float(negative.sum(1).float().mean()),
            negative_probability_below_1e_6_fraction=float((probabilities[negative] < 1e-6).float().mean()),
            mean_video_norm=float(output.video_mean.norm(dim=-1).mean()),
        )
        scaled = SimpleNamespace(temporal_mean=output.temporal_mean * 2,
                                 video_mean=output.video_mean * 2,
                                 temporal_variance=output.temporal_variance * 4,
                                 video_variance=output.video_variance * 4,
                                 temporal_valid_mask=valid)
        shifted = objective(output=scaled, labels=labels, original_ids=cache["ids"], epoch=19)
        diagnostics["double_latent_scale_contrastive_change"] = float((shifted.contrastive_sum - result.contrastive_sum) / result.sample_count)
        diagnostics["double_latent_scale_total_change"] = float((shifted.loss_sum - result.loss_sum) / result.sample_count)
    means = output.video_mean.detach().requires_grad_()
    probe = SimpleNamespace(**{**output.__dict__, "video_mean": means})
    loss = objective(output=probe, labels=labels, original_ids=cache["ids"], epoch=19)
    gradient = torch.autograd.grad(loss.loss_sum, means)[0]
    radial = (gradient * means).sum(-1).abs() / (gradient.norm(dim=-1) * means.norm(dim=-1)).clamp_min(1e-12)
    diagnostics["maximum_radial_gradient_fraction"] = float(radial.max())
    masked_path = ROOT / "checkpoints/masked_video_features.pt"
    if masked_path.exists():
        masked = torch.load(masked_path, weights_only=True)
        a, b = cache["features"][cache["valid"]], masked["features"][masked["valid"]]
        diagnostics["valid_feature_cosine_before_after_attention_fix"] = float(F.cosine_similarity(a, b).mean())
        diagnostics["relative_valid_feature_change"] = float((a - b).norm() / a.norm())
    directory = ROOT / "reports/real_video_100_epochs/investigation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
