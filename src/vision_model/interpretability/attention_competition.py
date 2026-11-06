"""Opt-in sensitivity to normalized attention competition, not liquid scores."""
import torch

from vision_model.training.online_trajectory import autocast_for, explicit_prefix_forward


def attention_logit_gradient(probability, probability_gradient):
    """Exact softmax VJP: dS/dlogit = A * (dS/dA - sum_key A*dS/dA).

    A row-constant derivative changes the extension outside the probability
    simplex but has no effect on the actual normalized attention competition.
    This is an ordinary chain-rule diagnostic, not LeGrad or relevance LRP.
    """
    if probability.shape != probability_gradient.shape or probability.ndim < 2:
        raise ValueError("Need matching query/key attention and derivative tensors")
    return probability * (probability_gradient -
        (probability * probability_gradient).sum(-1, keepdim=True))


def explain_attention_competition(encoder, pixels, reference, score_function, device):
    """Replay all real prefixes against saved explicit-attention features.

    Retain every head, query, key and block before aggregation. Positive and
    signed key sensitivity are separate outputs. No image labels/ROIs are used
    to select a layer, head, threshold or region.
    """
    n, views, frames = pixels.shape[:3]
    time = frames // 2
    h, w = (s // encoder.architecture.patch_size for s in pixels.shape[-2:])
    layers = len(encoder.blocks)
    if frames % 2 or reference.shape[:3] != (n, views, time):
        raise ValueError("Require matching complete, valid causal prefixes")
    leaf = reference.to(device).detach().requires_grad_()
    scores = score_function(leaf)
    upstream = [torch.autograd.grad(scores[j], leaf, retain_graph=j < n-1)[0].detach() for j in range(n)]
    output = {key: torch.zeros(n, layers, time, h, w) for key in
        ("probability_positive", "weighted_positive", "logit_positive", "logit_signed")}
    statistics, replay_error = [], 0.
    for t in range(time):
        clip = pixels[:, :, :2*(t+1)].flatten(0,1).permute(0,2,1,3,4).to(device).requires_grad_()
        with autocast_for(device):
            feature, attention, _ = explicit_prefix_forward(encoder, clip)
        error = float((feature.detach().reshape(n, views, -1) - leaf[:, :, t].detach()).abs().max())
        replay_error = max(replay_error, error)
        if error > 1e-6:
            raise RuntimeError(f"Saved explicit-feature replay changed: {error}")
        for j in range(n):
            scalar = (feature.reshape(n,views,-1) * upstream[j][:, :, t]).sum()
            derivatives = torch.autograd.grad(scalar, attention, retain_graph=j<n-1)
            for layer, (a, g) in enumerate(zip(attention, derivatives)):
                a, g = a[j*views].detach(), g[j*views].detach()
                projected = attention_logit_gradient(a, g)
                values = dict(probability_positive=g.clamp_min(0), weighted_positive=(a*g).clamp_min(0),
                              logit_positive=projected.clamp_min(0), logit_signed=projected)
                for key, value in values.items():
                    output[key][j, layer, :t+1] += value.mean((0,1)).reshape(t+1,h,w).cpu()
                common = (a*g).sum(-1)
                energy = (a*g.square()).sum(-1).sum()
                statistics.append(dict(source_in_group=j, prefix_tubelet=t, block=layer,
                    row_constant_weighted_energy_fraction=float(common.square().sum()/energy.clamp_min(1e-30)),
                    logit_gradient_max_row_sum_error=float(projected.sum(-1).abs().max()),
                    logit_gradient_l1=float(projected.abs().sum())))
        del clip, feature, attention, derivatives
    output.update(scores=scores.detach().cpu(), statistics=statistics, explicit_replay_max_error=replay_error,
        target="Final trained prototype spill-minus-clean cosine margin",
        method="Exact softmax chain-rule logit sensitivity; not a reproduced published localization method",
        aggregation="Mean all heads and queries, mean blocks for aggregate; sum real-prefix contributions onto key tubelets",
        pixel_probability=False, localization_validated=False)
    return output
