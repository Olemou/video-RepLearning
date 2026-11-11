"""Decision-connected, explicitly qualified multi-layer JEPA diagnostics."""
from __future__ import annotations

import torch

from vision_model.training.online_trajectory import (
    autocast_for, explicit_prefix_forward, rollout_last_tubelet)


def explain_prefix_group(encoder, pixels, score_function, device):
    """Explain one identity-view decision per source, retaining partner/views.

    pixels [N,V,2T,3,H,W] must use the same source/view batch as production.
    score_function receives [N,V,T,D] and returns N decision scalars.
    This is a final-decision attention-gradient adaptation, not original LeGrad.
    All prefixes here must be real; variable-tail batches are rejected.
    """
    if pixels.ndim != 6 or pixels.shape[2] % 2:
        raise ValueError("Need full observed tubelets")
    n, views, frames = pixels.shape[:3]
    time = frames // 2
    cfg = encoder.architecture
    side_h, side_w = pixels.shape[-2] // cfg.patch_size, pixels.shape[-1] // cfg.patch_size
    spatial = side_h * side_w
    layers = len(encoder.blocks)
    explicit = []
    with torch.no_grad():
        for t in range(time):
            clip = pixels[:, :, :2 * (t + 1)].flatten(0, 1).permute(0, 2, 1, 3, 4).to(device)
            with autocast_for(device):
                feature, _, _ = explicit_prefix_forward(encoder, clip, capture=False)
            explicit.append(feature.reshape(n, views, -1))
    reference = torch.stack(explicit, 2)
    leaf = reference.detach().requires_grad_()
    scores = score_function(leaf)
    if scores.shape != (n,):
        raise ValueError("One scalar per source is required")
    upstream = [torch.autograd.grad(scores[j], leaf, retain_graph=j < n - 1)[0].detach()
                for j in range(n)]
    positive = torch.zeros(n, layers, time, side_h, side_w)
    signed = torch.zeros_like(positive)
    attention_gradient = torch.zeros(n, time, side_h, side_w)
    raw = torch.zeros_like(attention_gradient)
    rollout = torch.zeros_like(attention_gradient)
    max_replay_error = 0.0
    for t in range(time):
        clip = pixels[:, :, :2 * (t + 1)].flatten(0, 1).permute(0, 2, 1, 3, 4).to(device).requires_grad_()
        with autocast_for(device):
            feature, attention, _ = explicit_prefix_forward(encoder, clip)
        error = float((feature.detach().reshape(n, views, -1) - reference[:, :, t]).abs().max())
        max_replay_error = max(max_replay_error, error)
        if error > 1e-6:
            raise RuntimeError(f"Explicit attribution replay changed its own features: {error}")
        for j in range(n):
            scalar = (feature.reshape(n, views, -1) * upstream[j][:, :, t]).sum()
            gradients = torch.autograd.grad(scalar, attention, retain_graph=j < n - 1)
            identity = j * views
            for layer, gradient in enumerate(gradients):
                g = gradient[identity]
                positive[j, layer, :t + 1] += g.clamp_min(0).mean((0, 1)).reshape(t + 1, side_h, side_w).cpu()
                signed[j, layer, :t + 1] += g.mean((0, 1)).reshape(t + 1, side_h, side_w).cpu()
            a = attention[-1][identity, :, -spatial:]
            g = gradients[-1][identity, :, -spatial:]
            attention_gradient[j, :t + 1] += (a * g).clamp_min(0).mean((0, 1)).reshape(t + 1, side_h, side_w).detach().cpu()
            raw[j, :t + 1] += a.mean((0, 1)).reshape(t + 1, side_h, side_w).detach().cpu()
            roll = rollout_last_tubelet([a[identity:identity + 1] for a in attention], spatial)
            rollout[j, :t + 1] += roll.reshape(t + 1, side_h, side_w).cpu()
        del feature, attention, gradients, clip
    return dict(explicit_features=reference.detach().cpu(), explicit_scores=scores.detach().cpu(),
        positive_gradients=positive, signed_gradients=signed,
        multilayer_positive=positive.mean(1), final_attention_gradient=attention_gradient,
        raw_final_attention=raw, rollout=rollout,
        replay_max_feature_error=max_replay_error,
        method="all-layer final-trajectory-margin attention gradients; not original layerwise LeGrad",
        aggregation="All heads/queries; mean across layers; sum causal-prefix contributions onto key tubelets",
        rollout_readout="Uniform final-tubelet query; residual-aware product; no CLS; class-agnostic",
        identity_view_only=True, pixel_probabilities=False)
