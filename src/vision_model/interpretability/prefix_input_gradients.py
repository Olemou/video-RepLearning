"""Production-forward input gradients for independently encoded causal prefixes."""
import torch

from vision_model.training.online_trajectory import (
    autocast_for, collect_prefix_features, prefix_batches)


def prefix_input_gradient(wrapper, pixels, valid, score_function, device):
    """Return a scalar score and its gradient at normalized input pixels.

    Replay the actual SDPA path, preserving source/view batching. Contributions
    from later prefixes accumulate onto their earlier observed input frames.
    No attention substitution, parameter update or teacher gradient is used.
    """
    pixels = pixels.detach().float()
    reference = collect_prefix_features(wrapper, pixels, valid, device,
                                        chunk_sources=len(pixels))
    leaf = reference.detach().requires_grad_()
    score = score_function(leaf)
    if score.ndim != 0 or not torch.isfinite(score):
        raise ValueError("Expected one finite differentiable scalar")
    cotangent, = torch.autograd.grad(score, leaf)
    result = torch.zeros_like(pixels)
    replay_error = 0.
    for t, sources, clip in prefix_batches(pixels, valid, len(pixels)):
        clip = clip.to(device).detach().requires_grad_()
        with autocast_for(device):
            feature = wrapper(clip).reshape(len(sources), pixels.shape[1], -1)
        error = float((feature.detach() - reference[sources.to(device), :, t]).abs().max())
        replay_error = max(replay_error, error)
        if error > 1e-6:
            raise RuntimeError(f"Input-gradient production replay differs: {error}")
        gradient, = torch.autograd.grad(
            (feature * cotangent[sources.to(device), :, t]).sum(), clip)
        result[sources, :, :2 * (t + 1)] += gradient.detach().cpu().permute(
            0, 2, 1, 3, 4).reshape(len(sources), pixels.shape[1], 2 * (t + 1),
                                  *pixels.shape[-3:])
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite input gradient")
    return dict(score=float(score.detach()), gradient=result,
                features=reference.detach().cpu(), replay_max_error=replay_error)


def trapezoid_attributions(start, end, gradients):
    """Trapezoidal straight-path IG; callers must check score completeness."""
    if len(gradients) < 2 or any(g.shape != end.shape for g in gradients):
        raise ValueError("Need at least two matching endpoint/path gradients")
    integral = (gradients[0] + gradients[-1]) * .5
    for g in gradients[1:-1]:
        integral = integral + g
    return (end - start) * (integral / (len(gradients) - 1))
