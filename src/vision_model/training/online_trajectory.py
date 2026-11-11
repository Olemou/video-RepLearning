"""Opt-in online V-JEPA trajectory training; original cached trainer is unchanged.

Pooled feature gradients can be replayed through independently encoded causal
prefixes. This bounds activation memory without changing the global loss.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vision_model.models.encoder import _rotate


class OnlinePrefixEncoder(nn.Module):
    """One observed prefix in, final tubelet's spatial mean out.

    Weights remain FP32; CUDA forward may use BF16 autocast. No full-video
    encode is reused to construct earlier prefix features.
    """

    def __init__(self, encoder, *, activation_checkpointing=False):
        super().__init__()
        self.encoder = encoder
        self.activation_checkpointing = activation_checkpointing

    def forward(self, clip):
        cfg = self.encoder.architecture
        if clip.ndim != 5 or clip.shape[2] % cfg.tubelet_size:
            raise ValueError("Expected complete tubelets [B,3,T,H,W]")
        frames = clip.shape[2] // cfg.tubelet_size
        height, width = clip.shape[-2] // cfg.patch_size, clip.shape[-1] // cfg.patch_size
        if self.activation_checkpointing and torch.is_grad_enabled():
            value = self.encoder.patch_embed(clip)
            for block in self.encoder.blocks:
                value = checkpoint(block, value, frames=frames, height=height,
                                   width=width, use_reentrant=False)
            value = self.encoder.norm(value)
        else:
            value = self.encoder(clip)
        return value[:, -(height * width):].mean(1).float()


def autocast_for(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def prefix_batches(pixels, valid, chunk_sources=2):
    """Yield source/view-flattened prefixes and their flat source indices.

    pixels: [source,view,frame,channel,height,width], already normalized.
    Only complete valid tubelets are ever sent to the encoder.
    """
    if pixels.ndim != 6 or valid.shape != (pixels.shape[0], pixels.shape[2] // 2):
        raise ValueError("Expected pixels [N,V,2T,3,H,W], valid [N,T]")
    if chunk_sources < 1 or not valid.any(1).all() or not valid[:, 0].all():
        raise ValueError("Need positive chunk size and nonempty valid prefixes")
    if ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("Validity must be a contiguous prefix")
    views = pixels.shape[1]
    for t in range(valid.shape[1]):
        sources = valid[:, t].nonzero().flatten()
        for begin in range(0, len(sources), chunk_sources):
            selected = sources[begin:begin + chunk_sources]
            clip = pixels[selected, :, :2 * (t + 1)]
            clip = clip.flatten(0, 1).permute(0, 2, 1, 3, 4).contiguous()
            yield t, selected, clip


@torch.no_grad()
def collect_prefix_features(model, pixels, valid, device, *, chunk_sources=2):
    width = model.encoder.architecture.hidden_size
    result = torch.zeros(pixels.shape[0], pixels.shape[1], valid.shape[1], width,
                         device=device)
    for t, sources, clip in prefix_batches(pixels, valid, chunk_sources):
        with autocast_for(device):
            value = model(clip.to(device=device))
        result[sources.to(device), :, t] = value.reshape(len(sources), pixels.shape[1], width)
    return result


def replay_prefix_gradients(wrapped, pixels, valid, cotangent, device, *,
                            chunk_sources=2, replay_reference=None, max_replay_error=1e-6):
    """Apply pooled-feature cotangents, synchronizing DDP only on the last chunk.

    With DistributedTrajectoryStep's replicated global objective the local
    cotangents carry a world-size factor. DDP's averaging cancels that factor.
    Do not apply an additional world-size division here.
    """
    batches = list((t, sources) for t, sources, _ in prefix_batches(pixels, valid, chunk_sources))
    maximum = 0.0
    for j, (t, sources, clip) in enumerate(prefix_batches(pixels, valid, chunk_sources)):
        sync = (wrapped.no_sync() if hasattr(wrapped, "no_sync") and j < len(batches) - 1
                else nullcontext())
        with sync, autocast_for(device):
            actual = wrapped(clip.to(device=device)).reshape(len(sources), pixels.shape[1], -1)
            if replay_reference is not None:
                error = float((actual.detach() - replay_reference[sources.to(device), :, t]).abs().max())
                maximum = max(maximum, error)
                if error > max_replay_error:
                    raise RuntimeError(f"Gradient replay changed prefix features: {error:g}")
            (actual * cotangent[sources.to(device), :, t]).sum().backward()
    return maximum


def parameter_groups(encoder):
    return {"patch_embed": list(encoder.patch_embed.parameters()),
            **{f"block_{i:02d}": list(block.parameters())
               for i, block in enumerate(encoder.blocks)},
            "final_norm": list(encoder.norm.parameters())}


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def explicit_attention(module, value, *, frames, height, width):
    """Attention used by the forward itself, with differentiable probabilities.

    This diagnostic path is separate from production SDPA. Its pooled feature
    and decision parity must be measured; it is never silently treated as exact
    BF16 FlashAttention replay.
    """
    batch, tokens, channels = value.shape
    if tokens != frames * height * width:
        raise ValueError("Attention grid mismatch")
    q, k, v = module.qkv(value).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
    positions = module._positions(torch.arange(tokens, device=value.device), height, width)
    offset, qs, ks = 0, [], []
    for size, position in zip((module.d_dim, module.h_dim, module.w_dim), positions):
        qs.append(_rotate(q[..., offset:offset + size], position))
        ks.append(_rotate(k[..., offset:offset + size], position))
        offset += size
    if offset < module.head_dim:
        qs.append(q[..., offset:]); ks.append(k[..., offset:])
    with torch.autocast(device_type=value.device.type, enabled=False):
        probability = (torch.cat(qs, -1).float() @ torch.cat(ks, -1).float().transpose(-1, -2)
                       / math.sqrt(module.head_dim)).softmax(-1)
        attended = probability @ v.float()
    # SDPA returns the input dtype; preserve that boundary before projection.
    return module.proj(attended.to(v.dtype).transpose(1, 2).reshape(batch, tokens, channels)), probability


def explicit_prefix_forward(encoder, clip, *, capture=True):
    cfg = encoder.architecture
    frames, height, width = clip.shape[2] // cfg.tubelet_size, clip.shape[-2] // cfg.patch_size, clip.shape[-1] // cfg.patch_size
    value = encoder.patch_embed(clip)
    attention = []
    for block in encoder.blocks:
        output, probability = explicit_attention(block.attn, block.norm1(value),
                                                  frames=frames, height=height, width=width)
        value = value + output
        value = value + block.mlp(block.norm2(value))
        if capture:
            attention.append(probability)
    tokens = encoder.norm(value)
    return tokens[:, -(height * width):].mean(1).float(), attention, tokens


def rollout_last_tubelet(attention, spatial):
    """Mean last-tubelet query times residual-aware attention product.

    No CLS token is invented. Vector propagation avoids a dense rollout product.
    """
    if not attention:
        raise ValueError("Need all attention layers")
    length = attention[0].shape[-1]
    query = attention[0].new_zeros(length)
    query[-spatial:] = 1.0 / spatial
    for probability in reversed(attention):
        matrix = probability.detach().float().mean((0, 1))
        # Each attention row sums to one; normalize after adding residual I.
        query = (query @ matrix + query) * 0.5
    return query
