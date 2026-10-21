"""Final-block replay for exact query selection and class-conditioned VJPs.

This is a one-block attention-times-gradient diagnostic, not full transformer
relevance propagation, pixel attribution, or a trained segmentation output.
"""
import math

import torch

from .encoder import _rotate


def replay_last_tubelet(encoder, hidden, *, frames: int, height: int, width: int):
    """Return final normalized last-tubelet tokens and their attention [1,H,S,L].

    Last-block MLPs and normalization operate tokenwise. Only the final S queries
    are consumed by this pipeline, so reconstructing them is sufficient. All
    prefix tokens remain keys/values. RoPE layout matches the custom encoder.
    Probabilities are calculated in float32; bf16 fused rounding can differ.
    """
    spatial = height * width
    if hidden.shape[0] != 1 or hidden.shape[1] != frames * spatial:
        raise ValueError("Expected one unpadded video with matching token grid")
    block = encoder.blocks[-1]
    module = block.attn
    value = block.norm1(hidden)
    q, k, v = module.qkv(value).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
    positions = module._positions(torch.arange(hidden.shape[1], device=hidden.device), height, width)
    offset, qs, ks = 0, [], []
    for size, position in zip((module.d_dim, module.h_dim, module.w_dim), positions):
        qs.append(_rotate(q[..., offset:offset + size], position))
        ks.append(_rotate(k[..., offset:offset + size], position))
        offset += size
    qs.append(q[..., offset:])
    ks.append(k[..., offset:])
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        query, key = torch.cat(qs, -1)[..., -spatial:, :].float(), torch.cat(ks, -1).float()
        attention = (query @ key.transpose(-1, -2) / math.sqrt(module.head_dim)).softmax(-1)
    attended = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(1, spatial, hidden.shape[-1])
    state = hidden[:, -spatial:] + module.proj(attended)
    state = state + block.mlp(block.norm2(state))
    return encoder.norm(state), attention


def temporal_layer_attention(layer, hidden, valid):
    """Reconstruct all heads of a pre-normalized causal temporal adapter layer."""
    module = layer.self_attn
    value = layer.norm1(hidden)
    batch, time, width = value.shape
    q, k, _ = torch.nn.functional.linear(value, module.in_proj_weight, module.in_proj_bias).unflatten(
        -1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
    score = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(width / module.num_heads)
    causal = torch.ones(time, time, dtype=torch.bool, device=value.device).triu(1)
    score = score.masked_fill(causal[None, None] | ~valid[:, None, None, :], -torch.inf)
    probability = score.softmax(-1)
    return probability.masked_fill(~valid[:, None, :, None], 0)
