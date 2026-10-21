"""Memory-bounded, class-independent incoming attention for one video."""
import math
import torch
from .encoder import _rotate


@torch.no_grad()
def received_attention(module, value, *, frames, height, width, token_valid_mask=None, chunk_size=128):
    """Average softmax(QK'/sqrt(d)) over heads and valid query tokens.

    Uses the model's RoPE layout; excludes invalid queries AND keys. Computes
    probabilities in float32, so bf16 fused-attention rounding may differ.
    Returned [T,H,W] is probability mass over keys, summing to one.
    """
    if value.shape[0] != 1 or chunk_size < 1:
        raise ValueError("Attention export requires batch size one and a positive chunk size")
    tokens = value.shape[1]
    if tokens != frames * height * width:
        raise ValueError("Token/grid mismatch")
    valid = (torch.ones(tokens, dtype=torch.bool, device=value.device) if token_valid_mask is None
             else token_valid_mask[0].bool())
    if valid.shape != (tokens,) or not valid.any():
        raise ValueError("At least one valid token is required")
    query, key, _ = module.qkv(value).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
    positions = module._positions(torch.arange(tokens, device=value.device), height, width)
    offset, qs, ks = 0, [], []
    for size, position in zip((module.d_dim, module.h_dim, module.w_dim), positions):
        qs.append(_rotate(query[..., offset:offset + size], position))
        ks.append(_rotate(key[..., offset:offset + size], position))
        offset += size
    qs.append(query[..., offset:])
    ks.append(key[..., offset:])
    q = torch.cat(qs, -1)[0, :, valid].float()
    k = torch.cat(ks, -1)[0, :, valid].float()
    mass = torch.zeros(int(valid.sum()), device=value.device)
    # Disable surrounding encoder autocast for the probability computation.
    with torch.autocast(device_type=value.device.type, enabled=False):
        for start in range(0, q.shape[1], chunk_size):
            probability = (q[:, start:start + chunk_size] @ k.transpose(-1, -2) / math.sqrt(module.head_dim)).softmax(-1)
            mass += probability.sum((0, 1))
    result = torch.zeros(tokens, device=value.device)
    result[valid] = mass / (module.num_heads * q.shape[1])
    return result.reshape(frames, height, width)
