"""Differentiable all-head V-JEPA attention capture for layer comparisons.

This explains a supplied scalar through a frozen encoder suffix. It is not
LeGrad's intermediate-classifier method, segmentation, or input attribution.
"""
from contextlib import contextmanager
import math
import types

import torch

from vision_model.models.encoder import _rotate


@contextmanager
def capture_spatial_attention(encoder, layers):
    """Use captured post-softmax probabilities in the actual value product.

    Restore every forward even on failure. No token or head is selected.
    Explicit probabilities use float32 as in final-block replay; callers must
    compare replay features and scalar with the original fused forward.
    """
    modules = [encoder.blocks[i].attn for i in layers]
    originals = [(m.forward, "forward" in m.__dict__) for m in modules]
    captured = {}
    try:
        for index, module in zip(layers, modules):
            def forward(self, value, *, frames, height, width, token_valid_mask=None, _index=index):
                batch, tokens, channels = value.shape
                if tokens != frames * height * width:
                    raise ValueError("Token/grid mismatch")
                q, k, v = self.qkv(value).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
                positions = self._positions(torch.arange(tokens, device=value.device), height, width)
                offset, qs, ks = 0, [], []
                for size, position in zip((self.d_dim, self.h_dim, self.w_dim), positions):
                    qs.append(_rotate(q[..., offset:offset + size], position))
                    ks.append(_rotate(k[..., offset:offset + size], position))
                    offset += size
                qs.append(q[..., offset:])
                ks.append(k[..., offset:])
                with torch.autocast(device_type=value.device.type, enabled=False):
                    logits = torch.cat(qs, -1).float() @ torch.cat(ks, -1).float().transpose(-1, -2)
                    logits = logits / math.sqrt(self.head_dim)
                    if token_valid_mask is not None:
                        logits = logits.masked_fill(~token_valid_mask[:, None, None, :], -torch.inf)
                    attention = logits.softmax(-1)
                captured[_index] = attention
                attended = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(batch, tokens, channels)
                return self.proj(attended)
            module.forward = types.MethodType(forward, module)
        yield captured
    finally:
        for module, (original, was_instance) in zip(modules, originals):
            if was_instance:
                module.forward = original
            else:
                del module.forward
