"""Selective final-attention adaptation from full causal-prefix input tokens.

Only the last attention module and the pointwise projector can train. Frozen
normalizations/MLP still run with autograd. The final block runs in float32:
small attention updates must not disappear in bfloat16 parameter rounding.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from vision_model.models.final_block_maps import replay_last_tubelet
from .patch_vico import PatchProjector


class FinalAttentionStudent(nn.Module):
    def __init__(self, block, norm, *, train_attention=False, hidden=256, embedding_dim=256):
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(block).float()])
        self.norm = copy.deepcopy(norm).float()
        self.requires_grad_(False)
        self.blocks[-1].attn.requires_grad_(train_attention)
        self.projector = PatchProjector(block.attn.qkv.in_features, hidden, embedding_dim)

    def forward(self, prefix, *, frames, height=16, width=16, capture=False):
        with torch.autocast(device_type=prefix.device.type, enabled=False):
            tokens, attention = replay_last_tubelet(self, prefix.float(), frames=frames,
                                                   height=height, width=width)
            q = self.projector(tokens)
        return (q, attention, tokens) if capture else q

    def parameter_audit(self):
        return {name: dict(shape=list(p.shape), count=p.numel(), dtype=str(p.dtype))
                for name, p in self.named_parameters() if p.requires_grad}


def prefix_key_map(attention, gradient=None, *, kind='raw'):
    """Aggregate all heads/consumed queries, retaining the original key times.

    No normalization, threshold, head selection or projection of past keys onto
    the current frame. Positive gradient is a one-block adaptation of LeGrad;
    attention-times-gradient is a different, explicitly named diagnostic.
    """
    if kind == 'raw': value = attention
    elif gradient is None: raise ValueError('A class-target gradient is required')
    elif kind == 'positive_gradient': value = gradient.relu()
    elif kind == 'signed_attention_gradient': value = attention * gradient
    else: raise ValueError(kind)
    return value.mean((1, 2))


def attention_logit_gradient(attention,gradient):
    """Exact softmax-chain derivative, invariant to row-constant A gradients.

    These signed values describe sensitivity to pre-softmax scores. This is a
    separate diagnostic, not LeGrad, a loss, a probability, or pixel relevance.
    """
    return attention*(gradient-(attention*gradient).sum(-1,keepdim=True))
