"""Verified local supervision on the same tokens used by a causal trajectory.

No episode-to-patch labels, latent instance selection, decoder or mirror loss.
The original trajectory targets remain detached; student means retain a graph.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from vision_model.models.final_block_maps import replay_last_tubelet
from .trajectory import DistributedTrajectoryStep


class SpillPatchHead(nn.Module):
    """Pointwise two-prototype logits; sigmoid is over classes, not positions."""
    def __init__(self, width=1024, hidden=128, embedding_dim=128, temperature=.1):
        super().__init__()
        if min(width, hidden, embedding_dim) < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid local head dimensions or temperature")
        self.configuration = dict(width=width, hidden=hidden, embedding_dim=embedding_dim, temperature=temperature)
        self.projector = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                       nn.GELU(), nn.Linear(hidden, embedding_dim))
        self.prototypes = nn.Parameter(torch.randn(2, embedding_dim))
        self.temperature = temperature

    def forward(self, patches):
        z = F.normalize(self.projector(patches.float()), dim=-1)
        direction = F.normalize(self.prototypes, dim=-1)
        return (z @ (direction[1] - direction[0])) / self.temperature


class SpillPrefixStudent(nn.Module):
    """Original final suffix plus a supervised local head, sharing actual tokens."""
    def __init__(self, block, norm, *, train_attention=False, head=None):
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(block).float()])
        self.norm = copy.deepcopy(norm).float()
        self.requires_grad_(False)
        self.blocks[0].attn.requires_grad_(train_attention)
        self.local_head = head or SpillPatchHead(block.attn.qkv.in_features)

    def forward(self, prefix, *, frames, height=16, width=16, capture=False):
        with torch.autocast(device_type=prefix.device.type, enabled=False):
            tokens, attention = replay_last_tubelet(self, prefix.float(), frames=frames, height=height, width=width)
            output = dict(student_features=tokens.mean(1), local_logits=self.local_head(tokens))
        if capture:
            output.update(tokens=tokens, attention=attention)
        return output

    def parameter_audit(self):
        return {name: dict(shape=list(p.shape), count=p.numel())
                for name, p in self.named_parameters() if p.requires_grad}


def foreground_background_loss(logits, targets, *, distributed=True):
    """Balanced partial BCE: -1=ignore, 0=verified non-liquid, 1=verified liquid.

    Every class has equal global weight. A rank can have no annotations or only
    one class; global counts determine normalization, not local point counts.
    DDP's gradient averaging is compensated by world*local_class_sum/count.
    """
    if logits.shape != targets.shape or targets.dtype != torch.long:
        raise ValueError("Need matching logits and long sparse labels")
    if targets.device != logits.device or not ((targets == -1) | (targets == 0) | (targets == 1)).all():
        raise ValueError("Sparse labels must be -1, 0 or 1 on the logits device")
    active = targets >= 0
    if not torch.isfinite(logits[active]).all():
        raise ValueError("Annotated logits must be finite")
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    counts = torch.stack([(targets == label).sum() for label in (0, 1)]).to(dtype=torch.float64)
    if world > 1:
        dist.all_reduce(counts)
    if (counts == 0).any():
        raise ValueError("Training requires verified foreground AND background; no episode-label fallback")
    # Index before evaluation: ignored NaNs/padding never enter softplus.
    negative, positive = logits[targets == 0], logits[targets == 1]
    loss = world * .5 * (F.softplus(negative).double().sum() / counts[0]
                         + F.softplus(-positive).double().sum() / counts[1])
    return loss, dict(background_points=int(counts[0]), foreground_points=int(counts[1]))


def localized_trajectory_loss(trajectory, teacher_targets, student_features, local_logits,
                             sparse_targets, valid, timestamps, labels, *, fg_weight, epoch,
                             distributed=True):
    """Original four terms, plus supervised local BCE, without extra retention."""
    if not math.isfinite(fg_weight) or fg_weight <= 0:
        raise ValueError("Set a positive finite FG/BG coefficient explicitly")
    expected = (*teacher_targets.shape[:-1], local_logits.shape[-1])
    if local_logits.shape != expected or sparse_targets.shape != expected:
        raise ValueError("Local tensors must align with [source,view,time,patch]")
    if (sparse_targets.masked_select(~valid[:, None, :, None].expand_as(sparse_targets)) != -1).any():
        raise ValueError("Padded times cannot carry spatial labels")
    if distributed:
        temporal, detail, output = DistributedTrajectoryStep(trajectory)(
            teacher_targets, valid, timestamps, labels, epoch=epoch, student_features=student_features)
    else:
        from .trajectory import trajectory_loss
        temporal, detail, output = trajectory_loss(trajectory, teacher_targets, valid, timestamps,
            labels, epoch=epoch, student_features=student_features)
    local, coverage = foreground_background_loss(local_logits, sparse_targets, distributed=distributed)
    total = temporal + fg_weight * local
    detail.update(trajectory_loss=temporal, foreground_background_loss=local,
                  joint_loss=total, fg_weight=fg_weight, **coverage)
    return total, detail, output
