"""Weakly supervised spatiotemporal multiple-instance spill localizer."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SpillLocalizationOutput:
    token_logits: torch.Tensor
    video_logits: torch.Tensor


class SpillMILHead(nn.Module):
    """Predict patch evidence and aggregate only the strongest valid instances.

    Video labels supervise spatial tokens before pooling. This is deliberately
    separate from generic transformer attention, which is not class-conditioned.
    """

    def __init__(self, hidden_size: int = 1024, projection_size: int = 128,
                 top_fraction: float = 0.05) -> None:
        super().__init__()
        if hidden_size <= 0 or projection_size <= 0 or not 0 < top_fraction <= 1:
            raise ValueError("Invalid localizer dimensions or top fraction")
        self.top_fraction = top_fraction
        self.token_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
            nn.Linear(projection_size, 1),
        )

    def forward(self, tokens: torch.Tensor, temporal_valid_mask: torch.Tensor) -> SpillLocalizationOutput:
        if tokens.ndim != 5:
            raise ValueError("tokens must be [B,T,H,W,D]")
        batch, time, height, width, _ = tokens.shape
        valid = temporal_valid_mask.to(device=tokens.device, dtype=torch.bool)
        if valid.shape != (batch, time) or not valid.any(1).all():
            raise ValueError("temporal_valid_mask must be [B,T] with a valid tubelet per video")
        logits = self.token_head(tokens.float()).squeeze(-1)
        # A spill occupies a neighborhood. Aggregate a 3x3 local field so a
        # single isolated patch cannot satisfy the positive video objective.
        logits = F.avg_pool2d(logits.flatten(0, 1).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        logits = logits.unflatten(0, (batch, time))
        logits = logits.masked_fill(~valid[:, :, None, None], -torch.inf)
        video_logits = []
        for index in range(batch):
            values = logits[index][valid[index]].flatten()
            count = max(1, math.ceil(values.numel() * self.top_fraction))
            video_logits.append(values.topk(count).values.mean())
        return SpillLocalizationOutput(logits, torch.stack(video_logits))


class SpillSegmentationHead(nn.Module):
    """Decode local V-JEPA tubelet tokens into spatial spill logits.

    Unlike transformer attention or video-level MIL, this head is trained from
    pixel masks.  Its output is intentionally kept on the encoder's 16x16 grid;
    interpolation to video resolution is a rendering operation, not supervision.
    """

    def __init__(self, hidden_size: int = 1024, decoder_size: int = 128) -> None:
        super().__init__()
        if hidden_size <= 0 or decoder_size < 2:
            raise ValueError("hidden_size must be positive and decoder_size must be at least 2")
        self.normalization = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, decoder_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_size, decoder_size, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_size, decoder_size // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_size // 2, 1, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("tokens must be [B,T,H,W,D]")
        batch, time, height, width, _ = tokens.shape
        features = self.projection(self.normalization(tokens.float()))
        features = features.permute(0, 1, 4, 2, 3).reshape(batch * time, -1, height, width)
        return self.decoder(features).reshape(batch, time, height, width)


def spill_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_tubelets: torch.Tensor,
    *,
    positive_weight: float | torch.Tensor = 1.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted BCE plus soft Dice over explicitly annotated tubelets only."""
    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("logits and targets must have matching [B,T,H,W] shapes")
    if supervised_tubelets.shape != logits.shape[:2]:
        raise ValueError("supervised_tubelets must be [B,T]")
    selected = supervised_tubelets.to(device=logits.device, dtype=torch.bool)
    if not selected.any():
        raise ValueError("At least one annotated tubelet is required")
    prediction = logits[selected]
    truth = targets.to(device=logits.device, dtype=prediction.dtype)[selected].clamp(0, 1)
    weight = torch.as_tensor(positive_weight, device=logits.device, dtype=prediction.dtype)
    if weight.numel() != 1 or weight <= 0 or not torch.isfinite(weight):
        raise ValueError("positive_weight must be one finite positive scalar")
    bce = F.binary_cross_entropy_with_logits(prediction, truth, pos_weight=weight)
    probability = prediction.sigmoid()
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    # Empty ground-truth masks are governed by BCE. Excluding them from Dice
    # avoids an epsilon-only term whose gradient is effectively zero.
    positive = truth.sum(dim=(-2, -1)) > 0
    dice = (1 - (2 * intersection[positive] + 1) / (denominator[positive] + 1)).mean() if positive.any() else bce.new_zeros(())
    total = bce + float(dice_weight) * dice
    return total, {"binary_cross_entropy": bce.detach(), "dice_loss": dice.detach()}


def spill_mil_loss(output: SpillLocalizationOutput, labels: torch.Tensor,
                   temporal_valid_mask: torch.Tensor, *, sparsity_weight: float = 0.05,
                   smoothness_weight: float = 0.02) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Video MIL loss plus clean-token, sparse-positive and smoothness constraints."""
    labels = labels.to(device=output.video_logits.device, dtype=torch.float32)
    valid = temporal_valid_mask.to(device=output.video_logits.device, dtype=torch.bool)
    if labels.shape != output.video_logits.shape:
        raise ValueError("labels must have one value per video")
    # Equal total mass per class prevents the 17/7 split from dominating.
    counts = torch.stack(((labels == 0).sum(), (labels == 1).sum())).clamp_min(1)
    sample_weight = torch.where(labels.bool(), labels.numel() / (2 * counts[1]),
                                labels.numel() / (2 * counts[0])).to(output.video_logits.dtype)
    video = (F.binary_cross_entropy_with_logits(output.video_logits, labels, reduction="none") * sample_weight).mean()
    probability = output.token_logits.sigmoid()
    clean_terms, positive_terms, sparse_terms, smooth_terms = [], [], [], []
    for index in range(len(labels)):
        logits = output.token_logits[index][valid[index]]
        probs = probability[index][valid[index]]
        if labels[index] < .5:
            # A clean label is strong negative evidence for every observed patch.
            clean_terms.append(F.softplus(logits).mean())
        else:
            count = max(1, math.ceil(logits.numel() * .05))
            positive_terms.append(F.softplus(-logits.flatten().topk(count).values).mean())
            sparse_terms.append(probs.mean())
        smooth_terms.extend((
            (probs[:, 1:] - probs[:, :-1]).abs().mean(),
            (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean(),
        ))
        if len(probs) > 1:
            smooth_terms.append((probs[1:] - probs[:-1]).abs().mean())
    zero = output.video_logits.new_zeros(())
    clean = torch.stack(clean_terms).mean() if clean_terms else zero
    positive = torch.stack(positive_terms).mean() if positive_terms else zero
    sparsity = torch.stack(sparse_terms).mean() if sparse_terms else zero
    smoothness = torch.stack(smooth_terms).mean() if smooth_terms else zero
    total = video + .25 * clean + .25 * positive + sparsity_weight * sparsity + smoothness_weight * smoothness
    return total, dict(video=video.detach(), clean_tokens=clean.detach(), positive_top=positive.detach(),
                       positive_sparsity=sparsity.detach(), smoothness=smoothness.detach())
