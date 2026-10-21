"""Stable uncertainty-weighted supervised contrastive objective from main.pdf."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GaussianContrastiveLossConfig:
    temperature: float = 0.1
    compatibility_scale: float = 1.0
    positive_lambda_min: float = 0.1
    negative_temperature_high: float = 1.0
    negative_temperature_low: float = 0.1
    uncertainty_weight: float = 0.01
    epsilon: float = 1e-6
    total_epochs: int = 100
    pair_weighting: bool = True

    def __post_init__(self) -> None:
        positive = {
            "temperature": self.temperature,
            "compatibility_scale": self.compatibility_scale,
            "negative_temperature_high": self.negative_temperature_high,
            "negative_temperature_low": self.negative_temperature_low,
            "epsilon": self.epsilon,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.positive_lambda_min <= 1:
            raise ValueError("positive_lambda_min must be in [0,1]")
        if not math.isfinite(self.uncertainty_weight) or self.uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be non-negative")
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")


@dataclass(frozen=True)
class GaussianLossOutput:
    loss_sum: torch.Tensor
    sample_count: int
    contrastive_sum: torch.Tensor
    positive_entropy_sum: torch.Tensor
    uncertainty_sum: torch.Tensor
    positive_weight_sum: torch.Tensor
    negative_weight_sum: torch.Tensor
    positive_lambda: float
    negative_temperature: float


def _identity_tensor(ids: torch.Tensor | Sequence[Any], device: torch.device) -> torch.Tensor:
    if torch.is_tensor(ids):
        if ids.ndim != 1:
            raise ValueError("original_ids tensor must be one-dimensional")
        return ids.to(device=device)
    mapping: dict[Any, int] = {}
    encoded = []
    for value in ids:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return torch.tensor(encoded, device=device, dtype=torch.long)


class StableUncertaintyContrastiveLoss(nn.Module):
    """One weighted SupCon objective plus detached-residual uncertainty NLL."""

    name = "stable-uncertainty-contrastive-v1"

    def __init__(self, config: GaussianContrastiveLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or GaussianContrastiveLossConfig()

    def curriculum(self, epoch: int) -> tuple[float, float]:
        progress = min(max(float(epoch) / self.config.total_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        positive_lambda = self.config.positive_lambda_min + (1.0 - self.config.positive_lambda_min) * cosine
        negative_temperature = self.config.negative_temperature_low + (
            self.config.negative_temperature_high - self.config.negative_temperature_low
        ) * cosine
        return positive_lambda, negative_temperature

    def forward(
        self,
        *,
        output: Any,
        labels: torch.Tensor,
        original_ids: torch.Tensor | Sequence[Any],
        epoch: int,
        anchor_mask: torch.Tensor | None = None,
    ) -> GaussianLossOutput:
        # .float() alone does not prevent autocast from lowering matmul precision.
        # Small excesses above the entropy floor need float32 cosine/logsumexp.
        if not hasattr(output, "video_mean"):
            raise TypeError("Model output must define video_mean")
        with torch.autocast(device_type=output.video_mean.device.type, enabled=False):
            return self._forward_float32(output=output, labels=labels, original_ids=original_ids, epoch=epoch, anchor_mask=anchor_mask)

    def _forward_float32(
        self,
        *,
        output: Any,
        labels: torch.Tensor,
        original_ids: torch.Tensor | Sequence[Any],
        epoch: int,
        anchor_mask: torch.Tensor | None = None,
    ) -> GaussianLossOutput:
        required = ("temporal_mean", "temporal_variance", "video_mean", "video_variance", "temporal_valid_mask")
        if any(not hasattr(output, name) for name in required):
            raise TypeError(f"Model output must define {required}")
        mean_t = output.temporal_mean.float()
        variance_t = output.temporal_variance.float()
        mean = output.video_mean.float()
        variance = output.video_variance.float()
        valid_time = output.temporal_valid_mask.to(device=mean.device, dtype=torch.bool)
        labels = labels.to(device=mean.device, dtype=torch.long)
        identities = _identity_tensor(original_ids, mean.device)
        batch = mean.shape[0]
        if labels.shape != (batch,) or identities.shape != (batch,):
            raise ValueError("labels and original_ids must have one value per representation")
        if mean_t.shape != variance_t.shape or mean.shape != variance.shape:
            raise ValueError("Gaussian mean/variance shapes must match")
        if mean_t.shape[:2] != valid_time.shape or mean_t.shape[0] != batch:
            raise ValueError("Temporal Gaussian tensors and validity mask do not align")
        if mean_t.ndim != 3 or mean.ndim != 2 or mean_t.shape[-1] != mean.shape[-1]:
            raise ValueError("Expected temporal [B,T,D] and video [B,D] Gaussian tensors")
        if torch.any(~valid_time.any(dim=1)):
            raise ValueError("Every video must contain at least one valid temporal position")
        if torch.any(variance_t <= 0) or torch.any(variance <= 0):
            raise ValueError("All predicted variances must be positive")

        normalized = F.normalize(mean, dim=-1)
        logits = normalized @ normalized.transpose(0, 1) / self.config.temperature
        self_mask = torch.eye(batch, device=mean.device, dtype=torch.bool)
        same_class = labels[:, None].eq(labels[None, :])
        positives = same_class & ~self_mask
        negatives = ~same_class
        valid_anchor = positives.any(dim=1) & negatives.any(dim=1)
        if not valid_anchor.any():
            raise ValueError("Contrastive batch has no anchor with both a positive and a negative")
        if anchor_mask is not None:
            if anchor_mask.shape != (batch,):
                raise ValueError("anchor_mask must have one value per representation")
            valid_anchor = valid_anchor & anchor_mask.to(device=mean.device, dtype=torch.bool)
        valid_count = int(valid_anchor.sum().item())

        delta = ((mean[:, None] - mean[None, :]).square() / (
            variance[:, None] + variance[None, :] + self.config.epsilon
        )).mean(dim=-1).detach()
        positive_lambda, negative_temperature = self.curriculum(epoch)
        compatibility = 1.0 / (1.0 + delta / self.config.compatibility_scale)
        # Algebraically identical, but avoids cancellation to exact zero when
        # lambda=1 and compatibility is smaller than float32 machine epsilon.
        positive_weights = (1.0 - positive_lambda) + positive_lambda * compatibility
        positive_weights = torch.where(identities[:, None].eq(identities[None, :]), 1.0, positive_weights)
        positive_weights = positive_weights * positives

        negative_scores = (-delta / negative_temperature).masked_fill(~negatives, -torch.inf)
        negative_probabilities = torch.softmax(negative_scores, dim=1)
        negative_counts = negatives.sum(dim=1, keepdim=True)
        negative_weights = negative_probabilities * negative_counts
        negative_weights = torch.where(negatives, negative_weights, torch.zeros_like(negative_weights))
        if not self.config.pair_weighting:
            positive_weights = positives.to(logits.dtype)
            negative_weights = negatives.to(logits.dtype)

        positive_terms = logits.masked_fill(~positives, -torch.inf)
        negative_terms = (logits + negative_weights.clamp_min(torch.finfo(logits.dtype).tiny).log()).masked_fill(
            ~negatives, -torch.inf
        )
        log_partition = torch.logsumexp(torch.cat((positive_terms, negative_terms), dim=1), dim=1)
        log_probability = logits - log_partition[:, None]
        positive_mass = positive_weights.sum(dim=1)
        anchor_loss = -(positive_weights * torch.where(positives, log_probability, torch.zeros_like(log_probability))).sum(dim=1)
        # epsilon stabilizes the Gaussian distance, not the total target mass.
        # Clamping small positive mass to epsilon silently removes supervision.
        anchor_loss = anchor_loss / positive_mass.clamp_min(torch.finfo(logits.dtype).tiny)
        contrastive_sum = anchor_loss[valid_anchor].sum()

        # The weighted positive cross-entropy has a non-zero irreducible term
        # whenever an anchor has multiple positives.  Reporting its entropy
        # separately makes loss curves comparable across batch/class sizes;
        # it does not change the optimized objective.
        positive_distribution = positive_weights / positive_mass.unsqueeze(1).clamp_min(torch.finfo(logits.dtype).tiny)
        positive_log_distribution = torch.where(
            positives,
            positive_distribution.clamp_min(torch.finfo(positive_distribution.dtype).tiny).log(),
            torch.zeros_like(positive_distribution),
        )
        positive_entropy = -(positive_distribution * positive_log_distribution).sum(dim=1)
        positive_entropy_sum = positive_entropy[valid_anchor].sum()

        # main.pdf section 8: detach the complete residual so L_unc trains only variance.
        residual = (mean_t - mean.detach().unsqueeze(1)).detach()
        uncertainty_elements = residual.square() / variance_t + variance_t.log()
        time_weights = valid_time.unsqueeze(-1).to(uncertainty_elements.dtype)
        per_sample_uncertainty = (uncertainty_elements * time_weights).sum(dim=(1, 2))
        per_sample_uncertainty = per_sample_uncertainty / (
            valid_time.sum(dim=1).clamp_min(1).to(uncertainty_elements.dtype) * mean_t.shape[-1]
        )
        uncertainty_sum = per_sample_uncertainty[valid_anchor].sum()
        loss_sum = contrastive_sum + self.config.uncertainty_weight * uncertainty_sum
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Gaussian contrastive loss is not finite")
        return GaussianLossOutput(
            loss_sum=loss_sum,
            sample_count=valid_count,
            contrastive_sum=contrastive_sum.detach(),
            positive_entropy_sum=positive_entropy_sum.detach(),
            uncertainty_sum=uncertainty_sum.detach(),
            positive_weight_sum=positive_weights[valid_anchor].sum().detach(),
            negative_weight_sum=negative_weights[valid_anchor].sum().detach(),
            positive_lambda=positive_lambda,
            negative_temperature=negative_temperature,
        )
