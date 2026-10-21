"""ViCo-LWIR Eq. 10: each positive competes with the anchor's negatives.

This is deliberately distinct from shared-partition SupCon (main.pdf Eq. 2).
Cosine is divided by temperature once. Eq. 7's sqrt(negative count) factor
is used, rather than the inconsistent unit-mass pseudocode in Algorithm 1.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def vico_pair_weights(
    cosine: torch.Tensor, labels: torch.Tensor, source_ids: torch.Tensor, *,
    epoch: int, curriculum_epochs: int, temperature: float, epsilon: float,
    positive_weighting: bool = True, negative_weighting: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached curriculum weights and log negative weights, Eq. 5–8.

    Average ranks resolve ties without making weights depend on sample order.
    Self pairs participate in the paper's rowwise evidence/rank computation,
    but never participate in the contrastive numerator or denominator.
    """
    with torch.no_grad():
        # Eq. 2 uses cosine evidence, not another division by temperature.
        cohesion = cosine.softmax(1).exp()
        noncohesion = (-cosine).softmax(1).exp()
        uncertainty = 2.0 / (cohesion + noncohesion + 2.0)
        greater = (uncertainty[:, None, :] > uncertainty[:, :, None]).sum(-1)
        equal = (uncertainty[:, None, :] == uncertainty[:, :, None]).sum(-1)
        rank_fraction = (greater.to(cosine.dtype) + (equal.to(cosine.dtype) - 1) / 2) / (len(labels) - 1)
        progress = min(max(epoch / curriculum_epochs, 0.0), 1.0)
        positive_weights = 1 + torch.tanh(progress * rank_fraction)
        same_source = source_ids[:, None].eq(source_ids[None, :])
        positive_weights = torch.where(same_source, 1.0, positive_weights)
        if not positive_weighting:
            positive_weights = torch.ones_like(cosine)

        negative = labels[:, None].ne(labels[None, :])
        counts = negative.sum(1).to(cosine.dtype)
        logits = cosine / temperature
        scores = logits.masked_fill(~negative, -torch.inf)
        log_sum = torch.logsumexp(scores, 1)
        log_denominator = torch.logaddexp(log_sum, logits.new_full(log_sum.shape, math.log(epsilon)))
        if negative_weighting:
            log_negative_weights = scores - log_denominator[:, None] + 0.5 * counts.log()[:, None]
        else:
            # Same sqrt(N) total mass as Eq. 7, with uniform negative selection.
            log_negative_weights = (-0.5 * counts.log())[:, None].expand_as(cosine)
        log_negative_weights = log_negative_weights.masked_fill(~negative, -torch.inf)
    return positive_weights, log_negative_weights, uncertainty


def vico_pairwise_contrastive(
    embeddings: torch.Tensor, labels: torch.Tensor, *, source_ids: torch.Tensor | None = None,
    temperature: float = 0.1, epsilon: float = 1e-6, epoch: int = 0,
    curriculum_epochs: int = 100, positive_weighting: bool = True,
    negative_weighting: bool = True,
    fixed_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stable, unshifted Eq. 10 with genuine positive-vs-negative competition.

    pair_loss_ij = softplus(log(sum_k alpha_ik exp(s_ik) + epsilon)
                           - (log(w_ij) + s_ij)).

    Average positive log likelihoods OUTSIDE the log, as in the PDF. Positive
    weights modify pair logits; they do not normalize an external target q.
    Mining weights are treated as selection statistics, not trainable shortcuts.
    Fixed weights make that stop-gradient contract independently testable.
    """
    if embeddings.ndim != 2 or labels.shape != (len(embeddings),) or len(embeddings) < 3:
        raise ValueError("Expected embeddings [N,D] and labels [N]")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Temperature and epsilon must be finite and positive")
    if curriculum_epochs < 1:
        raise ValueError("Curriculum epochs must be positive")
    labels = labels.to(embeddings.device)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("ViCo video learning requires spill/no-spill labels 0 and 1")
    source_ids = (torch.arange(len(labels), device=labels.device) if source_ids is None
                  else source_ids.to(labels.device))
    if source_ids.shape != labels.shape:
        raise ValueError("Source IDs must have one value per embedding")
    same = labels[:, None].eq(labels[None, :])
    if (source_ids[:, None].eq(source_ids[None, :]) & ~same).any():
        raise ValueError("Views from one source video cannot have conflicting labels")
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError("Every anchor needs a positive and a negative")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        # Small genuine losses remain visible instead of subtracting nearly equal
        # cross-entropy and entropy scalars in float32.
        z = F.normalize(embeddings.double(), dim=-1)
        if not torch.isfinite(z).all() or (embeddings.norm(dim=-1) == 0).any():
            raise ValueError("Embeddings must be finite and nonzero")
        cosine = z @ z.T
        weights, log_alpha, uncertainty = vico_pair_weights(
            cosine, labels, source_ids, epoch=epoch, curriculum_epochs=curriculum_epochs,
            temperature=temperature, epsilon=epsilon, positive_weighting=positive_weighting,
            negative_weighting=negative_weighting)
        if fixed_weights is not None:
            weights, log_alpha = (v.detach().to(cosine) for v in fixed_weights)
            if weights.shape != cosine.shape or log_alpha.shape != cosine.shape:
                raise ValueError("Fixed pair weights must have shape [N,N]")
            if not torch.isfinite(weights[positive]).all() or (weights[positive] <= 0).any():
                raise ValueError("Positive weights must be finite and strictly positive")
            if not torch.isfinite(log_alpha[negative]).all():
                raise ValueError("Log negative weights must be finite on negatives")
        logits = cosine / temperature
        log_negative_mass = torch.logsumexp((logits + log_alpha).masked_fill(~negative, -torch.inf), 1)
        log_competition = torch.logaddexp(log_negative_mass, logits.new_full(log_negative_mass.shape, math.log(epsilon)))
        log_positive_mass = logits + weights.log()
        pair_losses = F.softplus(log_competition[:, None] - log_positive_mass).masked_fill(~positive, 0)
        per_anchor = pair_losses.sum(1) / positive.sum(1)
        positive_min = cosine.masked_fill(~positive, torch.inf).min(1).values
        negative_max = cosine.masked_fill(~negative, -torch.inf).max(1).values
        max_pair = pair_losses[positive].max()
        # Equal-weight/antipodal diagnostic has no log(P) term. Epsilon is kept
        # exactly as written in Eq. 10. Actual curriculum weights can lower it.
        counts = negative.sum(1).to(cosine.dtype)
        negative_exp = math.exp(-1 / temperature)
        alpha_mass = counts.sqrt() * (counts * negative_exp) / (counts * negative_exp + epsilon)
        reference = torch.log1p(alpha_mass * math.exp(-2 / temperature)
                               + epsilon * negative_exp).mean()
        return per_anchor.mean(), {
            "vico_loss": per_anchor.mean(), "max_anchor_loss": per_anchor.max(),
            "max_positive_pair_loss": max_pair,
            "minimum_pair_separation": (positive_min - negative_max).min(),
            "minimum_positive_cosine": positive_min.min(), "maximum_negative_cosine": negative_max.max(),
            "mean_positive_cosine": cosine[positive].mean(), "mean_negative_cosine": cosine[negative].mean(),
            "mean_negative_win_probability": (-torch.expm1(-pair_losses[positive])).mean(),
            "uncertainty_min": uncertainty.min(), "uncertainty_max": uncertainty.max(),
            "positive_weight_min": weights[positive].min(), "positive_weight_max": weights[positive].max(),
            "equal_weight_antipodal_reference": reference,
        }
