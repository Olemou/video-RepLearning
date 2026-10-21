"""Supervised spill/no-spill representation learning from frozen V-JEPA videos."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from .vico import vico_pairwise_contrastive


def masked_temporal_mean(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or valid.shape != features.shape[:2]:
        raise ValueError("Expected features [B,T,D] and valid [B,T]")
    valid = valid.to(device=features.device, dtype=torch.bool)
    if not valid.any(1).all():
        raise ValueError("Every video view must contain a valid tubelet")
    return features.masked_fill(~valid[..., None], 0).sum(1) / valid.sum(1, keepdim=True)


def temporal_views(valid: torch.Tensor, *, keep_fraction: float, generator: torch.Generator) -> torch.Tensor:
    """Draw two independent valid-tubelet views per video, on the input device."""
    if valid.ndim != 2 or not 0 < keep_fraction <= 1:
        raise ValueError("Expected valid [B,T] and keep_fraction in (0,1]")
    result = torch.zeros((len(valid), 2, valid.shape[1]), dtype=torch.bool, device=valid.device)
    for row in range(len(valid)):
        indices = torch.where(valid[row])[0]
        if len(indices) == 0:
            raise ValueError("Every video must contain a valid tubelet")
        count = min(len(indices), max(1, math.ceil(len(indices) * keep_fraction)))
        for view in range(2):
            order = torch.randperm(len(indices), generator=generator, device="cpu").to(indices.device)
            result[row, view, indices[order[:count]]] = True
    return result


class ResidualContrastiveHead(nn.Module):
    """Learn a low-rank change to normalized pretrained video embeddings."""

    def __init__(self, width: int, bottleneck: int = 64) -> None:
        super().__init__()
        if width <= 0 or bottleneck <= 0:
            raise ValueError("width and bottleneck must be positive")
        self.down = nn.Linear(width, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        raw = F.normalize(pooled.float(), dim=-1)
        delta = self.up(F.gelu(self.down(raw)))
        return F.normalize(raw + delta, dim=-1)


class NonlinearContrastiveHead(nn.Module):
    """Compact MLP projector that need not cancel the frozen identity features."""

    def __init__(self, width: int, bottleneck: int = 64, embedding_dim: int = 128) -> None:
        super().__init__()
        if min(width, bottleneck, embedding_dim) <= 0:
            raise ValueError("Head dimensions must be positive")
        self.network = nn.Sequential(nn.Linear(width, bottleneck), nn.GELU(),
                                     nn.Linear(bottleneck, embedding_dim))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(F.normalize(pooled.float(), dim=-1)), dim=-1)


def supervised_contrastive_excess(
    embeddings: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return SupCon cross-entropy, its positive entropy, and their difference."""
    if embeddings.ndim != 2 or labels.shape != (len(embeddings),) or temperature <= 0:
        raise ValueError("Expected embeddings [N,D], labels [N], and positive temperature")
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), device=embeddings.device, dtype=torch.bool)
    positives = same & ~eye
    if not positives.any(1).all() or not (~same).any(1).all():
        raise ValueError("Every anchor needs a positive and a negative")
    z = F.normalize(embeddings if embeddings.dtype == torch.float64 else embeddings.float(), dim=-1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positives.sum(1)
    cross_entropy = -(log_prob.masked_fill(~positives, 0).sum(1) / count).mean()
    entropy = count.to(z.dtype).log().mean()
    return cross_entropy, entropy, cross_entropy - entropy


class VideoRepresentationModel(nn.Module):
    """First-stage video representation and cosine prototype classifier.

    Input is frozen V-JEPA tubelets, not pixels. Prototypes are fitted only
    from training videos; similarity scores are not calibrated probabilities.
    """

    def __init__(self, width: int, bottleneck: int = 64, *, head_type: str = "residual",
                 embedding_dim: int = 128) -> None:
        super().__init__()
        if head_type == "residual":
            self.head = ResidualContrastiveHead(width, bottleneck)
            output_width = width
        elif head_type == "mlp":
            self.head = NonlinearContrastiveHead(width, bottleneck, embedding_dim)
            output_width = embedding_dim
        else:
            raise ValueError("Unknown video head type")
        self.register_buffer("prototypes", torch.zeros(2, output_width))
        self.register_buffer("prototypes_fitted", torch.tensor(False))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.head(masked_temporal_mean(features, valid))

    @torch.no_grad()
    def fit_prototypes(self, features: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
            raise ValueError("Prototype fitting requires both binary video classes")
        z = self(features, valid)
        self.prototypes.copy_(F.normalize(torch.stack([z[labels == c].mean(0) for c in (0, 1)]), dim=-1))
        self.prototypes_fitted.fill_(True)

    @torch.no_grad()
    def predict(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.prototypes_fitted.item():
            raise ValueError("Fit training prototypes before prediction")
        embeddings = self(features, valid)
        scores = embeddings @ self.prototypes.T
        return {"embeddings": embeddings, "cosine_scores": scores, "labels": scores.argmax(-1)}


def fit_video_representation(
    features: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor, *,
    epochs: int = 2000, seed: int = 239, bottleneck: int = 64,
    learning_rate: float = 1e-3, temperature: float = 0.1, tether: float = 0.0,
    convergence_tolerance: float = 1e-5, convergence_patience: int = 5,
    objective: str = "supcon", positive_margin: float = 0.9, negative_margin: float = 0.1,
    weight_decay: float = 0.0, augmented_features: torch.Tensor | None = None,
    head_type: str = "residual", embedding_dim: int = 128, curriculum_epochs: int = 100,
    positive_weighting: bool = True, negative_weighting: bool = True,
    adam_epsilon: float = 1e-8,
    reset_optimizer_after_curriculum: bool = True,
) -> tuple[VideoRepresentationModel, list[dict[str, float | int | bool]]]:
    """Learn spill/no-spill geometry from distinct, complete training videos.

    Full-batch learning gives every anchor distinct same-class videos and
    opposite-class negatives. Optional mirrored features preserve temporal
    coverage. ViCo Eq. 10 gives each positive its own negative partition;
    its same-source positive views receive weight one. Shared-partition
    SupCon and margin objectives remain explicit comparison options.
    """
    if epochs < 1 or learning_rate <= 0 or temperature <= 0 or tether < 0:
        raise ValueError("Invalid training hyperparameters")
    if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Training requires binary video labels 0 and 1")
    if min(int((labels == c).sum()) for c in (0, 1)) < 2:
        raise ValueError("Need at least two distinct training videos per class")
    if convergence_tolerance <= 0 or convergence_patience < 1:
        raise ValueError("Convergence tolerance and patience must be positive")
    if weight_decay < 0:
        raise ValueError("Weight decay must be nonnegative")
    if not math.isfinite(adam_epsilon) or adam_epsilon <= 0:
        raise ValueError("Adam epsilon must be finite and positive")
    if objective not in {"supcon", "margin", "vico"}:
        raise ValueError("Unknown representation objective")
    # Cached encoder features are immutable training inputs.
    features = features.detach().float()
    valid = valid.to(features.device, dtype=torch.bool)
    labels = labels.to(features.device)
    pooled = masked_temporal_mean(features, valid)
    training_labels = labels
    source_ids = torch.arange(len(labels), device=features.device)
    if augmented_features is not None:
        if augmented_features.shape != features.shape:
            raise ValueError("Augmented features must match complete-video features")
        augmented_pooled = masked_temporal_mean(augmented_features.detach().to(features.device).float(), valid)
        pooled = torch.cat((pooled, augmented_pooled))
        training_labels = labels.repeat(2)
        source_ids = source_ids.repeat(2)
    raw = F.normalize(pooled, dim=-1)
    if not torch.isfinite(pooled).all():
        raise ValueError("Video features must be finite")
    torch.manual_seed(seed)
    model = VideoRepresentationModel(features.shape[-1], bottleneck, head_type=head_type,
                                     embedding_dim=embedding_dim).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                  eps=adam_epsilon)
    history = []
    converged_epochs = 0
    model.train()
    for epoch in range(epochs):
        if objective == "vico" and reset_optimizer_after_curriculum and epoch == curriculum_epochs:
            # The curriculum is now fixed. Discard second moments dominated by
            # the much larger initial losses before refining tiny genuine losses.
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                           weight_decay=weight_decay, eps=adam_epsilon)
        z = model.head(pooled)
        if objective == "supcon":
            _, _, excess = supervised_contrastive_excess(z.double(), training_labels, temperature=temperature)
        if head_type == "mlp" and tether != 0:
            raise ValueError("A pretrained-feature tether requires a residual head")
        proximal = (z - raw).square().sum(-1).mean() if head_type == "residual" else z.new_zeros(())
        if objective == "vico":
            task_loss, _ = vico_pairwise_contrastive(
                z, training_labels, source_ids=source_ids, temperature=temperature,
                epoch=epoch, curriculum_epochs=curriculum_epochs,
                positive_weighting=positive_weighting, negative_weighting=negative_weighting)
        elif objective == "margin":
            task_loss, _ = supervised_margin_contrastive(z, training_labels, positive_margin=positive_margin,
                                                         negative_margin=negative_margin, source_ids=source_ids)
        else:
            task_loss = excess
        loss = task_loss + tether * proximal
        if not torch.isfinite(loss):
            raise ValueError("Non-finite representation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            updated = model.head(pooled)
            diagnostics = representation_diagnostics(updated, training_labels, temperature=temperature)
            proximal_after = (float((updated - raw).square().sum(-1).mean())
                              if head_type == "residual" else 0.0)
            if objective == "vico":
                diagnostics = {"supcon_" + key: value for key, value in diagnostics.items()}
                current_loss, vico_metrics = vico_pairwise_contrastive(
                    updated, training_labels, source_ids=source_ids, temperature=temperature,
                    epoch=epoch + 1, curriculum_epochs=curriculum_epochs,
                    positive_weighting=positive_weighting, negative_weighting=negative_weighting)
                diagnostics.update({key: float(value) for key, value in vico_metrics.items()})
                task_loss_after = float(current_loss)
                diagnostics["contrastive_loss"] = task_loss_after
                stopping_error = diagnostics["max_positive_pair_loss"]
            elif objective == "margin":
                margin_loss, margin_metrics = supervised_margin_contrastive(
                    updated, training_labels, positive_margin=positive_margin, negative_margin=negative_margin,
                    source_ids=source_ids)
                diagnostics.update({key: float(value) for key, value in margin_metrics.items()})
                task_loss_after = float(margin_loss)
                stopping_error = diagnostics["max_margin_violation"]
            else:
                task_loss_after = diagnostics["contrastive_excess"]
                stopping_error = diagnostics["max_anchor_excess"]
        schedule_finished = objective != "vico" or epoch + 1 >= curriculum_epochs
        converged_epochs = (converged_epochs + 1 if
                            schedule_finished and stopping_error <= convergence_tolerance
                            and diagnostics["minimum_pair_separation"] > 0 else 0)
        history.append({"epoch": epoch + 1,
                        "optimizer_resets": int(objective == "vico" and reset_optimizer_after_curriculum
                                                 and epoch >= curriculum_epochs),
                        "loss": task_loss_after + tether * proximal_after,
                        "objective_loss": task_loss_after,
                        "tether_loss": proximal_after, **diagnostics,
                        "converged": converged_epochs >= convergence_patience})
        if converged_epochs >= convergence_patience:
            break
    model.eval()
    model.fit_prototypes(features, valid, labels)
    return model, history


@torch.no_grad()
def representation_diagnostics(embeddings: torch.Tensor, labels: torch.Tensor,
                               *, temperature: float = 0.1) -> dict[str, float]:
    """Decompose SupCon excess into negative leakage and positive nonuniformity."""
    # Float64 avoids mistaking float32 rounding at the entropy floor for convergence.
    ce, entropy, excess = supervised_contrastive_excess(embeddings.double(), labels, temperature=temperature)
    z = F.normalize(embeddings.double(), dim=-1)
    similarity = z @ z.T
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    negative = labels[:, None].ne(labels[None, :])
    logits = (similarity / temperature).masked_fill(eye, -torch.inf)
    log_partition = torch.logsumexp(logits, dim=1)
    log_positive_partition = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    anchor_leakage = log_partition - log_positive_partition
    leakage = anchor_leakage.mean()
    conditional_log_prob = logits - log_positive_partition[:, None]
    anchor_uniformity = (-(conditional_log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1))
                         - positive.sum(1).double().log())
    uniformity = anchor_uniformity.mean()
    pair_separation = (similarity.masked_fill(~positive, torch.inf).min(1).values
                       - similarity.masked_fill(~negative, -torch.inf).max(1).values)
    # Binary unit-vector optimum: each class collapses to one point and the
    # two class points are antipodal. Finite temperature leaves a tiny excess.
    finite_excess = torch.log1p(negative.sum(1) / positive.sum(1)
                              * math.exp(-2 / temperature)).mean()
    return {"contrastive_loss": float(ce), "positive_entropy": float(entropy),
            "contrastive_excess": float(excess),
            "max_anchor_excess": float((anchor_leakage + anchor_uniformity).max()),
            "minimum_pair_separation": float(pair_separation.min()),
            "negative_leakage_loss": float(leakage),
            "positive_nonuniformity_loss": float(uniformity),
            "mean_negative_probability": float((1 - (log_positive_partition - log_partition).exp()).mean()),
            "mean_positive_cosine": float(similarity[positive].mean()),
            "mean_negative_cosine": float(similarity[negative].mean()),
            "binary_unit_sphere_minimum_excess": float(finite_excess)}


def supervised_margin_contrastive(
    embeddings: torch.Tensor, labels: torch.Tensor, *, positive_margin: float = 0.9,
    negative_margin: float = 0.1, source_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-balanced binary contrastive loss with explicit cosine requirements.

    Unlike softmax SupCon, zero is attainable at finite logits and means that
    every positive pair is above its target and every negative pair below its
    target. Nonzero within-class variation is permitted by design.
    """
    if embeddings.ndim != 2 or labels.shape != (len(embeddings),):
        raise ValueError("Expected embeddings [N,D] and labels [N]")
    if not -1 < negative_margin < positive_margin < 1:
        raise ValueError("Require -1 < negative_margin < positive_margin < 1")
    if set(labels.tolist()) != {0, 1} or min(int((labels == c).sum()) for c in (0, 1)) < 2:
        raise ValueError("Need two binary classes and at least two videos per class")
    z = F.normalize(embeddings.double(), dim=-1)
    similarity = z @ z.T
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    negative = labels[:, None].ne(labels[None, :])
    if source_ids is not None:
        if source_ids.shape != labels.shape:
            raise ValueError("Source IDs must match labels")
        positive = positive & source_ids[:, None].ne(source_ids[None, :])
    if not positive.any(1).all():
        raise ValueError("Each anchor needs a positive from another video")
    positive_violation = (positive_margin - similarity).clamp_min(0).masked_fill(~positive, 0)
    negative_violation = (similarity - negative_margin).clamp_min(0).masked_fill(~negative, 0)
    per_anchor_positive = positive_violation.sum(1) / positive.sum(1)
    per_anchor_negative = negative_violation.sum(1) / negative.sum(1)
    positive_loss = torch.stack([per_anchor_positive[labels == c].mean() for c in (0, 1)]).mean()
    negative_loss = torch.stack([per_anchor_negative[labels == c].mean() for c in (0, 1)]).mean()
    return positive_loss + negative_loss, {
        "positive_margin_loss": positive_loss, "negative_margin_loss": negative_loss,
        "max_margin_violation": torch.maximum(positive_violation.max(), negative_violation.max()),
        "minimum_positive_cosine": similarity[positive].min(),
        "maximum_negative_cosine": similarity[negative].max(),
    }
