"""Causal trajectory adapter with a separate ViCo Eq. 10 semantic space."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .vico import vico_pairwise_contrastive


def standardized(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],))


class TrajectoryModel(nn.Module):
    """Inputs must be frozen, past-only prefix features, never full-video tokens.

    Shared states retain backbone width. Only whole-episode semantic embeddings
    are class clustered; intermediate states are not assigned episode labels.
    """

    def __init__(self, width: int, hidden: int = 128, embedding_dim: int = 128):
        super().__init__()
        self.down = nn.Linear(width, hidden)
        self.clock = nn.Linear(1, hidden, bias=False)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0,
                                            batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.up = nn.Linear(hidden, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.event_score = nn.Linear(width, 1)
        nn.init.zeros_(self.event_score.weight)
        nn.init.zeros_(self.event_score.bias)
        self.semantic = nn.Sequential(nn.Linear(width, 64), nn.GELU(),
                                      nn.Linear(64, embedding_dim))
        self.dynamics = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
        self.future_delta = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(),
                                         nn.Linear(hidden, width))
        nn.init.zeros_(self.future_delta[-1].weight)
        nn.init.zeros_(self.future_delta[-1].bias)

    def forward(self, teacher, valid, timestamps, *, detach_teacher: bool = True):
        """Detach frozen targets in training; allow feature gradients for attribution."""
        if teacher.ndim != 3 or valid.shape != teacher.shape[:2] or timestamps.shape != valid.shape:
            raise ValueError("Expected features [B,T,D], valid/timestamps [B,T]")
        if not valid.any(1).all() or not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("Every trajectory needs a nonempty contiguous valid prefix")
        # Sanitize padding before any linear layer or normalization.
        base = (teacher.detach() if detach_teacher else teacher).masked_fill(~valid[..., None], 0)
        clock = timestamps.masked_fill(~valid, 0)
        base_norm = standardized(base)
        token = self.down(base_norm) + self.clock(clock[..., None])
        causal = torch.ones(valid.shape[1], valid.shape[1], device=teacher.device,
                            dtype=torch.bool).triu(1)
        adapted = self.temporal(token, mask=causal, src_key_padding_mask=~valid)
        states = (base + self.up(adapted)).masked_fill(~valid[..., None], 0)
        normalized = standardized(states)
        score = self.event_score(normalized).squeeze(-1).masked_fill(~valid, -torch.inf)
        attention = score.softmax(1)
        mean = normalized.sum(1) / valid.sum(1, keepdim=True)
        # A mean component prevents the event attention from ignoring all context.
        pooled = 0.5 * mean + 0.5 * (attention[..., None] * normalized).sum(1)
        return {"states": states, "dynamics": self.dynamics(normalized),
                "embedding": F.normalize(self.semantic(pooled), dim=-1)}


def trajectory_loss(model, teacher, valid, timestamps, labels, *, epoch: int,
                    curriculum_epochs: int = 20, align_weight: float = 0.1,
                    predict_weight: float = 1.0, keep_weight: float = 0.1,
                    horizons: tuple[int, ...] = (1, 2, 4)):
    """Two matched views [N,2,T,D]; per-term means do not depend on padding.

    Predict normalized frozen teacher changes over approximately .5/1/2 seconds.
    Timestamp deltas, rather than step numbers, condition the predictor.
    """
    if teacher.ndim != 4 or teacher.shape[1] != 2:
        raise ValueError("Two synchronized views [N,2,T,D] are required")
    if any(w < 0 for w in (align_weight, predict_weight, keep_weight)):
        raise ValueError("Loss weights must be nonnegative")
    if not horizons or any(k < 1 for k in horizons):
        raise ValueError("Prediction horizons must be positive")
    n, views, time, width = teacher.shape
    mask = valid[:, None].expand(n, views, time).reshape(n * views, time)
    stamps = timestamps[:, None].expand(n, views, time).reshape(n * views, time)
    frozen = teacher.detach().reshape(n * views, time, width)
    output = model(frozen, mask, stamps)
    return trajectory_loss_from_output(model, teacher, valid, timestamps, labels, output,
        epoch=epoch, curriculum_epochs=curriculum_epochs, align_weight=align_weight,
        predict_weight=predict_weight, keep_weight=keep_weight, horizons=horizons)


def trajectory_loss_from_output(model, teacher, valid, timestamps, labels, output, *, epoch: int,
                                curriculum_epochs: int = 20, align_weight: float = 0.1,
                                predict_weight: float = 1.0, keep_weight: float = 0.1,
                                horizons: tuple[int, ...] = (1, 2, 4)):
    """Apply the same full-batch objective to gathered distributed model outputs."""
    if teacher.ndim != 4 or teacher.shape[1] != 2:
        raise ValueError("Two synchronized views [N,2,T,D] are required")
    if any(w < 0 for w in (align_weight, predict_weight, keep_weight)):
        raise ValueError("Loss weights must be nonnegative")
    if not horizons or any(k < 1 for k in horizons):
        raise ValueError("Prediction horizons must be positive")
    n, views, time, width = teacher.shape
    mask = valid[:, None].expand(n, views, time).reshape(n * views, time)
    stamps = timestamps[:, None].expand(n, views, time).reshape(n * views, time)
    frozen = teacher.detach().reshape(n * views, time, width)
    semantic, details = vico_pairwise_contrastive(
        output["embedding"], labels.repeat_interleave(2),
        source_ids=torch.arange(n, device=labels.device).repeat_interleave(2),
        epoch=epoch, curriculum_epochs=curriculum_epochs)
    dynamics = output["dynamics"].reshape(n, views, time, -1)
    alignment = F.smooth_l1_loss(dynamics[:, 0][valid], dynamics[:, 1][valid])
    target = standardized(frozen.masked_fill(~mask[..., None], 0)).detach()
    state = standardized(output["states"])
    retention = F.smooth_l1_loss(state[mask], target[mask])
    errors, persistence, count = [], [], 0
    for k in horizons:
        if k >= time:
            continue
        pairs = mask[:, :-k] & mask[:, k:]
        if not pairs.any():
            continue
        delta_t = (stamps[:, k:] - stamps[:, :-k])[pairs].unsqueeze(-1)
        context = output["dynamics"][:, :-k][pairs]
        predicted = state[:, :-k][pairs] + model.future_delta(torch.cat((context, delta_t), -1))
        future = target[:, k:][pairs]
        errors.append(F.smooth_l1_loss(predicted, future, reduction="sum"))
        persistence.append(F.smooth_l1_loss(target[:, :-k][pairs], future, reduction="sum"))
        count += int(pairs.sum()) * width
    if not count:
        raise ValueError("No valid future prediction pairs")
    prediction = torch.stack(errors).sum() / count
    baseline = torch.stack(persistence).sum() / count
    total = semantic + align_weight * alignment + predict_weight * prediction + keep_weight * retention
    details.update(total_loss=total, semantic_loss=semantic, alignment_loss=alignment,
                   prediction_loss=prediction, retention_loss=retention,
                   persistence_loss=baseline, prediction_to_persistence=prediction / baseline.clamp_min(1e-12))
    return total, details, output


class DistributedTrajectoryStep(nn.Module):
    """Full global ViCo and temporal objective from source-sharded DDP forwards.

    Every rank evaluates the same global loss. Differentiable all-gather sums
    output gradients in its backward; DDP then averages parameter gradients.
    """

    def __init__(self, model: TrajectoryModel):
        super().__init__()
        self.model = model

    def forward(self, teacher, valid, timestamps, labels, *, epoch: int,
                curriculum_epochs: int = 20, align_weight: float = .1,
                predict_weight: float = 1., keep_weight: float = .1,
                microbatch_sources: int | None = None):
        weights = dict(epoch=epoch, curriculum_epochs=curriculum_epochs,
                       align_weight=align_weight, predict_weight=predict_weight,
                       keep_weight=keep_weight)
        if microbatch_sources is not None and microbatch_sources < 1:
            raise ValueError("microbatch_sources must be positive")
        if microbatch_sources is None and (not dist.is_available() or not dist.is_initialized()
                                           or dist.get_world_size() == 1):
            return trajectory_loss(self.model, teacher, valid, timestamps, labels, **weights)
        n, views, time, width = teacher.shape
        if views != 2 or n == 0:
            raise ValueError("Each distributed rank needs source videos with two views")
        mask = valid[:, None].expand(n, views, time).reshape(n * views, time)
        stamps = timestamps[:, None].expand(n, views, time).reshape(n * views, time)
        source_chunk = microbatch_sources or n
        chunks = []
        for start in range(0, n, source_chunk):
            stop = min(start + source_chunk, n)
            chunks.append(self.model(
                teacher[start:stop].detach().reshape((stop - start) * views, time, width),
                mask[start * views:stop * views], stamps[start * views:stop * views]))
        local = {key: torch.cat([part[key] for part in chunks], dim=0) for key in chunks[0]}
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return trajectory_loss_from_output(self.model, teacher, valid, timestamps,
                                               labels, local, **weights)
        from torch.distributed.nn.functional import all_gather as differentiable_gather
        local_count = torch.tensor([n], device=teacher.device, dtype=torch.long)
        rank_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(rank_counts, local_count)
        counts = [int(value.item()) for value in rank_counts]
        maximum = max(counts)

        def gather_output(value):
            padding = value.new_zeros((2 * (maximum - n), *value.shape[1:]))
            pieces = differentiable_gather(torch.cat((value, padding), dim=0))
            return torch.cat([piece[:2 * count] for piece, count in zip(pieces, counts)], dim=0)

        gathered_output = {key: gather_output(value) for key, value in local.items()}

        def gather_data(value):
            padding = value.new_zeros((maximum - n, *value.shape[1:]))
            padded = torch.cat((value, padding), dim=0)
            pieces = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
            dist.all_gather(pieces, padded)
            return torch.cat([piece[:count] for piece, count in zip(pieces, counts)], dim=0)

        global_teacher = gather_data(teacher.detach())
        global_valid = gather_data(valid)
        global_timestamps = gather_data(timestamps)
        global_labels = gather_data(labels)
        return trajectory_loss_from_output(self.model, global_teacher, global_valid,
                                           global_timestamps, global_labels,
                                           gathered_output, **weights)
