"""Target-specific gradients of causal Transformer attention probabilities.

``positive_gradient`` follows LeGrad's positive-gradient, query/head/layer
averaging rule, adapted to temporal keys. It uses the *final* model scalar;
the original ViT method uses intermediate feature classifiers. Consequently
these results are LeGrad-inspired, not an exact reproduction of that method.
``attention_gradient`` is a separate positive A*dY/dA diagnostic.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
import math
import types

import torch
import torch.nn.functional as F

from vision_model.training.trajectory import TrajectoryModel, standardized


@dataclass(frozen=True)
class AttributionTarget:
    kind: Literal["state", "dynamics", "semantic_similarity", "future_prediction", "future_error"]
    timestep: int | None = None
    component: int | None = None
    reference: torch.Tensor | None = None
    future_timestep: int | None = None


@dataclass
class TemporalAttribution:
    video_id: str
    sample_id: str
    target: dict
    target_value: float
    target_timestamp: float | None
    valid_length: int
    frame_indices: torch.Tensor | None
    timestamps: torch.Tensor
    relevance: torch.Tensor  # [T], CPU
    layer_relevance: torch.Tensor  # [selected L,T], CPU
    head_relevance: torch.Tensor  # [selected L,selected H,T], CPU
    signed_gradients: torch.Tensor | None  # [L,H,T,T], CPU
    raw_attention: torch.Tensor | None  # [L,H,T,T], CPU
    relevance_maps: torch.Tensor | None  # [L,H,T,T], CPU
    event_pooling_weights: torch.Tensor  # [T], CPU, distinct from Transformer A
    layer_indices: tuple[int, ...]
    head_indices: tuple[int, ...]
    method: str
    normalization: str
    checkpoint_id: str
    temporal_context_policy: str
    gradient_statistics: dict | None = None

    def metadata(self) -> dict:
        return dict(video_id=self.video_id, sample_id=self.sample_id,
                    target=self.target, target_value=self.target_value,
                    target_timestamp=self.target_timestamp, valid_length=self.valid_length,
                    frame_indices=None if self.frame_indices is None else self.frame_indices.tolist(),
                    timestamps=self.timestamps.tolist(), relevance=self.relevance.tolist(),
                    layer_relevance=self.layer_relevance.tolist(),
                    head_relevance=self.head_relevance.tolist(),
                    event_pooling_weights=self.event_pooling_weights.tolist(),
                    layer_indices=self.layer_indices, head_indices=self.head_indices,
                    method=self.method, normalization=self.normalization,
                    checkpoint_id=self.checkpoint_id,
                    temporal_context_policy=self.temporal_context_policy,
                    gradient_statistics=self.gradient_statistics,
                    raw_attention_in_tensor_file=self.raw_attention is not None)


@contextmanager
def capture_attention(model: TrajectoryModel, layers: tuple[int, ...]):
    """Temporarily request differentiable per-head MHA probabilities.

    Eval mode has zero attention dropout in this architecture. PyTorch MHA's
    returned per-head weights may be a view created *after* its output BMM,
    so they are not differentiable ancestors of the target. This temporary
    implementation makes the probability tensor the value-BMM input.
    No global hook or persistent model mutation remains after the context.
    """
    modules = [model.temporal.layers[i].self_attn for i in layers]
    originals = [(module.forward, "forward" in module.__dict__) for module in modules]
    captured: dict[int, torch.Tensor] = {}
    try:
        for index, module in zip(layers, modules):
            def forward(_self, query, key, value, *, key_padding_mask=None,
                        need_weights=True, attn_mask=None, average_attn_weights=True,
                        is_causal=False, _index=index):
                if query is not key or query is not value or not _self.batch_first:
                    raise ValueError("Capture supports the trajectory encoder's batch-first self-attention")
                batch, time, width = query.shape
                heads = _self.num_heads
                qkv = F.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                q, k, v = qkv.reshape(batch, time, 3, heads, width // heads).permute(2, 0, 3, 1, 4).unbind(0)
                logits = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
                if attn_mask is not None:
                    mask = attn_mask[None, None] if attn_mask.ndim == 2 else attn_mask.reshape(batch, heads, time, time)
                    logits = logits.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else logits + mask
                elif is_causal:
                    logits = logits.masked_fill(torch.ones(time, time, dtype=torch.bool,
                                                           device=query.device).triu(1), -torch.inf)
                if key_padding_mask is not None:
                    mask = key_padding_mask[:, None, None, :]
                    logits = logits.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else logits + mask
                attention = logits.softmax(-1)
                captured[_index] = attention
                attended = (attention @ v).transpose(1, 2).reshape(batch, time, width)
                output = F.linear(attended, _self.out_proj.weight, _self.out_proj.bias)
                weights = attention if need_weights else None
                if weights is not None and average_attn_weights:
                    weights = weights.mean(1)
                return output, weights
            module.forward = types.MethodType(forward, module)
        yield captured
    finally:
        for module, (original, had_instance_forward) in zip(modules, originals):
            if had_instance_forward:
                module.forward = original
            else:
                del module.forward


def _scalar(model, output, teacher, timestamps, target: AttributionTarget):
    kind, step, component = target.kind, target.timestep, target.component
    if kind == "semantic_similarity":
        if target.reference is None:
            raise ValueError("semantic_similarity requires a reference embedding")
        reference = target.reference.to(device=teacher.device, dtype=teacher.dtype).flatten()
        if reference.numel() != output["embedding"].shape[-1] or not torch.isfinite(reference).all():
            raise ValueError("Reference embedding has wrong size or non-finite values")
        return F.cosine_similarity(output["embedding"][0], reference, dim=0)
    if step is None or step < 0 or step >= output["states"].shape[1]:
        raise ValueError("A valid target timestep is required")
    if kind in ("state", "dynamics"):
        vector = output["states" if kind == "state" else "dynamics"][0, step]
        if component is None or not 0 <= component < vector.numel():
            raise ValueError("A valid component index is required for a state target")
        return vector[component]
    if kind not in ("future_prediction", "future_error"):
        raise ValueError(f"Unsupported target: {kind}")
    future = target.future_timestep
    if future is None or future <= step or future >= teacher.shape[1] or future - step not in (1, 2, 4):
        raise ValueError("Future target needs a valid horizon of 1, 2, or 4 tubelets")
    delta = (timestamps[0, future] - timestamps[0, step]).reshape(1)
    dynamics = output["dynamics"][0, step]
    predicted = standardized(output["states"])[0, step] + model.future_delta(
        torch.cat((dynamics, delta), dim=0))
    if kind == "future_prediction":
        if component is None or not 0 <= component < predicted.numel():
            raise ValueError("future_prediction requires a valid component")
        return predicted[component]
    truth = standardized(teacher.detach())[0, future]
    return F.smooth_l1_loss(predicted, truth, reduction="mean")


def _normalize(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return values
    if mode == "sum":
        return values / values.sum().clamp_min(torch.finfo(values.dtype).eps)
    if mode == "max":
        return values / values.max().clamp_min(torch.finfo(values.dtype).eps)
    raise ValueError("normalization must be none, sum, or max")


def attribute_trajectory(model: TrajectoryModel, features: torch.Tensor, valid: torch.Tensor,
                         timestamps: torch.Tensor, target: AttributionTarget, *,
                         video_id: str, sample_id: str = "identity",
                         frame_indices: torch.Tensor | None = None,
                         checkpoint_id: str = "unspecified",
                         temporal_context_policy: str = "past_only_prefix_last_tubelet_v1",
                         method: str = "positive_gradient", normalization: str = "sum",
                         layers: tuple[int, ...] | None = None,
                         heads: tuple[int, ...] | None = None,
                         include_maps: bool = False,
                         max_attention_elements: int = 8_000_000) -> TemporalAttribution:
    """Attribute one sample; call repeatedly for batches to bound GPU memory.

    Feature columns are causal prefix summaries, never full-video features.
    Timestep targets run on the observed prefix only. The future teacher target
    is detached and does not enter the model's attention computation.
    """
    if temporal_context_policy != "past_only_prefix_last_tubelet_v1":
        raise ValueError("Attribution requires verified past-only prefix features")
    if features.ndim != 2 or valid.shape != features.shape[:1] or timestamps.shape != valid.shape:
        raise ValueError("Expected features [T,D], valid [T], timestamps [T]")
    if valid.dtype != torch.bool or not valid.any() or not valid[0] or ((~valid[:-1]) & valid[1:]).any():
        raise ValueError("Valid tokens must form a nonempty contiguous prefix")
    if not torch.isfinite(features[valid]).all() or not torch.isfinite(timestamps[valid]).all():
        raise ValueError("Valid features and timestamps must be finite")
    if frame_indices is not None:
        if (frame_indices.shape != valid.shape or
                (frame_indices[valid.to(device=frame_indices.device)] < 0).any()):
            raise ValueError("Invalid frame index metadata")
    length = int(valid.sum())
    if target.kind != "semantic_similarity" and (target.timestep is None or not 0 <= target.timestep < length):
        raise ValueError("Target timestep must be a valid token")
    if target.kind in ("future_prediction", "future_error"):
        if target.future_timestep is None or not target.timestep < target.future_timestep < length:
            raise ValueError("Future timestep must be valid and after target timestep")
    observed = length if target.kind == "semantic_similarity" else target.timestep + 1
    selected_layers = tuple(range(len(model.temporal.layers))) if layers is None else tuple(layers)
    all_heads = model.temporal.layers[0].self_attn.num_heads
    selected_heads = tuple(range(all_heads)) if heads is None else tuple(heads)
    if (not selected_layers or len(set(selected_layers)) != len(selected_layers)
            or min(selected_layers) < 0 or max(selected_layers) >= len(model.temporal.layers)
            or not selected_heads or len(set(selected_heads)) != len(selected_heads)
            or min(selected_heads) < 0 or max(selected_heads) >= all_heads):
        raise ValueError("Invalid layer or head selection")
    if method not in ("positive_gradient", "attention_gradient"):
        raise ValueError("Unknown attribution method")
    if len(selected_layers) * all_heads * observed * observed > max_attention_elements:
        raise MemoryError("Attention capture exceeds max_attention_elements; select layers or shorter input")
    old_mode = model.training
    model.eval()
    try:
        with torch.enable_grad(), capture_attention(model, selected_layers) as attention:
            output = model(features[None, :observed], valid[None, :observed],
                           timestamps[None, :observed])
            scalar = _scalar(model, output, features[None, :length],
                             timestamps[None, :length], target)
            maps = [attention[i] for i in selected_layers]
            gradients = torch.autograd.grad(scalar, maps, allow_unused=True)
            gradients = [torch.zeros_like(a) if g is None else g for a, g in zip(maps, gradients)]
            # Event pooling is an independent mechanism and is labeled separately.
            states = standardized(output["states"].detach())
            event = model.event_score(states).squeeze(-1).softmax(-1)[0]
            layer_maps, raw_maps, signed_maps = [], [], []
            for a, g in zip(maps, gradients):
                a = a[0, selected_heads]
                g = g[0, selected_heads]
                causal = torch.ones(observed, observed, dtype=torch.bool, device=a.device).tril()
                g = g.masked_fill(~causal, 0)
                a = a.masked_fill(~causal, 0)
                signed_maps.append(g.detach().cpu())
                raw_maps.append(a.detach().cpu())
                value = g.clamp_min(0) if method == "positive_gradient" else (a * g).clamp_min(0)
                layer_maps.append(value.detach().cpu())
    finally:
        model.train(old_mode)
    matrix = torch.stack(layer_maps)  # [L,H,Q,K]
    head_relevance = matrix.mean(2)  # mean over valid queries
    layer_relevance = head_relevance.mean(1)
    relevance = _normalize(layer_relevance.mean(0), normalization)
    signed = torch.stack(signed_maps)
    raw = torch.stack(raw_maps)
    # Softmax couples the probabilities in each row. Probability derivatives
    # can all be negative while derivatives of realizable logit changes have
    # both signs. Keep this distinction visible without changing LeGrad's rule.
    logit_gradient = raw * (signed - (raw * signed).sum(-1, keepdim=True))
    positive_mass = float(signed.clamp_min(0).sum())
    negative_mass = float((-signed).clamp_min(0).sum())
    gradient_statistics = dict(
        positive_mass=positive_mass, negative_mass=negative_mass,
        softmax_logit_positive_mass=float(logit_gradient.clamp_min(0).sum()),
        softmax_logit_negative_mass=float((-logit_gradient).clamp_min(0).sum()),
        nonzero_count=int(signed.count_nonzero()),
        positive_map_empty=not bool(matrix.count_nonzero()),
        interpretation=("Nonzero negative gradients were removed by positive clipping; an empty map does not imply no temporal dependence."
                        if not matrix.count_nonzero() and negative_mass > 0 else
                        "Attention-edge sensitivity; excludes residual and event-pooling paths. A zero derivative is not proof of no dependence."))
    # Restore original padded length with strict zeros outside the observed prefix.
    total = features.shape[0]
    def pad_last(value):
        return F.pad(value, (0, total - observed))
    def pad_square(value):
        return F.pad(value, (0, total - observed, 0, total - observed))
    return TemporalAttribution(
        video_id=video_id, sample_id=sample_id,
        target=dict(kind=target.kind, timestep=target.timestep, component=target.component,
                    future_timestep=target.future_timestep,
                    reference_embedding=(None if target.reference is None else
                                         target.reference.detach().cpu().flatten().tolist())),
        target_value=float(scalar.detach().cpu()),
        target_timestamp=None if target.timestep is None else float(timestamps[target.timestep]),
        valid_length=length,
        frame_indices=None if frame_indices is None else frame_indices.detach().cpu(),
        timestamps=timestamps.detach().cpu(), relevance=pad_last(relevance),
        layer_relevance=pad_last(layer_relevance), head_relevance=pad_last(head_relevance),
        signed_gradients=pad_square(torch.stack(signed_maps)) if include_maps else None,
        raw_attention=pad_square(torch.stack(raw_maps)) if include_maps else None,
        relevance_maps=pad_square(matrix) if include_maps else None,
        event_pooling_weights=pad_last(event.detach().cpu()),
        layer_indices=selected_layers, head_indices=selected_heads,
        method=method, normalization=normalization, checkpoint_id=checkpoint_id,
        temporal_context_policy=temporal_context_policy,
        gradient_statistics=gradient_statistics)


def attribute_batch(model: TrajectoryModel, features: torch.Tensor, valid: torch.Tensor,
                    timestamps: torch.Tensor, targets: list[AttributionTarget],
                    video_ids: list[str], **kwargs) -> list[TemporalAttribution]:
    """Sequential sample attribution avoids cross-sample gradients and quadratic spikes."""
    if features.ndim != 3 or len(targets) != len(video_ids) or len(targets) != features.shape[0]:
        raise ValueError("Batch inputs and target metadata disagree")
    return [attribute_trajectory(model, features[i], valid[i], timestamps[i], targets[i],
                                 video_id=video_ids[i], **kwargs) for i in range(len(targets))]


def perturbation_check(model: TrajectoryModel, features: torch.Tensor, valid: torch.Tensor,
                       timestamps: torch.Tensor, target: AttributionTarget,
                       relevance: torch.Tensor, *, count: int = 1) -> dict:
    """Compare high/low ranked historical-token replacement by the prefix mean.

    This changes the input distribution and is evidence, not causal proof.
    """
    length = int(valid.sum())
    candidates = torch.arange(length)
    if target.timestep is not None:
        candidates = candidates[candidates <= target.timestep]
    if count < 1 or 2 * count > len(candidates):
        raise ValueError("Need at least two disjoint historical token sets")
    ranked = candidates[torch.argsort(relevance[candidates], stable=True)]
    def score(x):
        observed = length if target.kind == "semantic_similarity" else target.timestep + 1
        with torch.no_grad():
            output = model(x[None, :observed], valid[None, :observed], timestamps[None, :observed])
            return float(_scalar(model, output, x[None, :length], timestamps[None, :length], target))
    old_mode = model.training
    model.eval()
    try:
        original = score(features)
        replacement = features[candidates].mean(0)
        changes = {}
        for name, indices in (("low", ranked[:count]), ("high", ranked[-count:])):
            changed = features.clone()
            changed[indices] = replacement
            changes[name] = abs(score(changed) - original)
        return dict(original_score=original, high_score_change=changes["high"],
                    low_score_change=changes["low"], replacement="historical_feature_mean",
                    caveat="Replacement may be out of distribution; this is not causal proof")
    finally:
        model.train(old_mode)
