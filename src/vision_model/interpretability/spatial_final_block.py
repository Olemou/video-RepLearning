"""Target-conditioned spatial sensitivity of frozen V-JEPA 2 final attention.

This is a one-block, final-tubelet-query adaptation of LeGrad's positive
attention-gradient rule. The scalar target comes from the trained trajectory
model; the backbone attention itself remains frozen. These maps are neither
the full multi-layer LeGrad method nor validated spill segmentation masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from vision_model.models.final_block_maps import replay_last_tubelet
from vision_model.training.trajectory import TrajectoryModel
from .temporal_attribution import AttributionTarget, _scalar


@dataclass
class SpatialFinalBlockResult:
    positive_maps: torch.Tensor  # [tubelet, patch_row, patch_column], CPU
    signed_maps: torch.Tensor
    raw_attention_maps: torch.Tensor
    feature_gradient_norms: torch.Tensor
    replay_feature_max_error: float
    replay_score_error: float
    replay_score: float
    method: str = "final_block_positive_attention_gradient_v1"

    def metadata(self) -> dict:
        return dict(method=self.method, encoder_block="final_only", backbone_frozen=True,
                    scalar_target_source="trained_trajectory_model",
                    attention_probability="post_softmax",
                    aggregation="ReLU(d target / d A), mean over all heads and consumed final-tubelet queries; sum over causal prefixes containing each key tubelet",
                    replay_feature_max_error=self.replay_feature_max_error,
                    replay_score_error=self.replay_score_error,
                    replay_score=self.replay_score,
                    feature_gradient_norms=self.feature_gradient_norms.tolist(),
                    spatial_grid=list(self.positive_maps.shape[1:]),
                    limitations=["Frozen V-JEPA attention did not learn from spill labels",
                                 "Final-block diagnostic only; not all-layer LeGrad",
                                 "Patch sensitivity is not a segmentation probability or validated spill mask"])


def _score_and_feature_gradients(model: TrajectoryModel, features: torch.Tensor,
                                 timestamps: torch.Tensor, target: AttributionTarget):
    length = len(features)
    observed = length if target.kind == "semantic_similarity" else target.timestep + 1
    source = features.detach().clone().requires_grad_(True)
    valid = torch.ones(1, observed, dtype=torch.bool, device=source.device)
    output = model(source[None, :observed], valid, timestamps[None, :observed],
                   detach_teacher=False)
    scalar = _scalar(model, output, source[None], timestamps[None], target)
    gradient = torch.autograd.grad(scalar, source)[0]
    return scalar.detach(), gradient.detach()


def attribute_spatial_final_block(encoder, model: TrajectoryModel,
                                  final_inputs: list[torch.Tensor],
                                  reference_features: torch.Tensor,
                                  timestamps: torch.Tensor,
                                  target: AttributionTarget, *,
                                  max_replay_feature_error: float = .08,
                                  max_replay_score_error: float = .05) -> SpatialFinalBlockResult:
    """Replay only consumed final-block queries, one causal prefix at a time.

    ``final_inputs[p]`` is the frozen hidden state entering the final block
    when prefix p was encoded. Upstream gradients of the trajectory scalar are
    computed at replayed teacher features, then propagated to each prefix's
    final-block attention probabilities. Previous prefixes remain detached.
    """
    if not final_inputs or len(final_inputs) != len(reference_features):
        raise ValueError("Need one final-block input per causal teacher feature")
    if reference_features.ndim != 2 or timestamps.shape != reference_features.shape[:1]:
        raise ValueError("Expected features [T,D] and timestamps [T]")
    if max_replay_feature_error < 0 or max_replay_score_error < 0:
        raise ValueError("Replay tolerances must be nonnegative")
    device = next(encoder.parameters()).device
    if next(model.parameters()).device != device:
        raise ValueError("Encoder and trajectory model must share a device")
    patch_grid = None
    replayed = []
    encoder.eval()
    model.eval()
    for prefix, original in enumerate(final_inputs, start=1):
        if original.ndim != 3 or original.shape[0] != 1:
            raise ValueError("Final-block inputs must be [1,prefix*spatial,D]")
        spatial = original.shape[1] // prefix
        side = int(spatial ** .5)
        if side * side != spatial or original.shape[1] != prefix * spatial:
            raise ValueError("Cannot reconstruct the final-block patch grid")
        if patch_grid is None:
            patch_grid = side
        elif patch_grid != side:
            raise ValueError("Patch grid changed between prefixes")
        hidden = original.to(device).clone()  # leave inference-mode storage behind
        with torch.no_grad():
            state, _ = replay_last_tubelet(encoder, hidden, frames=prefix,
                                           height=side, width=side)
        replayed.append(state.mean(1).float()[0].cpu())
    replayed = torch.stack(replayed)
    feature_error = float((replayed - reference_features.cpu()).abs().max())
    if feature_error > max_replay_feature_error:
        raise RuntimeError(f"Final-block replay differs from encoded features by {feature_error:g}")
    replay_device = replayed.to(device)
    times = timestamps.to(device)
    replay_score, upstream = _score_and_feature_gradients(model, replay_device, times, target)
    reference_score, _ = _score_and_feature_gradients(model, reference_features.to(device),
                                                     times, target)
    score_error = abs(float(replay_score) - float(reference_score))
    if score_error > max_replay_score_error:
        raise RuntimeError(f"Final-block replay changes the target score by {score_error:g}")
    length, side = len(replayed), patch_grid
    positive = torch.zeros(length, side, side)
    signed = torch.zeros_like(positive)
    raw = torch.zeros_like(positive)
    for prefix, original in enumerate(final_inputs, start=1):
        if not upstream[prefix - 1].abs().any():
            continue
        hidden = original.to(device).clone().requires_grad_(True)
        with torch.enable_grad():
            state, attention = replay_last_tubelet(encoder, hidden, frames=prefix,
                                                   height=side, width=side)
            feature = state.mean(1).float()[0]
            gradient = torch.autograd.grad(feature, attention,
                                           grad_outputs=upstream[prefix - 1])[0][0]
        # [head, consumed query, key] -> [prefix tubelet, patch row, patch col]
        positive[:prefix] += gradient.clamp_min(0).mean((0, 1)).reshape(prefix, side, side).float().cpu()
        signed[:prefix] += gradient.mean((0, 1)).reshape(prefix, side, side).float().cpu()
        raw[:prefix] += attention.detach()[0].mean((0, 1)).reshape(prefix, side, side).float().cpu()
    return SpatialFinalBlockResult(positive_maps=positive, signed_maps=signed,
                                   raw_attention_maps=raw,
                                   feature_gradient_norms=upstream.norm(dim=-1).cpu(),
                                   replay_feature_max_error=feature_error,
                                   replay_score_error=score_error,
                                   replay_score=float(replay_score))
