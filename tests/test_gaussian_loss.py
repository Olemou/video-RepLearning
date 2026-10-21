from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from vision_model.training.loss import (
    GaussianContrastiveLossConfig,
    StableUncertaintyContrastiveLoss,
)


def output_for(
    means: torch.Tensor,
    *,
    temporal_variance: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> SimpleNamespace:
    temporal_mean = means[:, None, :].repeat(1, 2, 1).clone()
    variance_t = torch.full_like(temporal_mean, temporal_variance)
    valid = valid_mask if valid_mask is not None else torch.ones(means.shape[0], 2, dtype=torch.bool)
    return SimpleNamespace(
        temporal_mean=temporal_mean,
        temporal_variance=variance_t,
        video_mean=means,
        video_variance=torch.full_like(means, temporal_variance),
        temporal_valid_mask=valid,
    )


class GaussianLossTests(unittest.TestCase):
    def test_matches_symmetric_hand_computation(self) -> None:
        means = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        loss = StableUncertaintyContrastiveLoss(
            GaussianContrastiveLossConfig(uncertainty_weight=0.0)
        )
        result = loss(
            output=output_for(means),
            labels=torch.tensor([0, 0, 1, 1]),
            original_ids=["a", "b", "c", "d"],
            epoch=0,
        )
        expected_per_anchor = math.log1p(2.0 * math.exp(-10.0))
        self.assertAlmostEqual(result.loss_sum.item(), 4.0 * expected_per_anchor, places=5)
        self.assertEqual(result.sample_count, 4)
        self.assertAlmostEqual(result.negative_weight_sum.item(), 8.0, places=5)

    def test_positive_entropy_explains_multi_positive_floor(self) -> None:
        means = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
             [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        )
        loss = StableUncertaintyContrastiveLoss(
            GaussianContrastiveLossConfig(uncertainty_weight=0.0)
        )
        result = loss(
            output=output_for(means),
            labels=torch.tensor([0, 0, 0, 1, 1, 1]),
            original_ids=["a", "b", "c", "d", "e", "f"],
            epoch=0,
        )
        expected_entropy = 6.0 * math.log(2.0)
        self.assertAlmostEqual(result.positive_entropy_sum.item(), expected_entropy, places=5)
        self.assertGreaterEqual(result.contrastive_sum.item() - result.positive_entropy_sum.item(), 0.0)
        self.assertLess(result.contrastive_sum.item() - result.positive_entropy_sum.item(), 0.001)

    def test_padding_is_excluded_from_uncertainty(self) -> None:
        means = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        output = output_for(means, temporal_variance=2.0)
        output.temporal_mean[:, 1] += 1000.0
        output.temporal_valid_mask[:, 1] = False
        loss = StableUncertaintyContrastiveLoss(
            GaussianContrastiveLossConfig(uncertainty_weight=1.0)
        )
        result = loss(
            output=output,
            labels=torch.tensor([0, 0, 1, 1]),
            original_ids=["a", "b", "c", "d"],
            epoch=0,
        )
        self.assertAlmostEqual(result.uncertainty_sum.item(), 4.0 * math.log(2.0), places=5)

    def test_uncertainty_gradient_reaches_variance_not_detached_residual(self) -> None:
        means = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], requires_grad=True
        )
        temporal_mean = means[:, None, :].repeat(1, 2, 1)
        temporal_mean = temporal_mean + torch.tensor([[[0.2, -0.1], [-0.2, 0.1]]])
        temporal_mean.retain_grad()
        raw_variance = torch.zeros_like(temporal_mean, requires_grad=True)
        variance_t = torch.nn.functional.softplus(raw_variance) + 1e-4
        output = SimpleNamespace(
            temporal_mean=temporal_mean,
            temporal_variance=variance_t,
            video_mean=means,
            video_variance=torch.ones_like(means),
            temporal_valid_mask=torch.ones(4, 2, dtype=torch.bool),
        )
        loss = StableUncertaintyContrastiveLoss(
            GaussianContrastiveLossConfig(uncertainty_weight=1.0)
        )
        result = loss(
            output=output,
            labels=torch.tensor([0, 0, 1, 1]),
            original_ids=["a", "b", "c", "d"],
            epoch=0,
        )
        result.loss_sum.backward()
        self.assertIsNotNone(raw_variance.grad)
        self.assertGreater(raw_variance.grad.abs().sum().item(), 0.0)
        # temporal_mean is used only by the detached uncertainty residual here.
        self.assertTrue(temporal_mean.grad is None or temporal_mean.grad.abs().sum().item() == 0.0)
        self.assertIsNotNone(means.grad)  # contrastive cosine geometry remains trainable


if __name__ == "__main__":
    unittest.main()


def test_small_compatibility_keeps_positive_supervision():
    means = (torch.tensor([[1., 0.], [.8, .2], [0., 1.], [.2, .8]]) * 10000).requires_grad_()
    output = output_for(means, temporal_variance=1e-6)
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(uncertainty_weight=0))
    result = objective(output=output, labels=torch.tensor([0, 0, 1, 1]), original_ids=list(range(4)), epoch=0)
    assert 0 < result.positive_weight_sum < 1e-6
    assert result.loss_sum > 0
    result.loss_sum.backward()
    assert torch.isfinite(means.grad).all()
    assert means.grad.abs().sum() > 0
