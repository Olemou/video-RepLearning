"""Independent derivative and bounded-cosine checks for the plateau audit."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest

from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss


def fixture():
    torch.manual_seed(71)
    mean = torch.randn(6, 4, requires_grad=True)
    temporal = torch.randn(6, 3, 4, requires_grad=True)
    variance = (torch.rand(6, 3, 4) + 0.2).requires_grad_()
    video_variance = (torch.rand(6, 4) + 0.2).requires_grad_()
    valid = torch.tensor([[True, True, False]] * 6)
    return SimpleNamespace(video_mean=mean, temporal_mean=temporal,
                           temporal_variance=variance, video_variance=video_variance,
                           temporal_valid_mask=valid)


def test_contrastive_gradient_matches_frozen_weight_analytic_derivative():
    out = fixture()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(uncertainty_weight=0))
    result = objective(output=out, labels=labels, original_ids=list(range(6)), epoch=23)
    actual, variance_grad = torch.autograd.grad(result.loss_sum, (out.video_mean, out.video_variance), allow_unused=True)
    with torch.no_grad():
        pos = labels[:, None].eq(labels[None, :]) & ~torch.eye(6, dtype=torch.bool)
        neg = labels[:, None].ne(labels[None, :])
        d = ((out.video_mean[:, None] - out.video_mean[None, :]).square() /
             (out.video_variance[:, None] + out.video_variance[None, :] + 1e-6)).mean(-1)
        lam, temp = objective.curriculum(23)
        weights = (1 - lam * (1 - 1 / (1 + d))) * pos
        q = weights / weights.sum(1, keepdim=True)
        nw = torch.softmax((-d / temp).masked_fill(~neg, -torch.inf), 1) * neg.sum(1, keepdim=True)
        z = F.normalize(out.video_mean, dim=-1)
        mass = torch.exp(z @ z.T / 0.1) * (pos + nw)
        p = mass / mass.sum(1, keepdim=True)
        ds = p - q
        dz = (ds + ds.T) @ z / 0.1
        expected = (dz - z * (dz * z).sum(1, keepdim=True)) / out.video_mean.norm(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
    assert variance_grad is None


def test_uncertainty_gradient_matches_formula_and_masks_padding():
    out = fixture()
    kwargs = dict(output=out, labels=torch.tensor([0, 0, 0, 1, 1, 1]), original_ids=list(range(6)), epoch=0)
    full = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(uncertainty_weight=1))(**kwargs)
    con = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(uncertainty_weight=0))(**kwargs)
    variance_grad, temporal_grad, mean_grad = torch.autograd.grad(
        full.loss_sum - con.loss_sum,
        (out.temporal_variance, out.temporal_mean, out.video_mean), allow_unused=True)
    residual = (out.temporal_mean - out.video_mean[:, None]).detach()
    expected = (1 / out.temporal_variance - residual.square() / out.temporal_variance.square()) / 8
    expected = expected * out.temporal_valid_mask[..., None]
    torch.testing.assert_close(variance_grad, expected)
    assert temporal_grad is None
    torch.testing.assert_close(mean_grad, torch.zeros_like(mean_grad))


def test_antipodal_classes_reach_uniform_positive_reference():
    out = fixture()
    out.video_mean = torch.tensor([[1., 0.]] * 3 + [[-1., 0.]] * 3)
    out.video_variance = torch.ones_like(out.video_mean)
    out.temporal_mean = out.video_mean[:, None].expand(-1, 3, -1)
    out.temporal_variance = torch.ones_like(out.temporal_mean)
    result = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(uncertainty_weight=0))(
        output=out, labels=torch.tensor([0, 0, 0, 1, 1, 1]), original_ids=list(range(6)), epoch=0)
    # Finite-temperature correction is log(1 + 3/2 exp(-20)), below float32 resolution.
    torch.testing.assert_close(result.loss_sum / 6, torch.tensor(2.).log(), atol=1e-6, rtol=0)


def test_autocast_preserves_float32_loss_and_gradients():
    out = fixture()
    objective = StableUncertaintyContrastiveLoss()
    kwargs = dict(output=out, labels=torch.tensor([0, 0, 0, 1, 1, 1]), original_ids=list(range(6)), epoch=9)
    reference = objective(**kwargs)
    expected = torch.autograd.grad(reference.loss_sum, out.video_mean)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = objective(**kwargs)
    gradient = torch.autograd.grad(actual.loss_sum, out.video_mean)[0]
    torch.testing.assert_close(actual.loss_sum, reference.loss_sum, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


def test_all_invalid_temporal_sample_is_rejected():
    out = fixture()
    out.temporal_valid_mask[0] = False
    with pytest.raises(ValueError, match="at least one valid temporal"):
        StableUncertaintyContrastiveLoss()(output=out, labels=torch.tensor([0, 0, 0, 1, 1, 1]),
                                           original_ids=list(range(6)), epoch=0)


def test_plain_supcon_is_exactly_unweighted_cross_entropy():
    out = fixture()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    objective = StableUncertaintyContrastiveLoss(GaussianContrastiveLossConfig(pair_weighting=False, uncertainty_weight=0))
    result = objective(output=out, labels=labels, original_ids=list(range(6)), epoch=0)
    z = F.normalize(out.video_mean, dim=-1)
    logits = z @ z.T / 0.1
    pos = labels[:, None].eq(labels[None, :]) & ~torch.eye(6, dtype=torch.bool)
    logp = logits - torch.logsumexp(logits.masked_fill(torch.eye(6, dtype=torch.bool), -torch.inf), 1)[:, None]
    expected = -logp[pos].sum() / 2
    torch.testing.assert_close(result.loss_sum, expected)
