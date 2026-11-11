import torch

from tests.test_online_trajectory import example, full_features
from vision_model.interpretability.prefix_input_gradients import (
    prefix_input_gradient, trapezoid_attributions)


def test_input_replay_matches_direct_autograd_and_accumulates_earlier_prefixes():
    torch.set_num_threads(1)
    encoder, model, pixels, _, valid, times, _ = example(2)
    encoder.requires_grad_(False); model.requires_grad_(False)

    def score(features):
        output = model(features.flatten(0, 1),
            valid[:, None].expand(-1, 2, -1).flatten(0, 1),
            times[:, None].expand(-1, 2, -1).flatten(0, 1), detach_teacher=False)
        return output["embedding"][0, 0]

    original = pixels.clone().requires_grad_()
    direct = score(full_features(encoder, original))
    expected, = torch.autograd.grad(direct, original)
    result = prefix_input_gradient(encoder, pixels, valid, score, torch.device("cpu"))
    torch.testing.assert_close(result["gradient"], expected, atol=2e-7, rtol=2e-4)
    assert abs(result["score"] - float(direct.detach())) < 1e-6
    assert result["replay_max_error"] == 0
    assert not result["gradient"][1].any()  # unrelated source
    assert not result["gradient"][0, 1].any()  # unused mirror decision
    assert result["gradient"][0, 0, :2].abs().sum() > 0
    assert all(p.grad is None for p in encoder.parameters())


def test_invalid_tail_has_zero_input_gradient():
    encoder, _, pixels, _, valid, _, _ = example(2)
    encoder.requires_grad_(False)
    valid[:, 2] = False
    pixels[:, :, 4:] = float("nan")
    result = prefix_input_gradient(encoder, pixels, valid,
        lambda f: f[:, :, :2].square().sum(), torch.device("cpu"))
    assert not result["gradient"][:, :, 4:].any()
    assert torch.isfinite(result["gradient"]).all()


def test_integrated_gradient_completeness_on_quadratic():
    start, end = torch.tensor([-1., 2.]), torch.tensor([2., 4.])
    gradients = [2 * torch.lerp(start, end, a) for a in torch.linspace(0, 1, 9)]
    attribution = trapezoid_attributions(start, end, gradients)
    torch.testing.assert_close(attribution.sum(), end.square().sum() - start.square().sum())
