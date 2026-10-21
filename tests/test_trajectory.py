import torch
import pytest
from vision_model.training.trajectory import TrajectoryModel, trajectory_loss


def example():
    torch.manual_seed(7)
    x = torch.randn(4, 2, 6, 16)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0]] * 4, dtype=torch.bool)
    times = torch.arange(6).float()[None].expand(4, -1) * .5
    return x, valid, times, torch.tensor([0, 0, 1, 1])


def test_joint_loss_weights_and_all_heads_learn():
    x, valid, times, labels = example()
    model = TrajectoryModel(16, hidden=16, embedding_dim=8)
    loss, m, _ = trajectory_loss(model, x, valid, times, labels, epoch=1)
    expected = m["semantic_loss"] + .1 * m["alignment_loss"] + m["prediction_loss"] + .1 * m["retention_loss"]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for module in (model.semantic, model.dynamics, model.future_delta, model.up):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_causal_states_do_not_change_when_future_is_changed():
    x, valid, times, _ = example()
    model = TrajectoryModel(16, hidden=16).eval()
    # Nonzero adapter makes this exercise the attention mask, not only the bypass.
    torch.nn.init.normal_(model.up.weight, std=.1)
    a = model(x[:, 0], valid, times)["states"]
    changed = x[:, 0].clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 10
    b = model(changed, valid, times)["states"]
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=1e-6)
    short = model(x[:, 0, :2], valid[:, :2], times[:, :2])["states"]
    torch.testing.assert_close(a[:, :2], short, rtol=1e-5, atol=1e-6)


def test_padding_nan_ignored_and_teacher_detached():
    x, valid, times, labels = example()
    model = TrajectoryModel(16, hidden=16)
    original, _, _ = trajectory_loss(model, x, valid, times, labels, epoch=3)
    x[:, :, 4:] = float("nan")
    times[:, 4:] = float("nan")
    x.requires_grad_()
    loss, metrics, _ = trajectory_loss(model, x, valid, times, labels, epoch=3)
    torch.testing.assert_close(loss, original)
    assert all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert x.grad is None


def test_no_future_pairs_rejected():
    x, valid, times, labels = example()
    with pytest.raises(ValueError, match="No valid future"):
        trajectory_loss(TrajectoryModel(16, hidden=16), x[:, :, :1], valid[:, :1], times[:, :1], labels, epoch=1)
