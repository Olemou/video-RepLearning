import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.training.online_trajectory import (
    OnlinePrefixEncoder, collect_prefix_features, replay_prefix_gradients,
    explicit_prefix_forward, rollout_last_tubelet, parameter_groups, gradient_norm)
from vision_model.training.trajectory import TrajectoryModel, DistributedTrajectoryStep, trajectory_loss


def example(count=4):
    torch.manual_seed(91)
    cfg = VJEPA2Architecture(frames_per_clip=6, crop_size=32, hidden_size=48,
                            num_hidden_layers=2, num_attention_heads=3, mlp_ratio=2.)
    encoder = OnlinePrefixEncoder(VJEPA2Encoder(cfg))
    trajectory = TrajectoryModel(48, hidden=16, embedding_dim=8)
    pixels = torch.randn(count, 2, 6, 3, 32, 32)
    teacher = torch.randn(count, 2, 3, 48)
    valid = torch.ones(count, 3, dtype=torch.bool)
    times = torch.arange(3).float()[None].expand(count, -1) * .5
    labels = torch.tensor([0, 0, 1, 1, 0][:count])
    return encoder, trajectory, pixels, teacher, valid, times, labels


def full_features(encoder, pixels):
    return torch.stack([encoder(pixels[:, :, :2 * (t + 1)].flatten(0, 1).permute(0, 2, 1, 3, 4))
                        .reshape(len(pixels), 2, -1) for t in range(pixels.shape[2] // 2)], 2)


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_replay_matches_end_to_end_four_loss_gradients(activation_checkpointing):
    torch.set_num_threads(1)
    encoder, trajectory, pixels, teacher, valid, times, labels = example()
    reference_encoder, reference_trajectory = copy.deepcopy(encoder), copy.deepcopy(trajectory)
    teacher.requires_grad_()
    expected, _, _ = trajectory_loss(reference_trajectory, teacher, valid, times, labels,
        epoch=3, student_features=full_features(reference_encoder, pixels))
    expected.backward()
    encoder.activation_checkpointing = activation_checkpointing
    features = collect_prefix_features(encoder, pixels, valid, torch.device("cpu"), chunk_sources=2)
    leaf = features.detach().requires_grad_()
    actual, metrics, _ = trajectory_loss(trajectory, teacher, valid, times, labels,
                                          epoch=3, student_features=leaf)
    actual.backward()
    error = replay_prefix_gradients(encoder, pixels, valid, leaf.grad, torch.device("cpu"),
                                    chunk_sources=2, replay_reference=features)
    assert error == 0
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, metrics["semantic_loss"] + .1 * metrics["alignment_loss"]
                               + metrics["prediction_loss"] + .1 * metrics["retention_loss"])
    assert teacher.grad is None
    for a, b in ((encoder, reference_encoder), (trajectory, reference_trajectory)):
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)
    assert all(gradient_norm(group) > 0 for group in parameter_groups(encoder.encoder).values())


def test_causal_prefix_and_invalid_tail_are_not_encoded():
    torch.set_num_threads(1)
    encoder, _, pixels, _, valid, _, _ = example()
    a = collect_prefix_features(encoder, pixels, valid, torch.device("cpu"))
    changed = pixels.clone(); changed[:, :, 4:] += 100
    b = collect_prefix_features(encoder, changed, valid, torch.device("cpu"))
    torch.testing.assert_close(a[:, :, :2], b[:, :, :2], rtol=0, atol=0)
    valid[:, 2] = False; changed[:, :, 4:] = float("nan")
    padded = collect_prefix_features(encoder, changed, valid, torch.device("cpu"))
    assert torch.isfinite(padded).all() and not padded[:, :, 2].any()


def test_attention_capture_is_used_and_rollout_has_no_cls_assumption():
    torch.set_num_threads(1)
    encoder, _, pixels, *_ = example()
    clip = pixels[:1, 0].permute(0, 2, 1, 3, 4)
    feature, attention, tokens = explicit_prefix_forward(encoder.encoder, clip)
    torch.testing.assert_close(feature, encoder(clip), atol=1e-6, rtol=1e-5)
    derivatives = torch.autograd.grad(feature.square().sum(), attention)
    assert all(torch.isfinite(d).all() and d.abs().sum() > 0 for d in derivatives)
    rollout = rollout_last_tubelet(attention, 4)
    matrices = [(a.detach().mean((0, 1)) + torch.eye(a.shape[-1])) / 2 for a in attention]
    product = matrices[1] @ matrices[0]
    torch.testing.assert_close(rollout, product[-4:].mean(0))
    torch.testing.assert_close(rollout.sum(), torch.tensor(1.))
    assert tokens.shape == (1, 12, 48)


def _distributed_worker(rank, rendezvous, count):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        encoder, trajectory, pixels, teacher, valid, times, labels = example(count)
        re, rt = copy.deepcopy(encoder), copy.deepcopy(trajectory)
        expected, _, _ = trajectory_loss(rt, teacher, valid, times, labels, epoch=3,
                                         student_features=full_features(re, pixels))
        expected.backward()
        ids = torch.arange(count)[rank::2]
        wrapped_encoder = DDP(encoder)
        wrapped_step = DDP(DistributedTrajectoryStep(trajectory))
        features = collect_prefix_features(encoder, pixels[ids], valid[ids], torch.device("cpu"))
        leaf = features.detach().requires_grad_()
        loss, _, _ = wrapped_step(teacher[ids], valid[ids], times[ids], labels[ids],
                                  epoch=3, student_features=leaf)
        loss.backward()
        replay_prefix_gradients(wrapped_encoder, pixels[ids], valid[ids], leaf.grad,
                                 torch.device("cpu"), replay_reference=features)
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
        for actual, reference in ((encoder, re), (trajectory, rt)):
            for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=5e-4, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("count", [4, 5])
def test_two_rank_online_replay_matches_serial(tmp_path, count):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), count), nprocs=2)


def test_multilayer_explanation_uses_the_requested_margin_and_real_prefixes():
    from vision_model.interpretability.online_trajectory import explain_prefix_group
    torch.set_num_threads(1)
    encoder, trajectory, pixels, _, valid, times, _ = example()
    encoder.requires_grad_(False); trajectory.requires_grad_(False)
    pixels, valid, times = pixels[:2], valid[:2], times[:2]
    directions = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)

    def score(features):
        out = trajectory(features.flatten(0, 1), valid[:, None].expand(-1, 2, -1).flatten(0, 1),
                         times[:, None].expand(-1, 2, -1).flatten(0, 1), detach_teacher=False)
        return (out["embedding"].reshape(2, 2, -1)[:, 0] @ directions.T).diff(dim=1).squeeze(1)

    result = explain_prefix_group(encoder.encoder, pixels, score, torch.device("cpu"))
    reference = collect_prefix_features(encoder, pixels, valid, torch.device("cpu"))
    torch.testing.assert_close(result["explicit_scores"], score(reference))
    assert result["positive_gradients"].shape == (2, 2, 3, 2, 2)
    assert result["replay_max_feature_error"] == 0
    assert all(torch.isfinite(result[k]).all() for k in (
        "positive_gradients", "signed_gradients", "multilayer_positive", "final_attention_gradient", "rollout"))
    assert result["positive_gradients"].sum() > 0
    assert result["pixel_probabilities"] is False
