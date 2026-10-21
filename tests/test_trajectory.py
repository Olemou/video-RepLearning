import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from vision_model.training.trajectory import TrajectoryModel, DistributedTrajectoryStep, trajectory_loss
from vision_model.training.distributed import unpadded_sampler_indices, audit_distributed_indices


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


@pytest.mark.parametrize("count,world", [(7, 3), (8, 3)])
def test_distributed_sampler_removes_only_padding_and_shuffles(count, world):
    partitions = []
    for epoch in (1, 2):
        shards = []
        removed = 0
        for rank in range(world):
            sampler = DistributedSampler(range(count), num_replicas=world,
                                         rank=rank, shuffle=True, seed=239, drop_last=False)
            sampler.set_epoch(epoch)
            indices, padding = unpadded_sampler_indices(sampler, count)
            shards.append(indices)
            removed += padding
        assert sorted(index for shard in shards for index in shard) == list(range(count))
        assert removed == world * ((count + world - 1) // world) - count
        partitions.append(shards)
    assert partitions[0] != partitions[1]


def _distributed_gradient_worker(rank, init_file, source_count, microbatch_sources):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        x, valid, times, labels = example()
        if source_count == 5:
            x = torch.cat((x, x[:1] + 0.3), dim=0)
            valid = torch.cat((valid, valid[:1]), dim=0)
            times = torch.cat((times, times[:1]), dim=0)
            labels = torch.cat((labels, labels[:1]), dim=0)
        torch.manual_seed(11)
        reference = TrajectoryModel(16, hidden=16, embedding_dim=8)
        expected, _, _ = trajectory_loss(reference, x, valid, times, labels, epoch=3)
        expected.backward()
        torch.manual_seed(11)
        model = TrajectoryModel(16, hidden=16, embedding_dim=8)
        distributed = DistributedDataParallel(DistributedTrajectoryStep(model))
        sampler = DistributedSampler(range(source_count), num_replicas=2, rank=rank,
                                     shuffle=True, seed=239, drop_last=False)
        sampler.set_epoch(3)
        indices, padding = unpadded_sampler_indices(sampler, source_count)
        order, counts = audit_distributed_indices(indices, source_count)
        assert sum(counts) == source_count
        assert padding == (1 if source_count == 5 and rank == 1 else 0)
        actual, metrics, output = distributed(x[indices], valid[indices],
                                               times[indices], labels[indices], epoch=3,
                                               microbatch_sources=microbatch_sources)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-4)
        torch.testing.assert_close(metrics["prediction_loss"],
                                   trajectory_loss(reference, x[order], valid[order],
                                                   times[order], labels[order], epoch=3)[1]["prediction_loss"],
                                   rtol=3e-5, atol=1e-4)
        assert len(output["embedding"]) == 2 * source_count
        actual.backward()
        for (name, parameter), (_, reference_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad,
                                       rtol=1e-3, atol=1e-4, msg=lambda message: f"{name}: {message}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("source_count", [4, 5])
@pytest.mark.parametrize("microbatch_sources", [None, 1])
def test_two_rank_gradient_matches_full_batch(tmp_path, source_count, microbatch_sources):
    mp.spawn(_distributed_gradient_worker,
             args=(str(tmp_path / "init"), source_count, microbatch_sources), nprocs=2)
