import math

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from vision_model.training.distributed import unpadded_sampler_indices
from vision_model.training.lr_search import batch_plan, scaled_lr, source_split, warmup_cosine
from vision_model.training.trajectory import DistributedTrajectoryStep, TrajectoryModel


def test_source_split_keeps_groups_and_classes_separate():
    labels = torch.tensor([0] * 24 + [1] * 24)
    sources = [f"source_{i}" for i in range(48)]
    train, validation = source_split(labels, sources, seed=239, validation_per_class=8)
    assert len(train) == 32 and len(validation) == 16
    assert set(train).isdisjoint(validation)
    assert torch.bincount(labels[validation], minlength=2).tolist() == [8, 8]


def test_batch_plan_and_candidate_scaling():
    with pytest.raises(ValueError):
        batch_plan(16, 2, 2)
    with pytest.raises(ValueError):
        batch_plan(34, 2, 2)
    assert batch_plan(64, 2, 2)["per_rank_microbatch"] == 16
    assert scaled_lr(.001, 64, policy="constant") == pytest.approx(.001)
    assert scaled_lr(.001, 64, policy="sqrt") == pytest.approx(.001 * math.sqrt(2))
    assert scaled_lr(.001, 64, policy="linear") == pytest.approx(.002)


def test_warmup_cosine_reaches_peak_then_final():
    rates = [warmup_cosine(step, 20, .001) for step in range(1, 21)]
    assert rates[0] < rates[1] == pytest.approx(.001)
    assert rates[-1] == pytest.approx(.00005)
    assert all(a >= b for a, b in zip(rates[1:], rates[2:]))


def test_odd_class_sampler_padding_is_excluded_before_batches():
    shards = []
    counts = []
    for rank in (0, 1):
        sampler = DistributedSampler(range(33), num_replicas=2, rank=rank,
                                     shuffle=True, seed=239, drop_last=False)
        sampler.set_epoch(3)
        indices, removed = unpadded_sampler_indices(sampler, 33)
        shards.append(indices)
        counts.append(removed)
    assert counts == [0, 1]
    assert sorted(shards[0] + shards[1]) == list(range(33))
    assert not set(shards[0]).intersection(shards[1])


def test_microbatched_global_objective_matches_unsplit_gradients():
    torch.manual_seed(11)
    teacher = torch.randn(4, 2, 6, 16)
    valid = torch.ones(4, 6, dtype=torch.bool)
    times = torch.arange(6).float().expand(4, -1)
    labels = torch.tensor([0, 0, 1, 1])
    first = DistributedTrajectoryStep(TrajectoryModel(16))
    second = DistributedTrajectoryStep(TrajectoryModel(16))
    second.load_state_dict(first.state_dict())
    a, _, _ = first(teacher, valid, times, labels, epoch=2)
    b, _, _ = second(teacher, valid, times, labels, epoch=2, microbatch_sources=2)
    a.backward()
    b.backward()
    assert a.item() == pytest.approx(b.item(), abs=1e-6)
    for p, q in zip(first.parameters(), second.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-5)
