"""Numerical training equivalence, including real two-process Gloo DDP."""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_training import TinyJEPA, context
from vision_model.training.trainer import Trainer, TrainerConfig
from vision_model.training.distributed import DistributedContext
from vision_model.training.loss import StableUncertaintyContrastiveLoss


def batch(values, labels):
    return {"pixel_values_videos": torch.tensor(values).reshape(-1, 1, 1, 1, 1),
            "labels": torch.tensor(labels), "video_id": [str(v) for v in values]}


def reference_update(model, batches):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    objective = StableUncertaintyContrastiveLoss()
    results = [objective(output=model(b["pixel_values_videos"]), labels=b["labels"],
                         original_ids=b["video_id"], epoch=0) for b in batches]
    (sum(r.loss_sum for r in results) / sum(r.sample_count for r in results)).backward()
    optimizer.step()
    return sum(r.sample_count for r in results)


def test_accumulation_uses_valid_anchor_count_and_handles_partial_window():
    batches = [batch([1., 1.1, -1.], [0, 0, 1]),
               batch([0.9, 1.2, -1.1, -0.8], [0, 0, 1, 1]),
               batch([1.3, 0.7, -0.7], [0, 0, 1])]
    expected, actual = TinyJEPA(), TinyJEPA()
    reference_update(expected, batches[:2])
    reference_update(expected, batches[2:])
    trainer = Trainer(model=actual, optimizer=torch.optim.SGD(actual.parameters(), lr=0.01),
                      distributed=context(), config=TrainerConfig(amp_dtype="float32",
                      gradient_accumulation_steps=2, max_grad_norm=None))
    metrics = trainer.train_epoch(batches, epoch=0)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b)
    assert trainer.global_step == 2
    assert metrics["valid_anchors"] == 8


def ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(TinyJEPA())
        batches = ([batch([1., 1.1, 1.2], [0, 0, 0]), batch([0.9], [0])]
                   if rank == 0 else
                   [batch([-1., -1.2], [1, 1]), batch([-0.9, -1.1], [1, 1])])
        trainer = Trainer(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                          distributed=DistributedContext(rank, 2, rank, torch.device("cpu")),
                          config=TrainerConfig(amp_dtype="float32", gradient_accumulation_steps=2,
                                               max_grad_norm=None))
        metrics = trainer.train_epoch(batches, epoch=0)
        torch.save({"state": model.module.state_dict(), "metrics": metrics}, Path(destination) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_ddp_global_pairs_match_single_process_with_unequal_and_zero_local_anchors(tmp_path):
    expected = TinyJEPA()
    count = reference_update(expected, [batch([1., 1.1, 1.2, -1., -1.2], [0, 0, 0, 1, 1]),
                                        batch([0.9, -0.9, -1.1], [0, 1, 1])])
    mp.spawn(ddp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(result["state"][name], value, atol=1e-7, rtol=1e-6)
        assert result["metrics"]["valid_anchors"] == count


class SkippingScaler:
    """Emulate one overflow without requiring a CUDA device in this test."""
    def __init__(self):
        self.value = 16.
    def scale(self, value):
        return value
    def unscale_(self, optimizer):
        pass
    def get_scale(self):
        return self.value
    def step(self, optimizer):
        pass
    def update(self):
        self.value /= 2


def test_skipped_update_does_not_advance_schedule_or_step():
    model = TinyJEPA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = Trainer(model=model, optimizer=optimizer, distributed=context(),
                      config=TrainerConfig(amp_dtype="float32"), scheduler=scheduler)
    trainer.scaler = SkippingScaler()
    previous_epoch = scheduler.last_epoch
    trainer.train_epoch([batch([1., 1.1, -1.], [0, 0, 1])], epoch=0)
    assert trainer.global_step == 0
    assert scheduler.last_epoch == previous_epoch
    assert model.weight.item() == 0.5
