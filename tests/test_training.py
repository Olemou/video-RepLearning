from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from vision_model.training.checkpoint import load_checkpoint, save_checkpoint
from vision_model.training.distributed import DistributedContext
from vision_model.training.optimizer import OptimizerConfig, build_optimizer, parameter_groups
from vision_model.training.loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss
from vision_model.training.trainer import Trainer, TrainerConfig


class TinyJEPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.log_variance = nn.Parameter(torch.tensor(-1.0))

    def forward(self, pixel_values_videos, temporal_valid_mask=None):
        batch = pixel_values_videos.shape[0]
        base = pixel_values_videos.float().mean(dim=(1, 2, 3, 4)) * self.weight
        temporal_mean = torch.stack(
            (torch.stack((base, torch.full_like(base, 0.5)), dim=-1),
             torch.stack((base + 0.2, torch.full_like(base, -0.25)), dim=-1)),
            dim=1,
        )
        variance = torch.nn.functional.softplus(self.log_variance).expand_as(temporal_mean)
        video_mean = temporal_mean.mean(dim=1)
        video_variance = variance.mean(dim=1) + temporal_mean.var(dim=1, unbiased=False)
        valid = torch.ones(batch, 2, dtype=torch.bool, device=base.device)
        return SimpleNamespace(
            temporal_mean=temporal_mean, temporal_variance=variance,
            video_mean=video_mean, video_variance=video_variance,
            temporal_valid_mask=valid,
        )


def context() -> DistributedContext:
    return DistributedContext(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


class OptimizerTests(unittest.TestCase):
    def test_parameter_groups_exclude_bias_and_one_dimensional_values(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        groups = parameter_groups(model, 0.1)
        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertIn(id(model[0].weight), decay_ids)
        self.assertIn(id(model[0].bias), no_decay_ids)
        self.assertIn(id(model[1].weight), no_decay_ids)
        self.assertEqual(groups[1]["weight_decay"], 0.0)


class TrainerTests(unittest.TestCase):
    def test_loss_requires_positive_and_negative(self) -> None:
        model = TinyJEPA()
        output = model(torch.ones(2, 2, 3, 2, 2))
        loss = StableUncertaintyContrastiveLoss()
        with self.assertRaisesRegex(ValueError, "both a positive and a negative"):
            loss(output=output, labels=torch.tensor([0, 0]), original_ids=["a", "b"], epoch=0)

    def test_curricula_and_detached_pair_weights(self) -> None:
        config = GaussianContrastiveLossConfig(total_epochs=10, uncertainty_weight=0.0)
        loss = StableUncertaintyContrastiveLoss(config)
        self.assertEqual(loss.curriculum(0), (1.0, 1.0))
        self.assertAlmostEqual(loss.curriculum(10)[0], 0.1)
        model = TinyJEPA()
        output = model(torch.tensor([1.0, 1.1, -1.0]).view(3, 1, 1, 1, 1))
        result = loss(output=output, labels=torch.tensor([0, 0, 1]), original_ids=["a", "b", "c"], epoch=0)
        result.loss_sum.backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(model.log_variance.grad is None or model.log_variance.grad.item() == 0.0)
        self.assertEqual(result.sample_count, 2)
        self.assertTrue(torch.isfinite(result.loss_sum))

    def test_gradient_flow_updates_encoder(self) -> None:
        student = TinyJEPA()
        optimizer = build_optimizer(student, OptimizerConfig(learning_rate=0.1, weight_decay=0.0))
        trainer = Trainer(
            model=student,
            optimizer=optimizer,
            distributed=context(),
            config=TrainerConfig(
                amp_dtype="float32", max_grad_norm=None,
                loss=GaussianContrastiveLossConfig(total_epochs=2),
            ),
        )
        before_student = student.weight.detach().clone()
        batch = {
            "pixel_values_videos": torch.tensor([1.0, 1.1, -1.0]).view(3, 1, 1, 1, 1),
            "labels": torch.tensor([0, 0, 1]),
            "video_id": ["zero-a", "zero-b", "one-a"],
        }
        metrics = trainer.train_epoch([batch], epoch=0)
        self.assertNotEqual(student.weight.item(), before_student.item())
        self.assertEqual(metrics["valid_anchors"], 2)
        self.assertIn("contrastive_reference_entropy", metrics)
        self.assertIn("contrastive_excess_loss", metrics)

    def test_checkpoint_round_trip(self) -> None:
        student = TinyJEPA()
        optimizer = build_optimizer(student, OptimizerConfig())
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=student,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                epoch=3,
                global_step=12,
                loss_name="stable-uncertainty-contrastive-v1",
            )
            student.weight.data.zero_()
            epoch, step = load_checkpoint(
                path,
                model=student,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                expected_loss_name="stable-uncertainty-contrastive-v1",
            )
            self.assertEqual((epoch, step), (3, 12))
            self.assertEqual(student.weight.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
