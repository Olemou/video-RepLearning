"""Supervised encoder training with the project Gaussian contrastive loss."""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .distributed import DistributedContext, all_reduce_sum, gather_contrastive_batch
from .loss import GaussianContrastiveLossConfig, StableUncertaintyContrastiveLoss

@dataclass(frozen=True)
class TrainerConfig:
    amp_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    seed: int = 239
    loss: GaussianContrastiveLossConfig = GaussianContrastiveLossConfig()
    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

def set_deterministic_seed(seed: int, rank: int = 0) -> None:
    value = seed + rank
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value): return value.to(device, non_blocking=True)
    if isinstance(value, list): return [_move(item, device) for item in value]
    if isinstance(value, tuple): return tuple(_move(item, device) for item in value)
    if isinstance(value, dict): return {key: _move(item, device) for key, item in value.items()}
    return value

class Trainer:
    def __init__(self, *, model: nn.Module, optimizer: torch.optim.Optimizer, distributed: DistributedContext, config: TrainerConfig, scheduler: Any | None = None, loss: StableUncertaintyContrastiveLoss | None = None) -> None:
        self.model, self.optimizer = model, optimizer
        self.loss = loss or StableUncertaintyContrastiveLoss(config.loss)
        self.distributed, self.config, self.scheduler, self.global_step = distributed, config, scheduler, 0
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if config.amp_dtype not in dtypes: raise ValueError(f"Unknown AMP dtype: {config.amp_dtype}")
        self.amp_dtype = dtypes[config.amp_dtype]
        self.scaler = torch.amp.GradScaler("cuda", enabled=distributed.device.type == "cuda" and self.amp_dtype == torch.float16)
        set_deterministic_seed(config.seed, distributed.rank)

    def train_epoch(self, loader: Iterable[Mapping[str, Any]], epoch: int) -> dict[str, float]:
        if not hasattr(loader, "__len__"): raise TypeError("Trainer requires a sized loader")
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.distributed.device, dtype=torch.float64)
        total_count = torch.zeros_like(total_loss)
        total_contrastive = torch.zeros_like(total_loss)
        total_positive_entropy = torch.zeros_like(total_loss)
        total_uncertainty = torch.zeros_like(total_loss)
        accumulated = torch.zeros_like(total_loss)
        for index, original in enumerate(loader):
            update = (index + 1) % self.config.gradient_accumulation_steps == 0 or index + 1 == len(loader)
            batch = _move(dict(original), self.distributed.device)
            pixels, labels = batch.pop("pixel_values_videos"), batch.pop("labels")
            temporal_valid_mask = batch.pop("temporal_valid_mask", None)
            original_ids = batch.get("video_id")
            if original_ids is None:
                raise KeyError("Batch must contain video_id values for same-original positive weighting")
            sync = self.model.no_sync() if hasattr(self.model, "no_sync") and not update else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=self.distributed.device.type, dtype=self.amp_dtype, enabled=self.distributed.device.type == "cuda" and self.amp_dtype != torch.float32):
                    output = self.model(pixels, temporal_valid_mask=temporal_valid_mask)
                    output, labels, original_ids, anchor_mask = gather_contrastive_batch(output, labels, original_ids)
                    result = self.loss(output=output, labels=labels, original_ids=original_ids, epoch=epoch, anchor_mask=anchor_mask)
                if result.loss_sum.ndim != 0 or not torch.isfinite(result.loss_sum):
                    raise FloatingPointError("Loss must return one finite scalar loss_sum")
                count = torch.as_tensor(result.sample_count, device=self.distributed.device, dtype=torch.float64)
                if count.ndim != 0 or count.item() < 0: raise ValueError("sample_count must be non-negative")
                self.scaler.scale(result.loss_sum).backward()
            accumulated += count
            total_loss += result.loss_sum.detach().double(); total_count += count
            total_contrastive += result.contrastive_sum.double()
            total_positive_entropy += result.positive_entropy_sum.double()
            total_uncertainty += result.uncertainty_sum.double()
            if update:
                global_count = all_reduce_sum(accumulated)
                if global_count.item() <= 0:
                    raise ValueError("Optimizer update requires at least one valid global anchor")
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None: parameter.grad.mul_(self.distributed.world_size / global_count.item())
                if self.config.max_grad_norm is not None: nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
                updated = self.scaler.get_scale() >= previous_scale
                if updated:
                    if self.scheduler is not None: self.scheduler.step()
                    self.global_step += 1
                accumulated.zero_()
        global_loss, global_count = all_reduce_sum(total_loss), all_reduce_sum(total_count)
        global_contrastive = all_reduce_sum(total_contrastive)
        global_positive_entropy = all_reduce_sum(total_positive_entropy)
        global_uncertainty = all_reduce_sum(total_uncertainty)
        positive_lambda, negative_temperature = self.loss.curriculum(epoch)
        return {
            "loss": float((global_loss / global_count).item()),
            "contrastive_loss": float((global_contrastive / global_count).item()),
            "contrastive_reference_entropy": float((global_positive_entropy / global_count).item()),
            "contrastive_excess_loss": float(((global_contrastive - global_positive_entropy) / global_count).item()),
            "uncertainty_loss": float((global_uncertainty / global_count).item()),
            "valid_anchors": float(global_count.item()),
            "positive_lambda": positive_lambda,
            "negative_temperature": negative_temperature,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
