"""AdamW parameter grouping and serializable schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 5.25e-4
    weight_decay: float = 0.04
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_steps: int = 0
    total_steps: int = 1000
    final_learning_rate: float = 1e-6


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise ValueError("No trainable parameters")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.total_steps <= 0 or config.warmup_steps < 0:
        raise ValueError("Invalid scheduler step counts")
    initial_lr = optimizer.param_groups[0]["lr"]
    final_ratio = config.final_learning_rate / initial_lr

    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-12, step / config.warmup_steps)
        duration = max(1, config.total_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / duration))
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
