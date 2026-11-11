"""Deterministic, source-disjoint trajectory hyperparameter search utilities."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict

import torch


def source_split(labels: torch.Tensor, source_sequences: list[str], *, seed: int,
                 validation_per_class: int) -> tuple[list[int], list[int]]:
    """Hold out complete source groups, with exactly the same count per class."""
    if len(labels) != len(source_sequences) or validation_per_class < 2:
        raise ValueError("Invalid labels, source groups, or validation count")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(source_sequences):
        groups[source].append(index)
    by_class = {0: [], 1: []}
    for source, indices in groups.items():
        classes = {int(labels[index]) for index in indices}
        if len(classes) != 1:
            raise ValueError(f"Source group {source} has conflicting class labels")
        by_class[classes.pop()].append((source, indices))
    selected: set[int] = set()
    for label in (0, 1):
        entries = sorted(by_class[label])
        random.Random(seed + label * 1009).shuffle(entries)
        count = 0
        for _, indices in entries:
            if count + len(indices) <= validation_per_class:
                selected.update(indices)
                count += len(indices)
            if count == validation_per_class:
                break
        if count != validation_per_class:
            raise ValueError("Cannot create an exact class-balanced source-group validation split")
    training = [index for index in range(len(labels)) if index not in selected]
    validation = sorted(selected)
    train_counts = torch.bincount(labels[training], minlength=2)
    if train_counts[0] != train_counts[1] or len(training) < 32:
        raise ValueError("Training split needs at least 32 balanced source videos")
    return training, validation


def batch_plan(batch_size: int, world_size: int, accumulation_steps: int) -> dict:
    if world_size < 1 or accumulation_steps < 1 or batch_size < 32:
        raise ValueError("Global effective batch size must be at least 32")
    if batch_size % (2 * world_size * accumulation_steps):
        raise ValueError("Batch must divide into equal class-balanced rank microbatches")
    micro = batch_size // (world_size * accumulation_steps)
    return dict(global_source_batch=batch_size, per_rank_microbatch=micro,
                world_size=world_size, accumulation_steps=accumulation_steps,
                global_trajectory_views=2 * batch_size)


def scaled_lr(base_lr: float, batch_size: int, *, reference_batch: int = 32,
              policy: str = "constant") -> float:
    powers = {"constant": 0.0, "sqrt": 0.5, "linear": 1.0}
    if policy not in powers or base_lr <= 0 or batch_size < 32:
        raise ValueError("Invalid learning-rate scaling policy or batch")
    return base_lr * (batch_size / reference_batch) ** powers[policy]


def warmup_cosine(step: int, total_steps: int, peak_lr: float, *,
                  warmup_fraction: float = .1, start_fraction: float = .1,
                  final_fraction: float = .05) -> float:
    if not 1 <= step <= total_steps or not 0 <= warmup_fraction < 1:
        raise ValueError("Invalid schedule step or warmup")
    warmup = max(1, min(total_steps - 1, math.ceil(total_steps * warmup_fraction))) if total_steps > 1 else 0
    if step <= warmup:
        return peak_lr * (start_fraction + (1 - start_fraction) * step / warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak_lr * (final_fraction + (1 - final_fraction) * (1 + math.cos(math.pi * progress)) / 2)


def cache_identity(cache: dict) -> str:
    """Cheap provenance fingerprint; callers also record the file's full SHA-256."""
    data = "\n".join(f"{i}:{digest}:{label}" for i, digest, label in zip(
        cache["ids"], cache["video_sha256"], cache["labels"].tolist()))
    return hashlib.sha256(data.encode()).hexdigest()
