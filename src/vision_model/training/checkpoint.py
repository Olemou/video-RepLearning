"""Atomic supervised encoder checkpoint serialization."""
from __future__ import annotations
import os, random, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

def _rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def save_checkpoint(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any | None, scaler: Any | None, epoch: int, global_step: int, loss_name: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format_version": 2, "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "scaler": None if scaler is None else scaler.state_dict(), "epoch": int(epoch), "global_step": int(global_step), "loss_name": loss_name, "rng_state": _rng()}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent); os.close(descriptor)
    try:
        torch.save(state, temporary); os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def load_checkpoint(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any | None, scaler: Any | None, expected_loss_name: str) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 2: raise RuntimeError(f"Unsupported checkpoint format: {state.get('format_version')}")
    if state.get("loss_name") != expected_loss_name: raise RuntimeError("Loss name mismatch")
    unwrap(model).load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None: scheduler.load_state_dict(state["scheduler"])
    if scaler is not None: scaler.load_state_dict(state["scaler"])
    rng = state["rng_state"]; random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng["cuda"])
    return int(state["epoch"]), int(state["global_step"])
