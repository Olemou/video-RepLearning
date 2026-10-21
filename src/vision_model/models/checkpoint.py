"""Strict encoder checkpoint compatibility checks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class VJEPA2Architecture:
    frames_per_clip: int = 64
    crop_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    @property
    def num_tokens(self) -> int:
        return (self.frames_per_clip // self.tubelet_size) * (self.crop_size // self.patch_size) ** 2

def validate_architecture(config: Any, expected: VJEPA2Architecture) -> None:
    mismatches = []
    for field in expected.__dataclass_fields__:
        actual, wanted = getattr(config, field, None), getattr(expected, field)
        if actual != wanted:
            mismatches.append(f"{field}: expected {wanted!r}, got {actual!r}")
    if mismatches:
        raise RuntimeError("Incompatible V-JEPA 2 encoder configuration:\\n- " + "\\n- ".join(mismatches))

def validate_loading_info(info: dict[str, Any]) -> None:
    bad = {key: value for key, value in info.items() if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"} and value}
    if bad:
        raise RuntimeError("Pretrained checkpoint did not load exactly:\\n" + "\\n".join(f"- {key}: {value}" for key, value in bad.items()))
