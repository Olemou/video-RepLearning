"""Validated configuration for thermal-video preprocessing and augmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalAugConfig:
    """Legacy parameter schema and optional normalization override.

    V-JEPA 2 ViT-L was loaded at a 256-pixel native crop, so this project uses
    256 rather than the generic 224-pixel value from the source augmentor.
    These fields do not execute augmentation. The supported spill-preserving
    video policy is ThermalAugmentor in transforms.py, configured explicitly.
    """

    image_size: int = 256

    mask_width_ratio: float = 0.6
    mask_height_ratio: float = 0.2
    max_attempts: int = 5
    erase_prob: float = 0.9

    random_affine_degrees: int = 10
    random_rotation_degrees: int = 5
    resized_crop_scale: tuple[float, float] = (0.85, 1.0)
    resized_crop_ratio: tuple[float, float] = (0.75, 1.25)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3
    degrees: int = 10
    translate: tuple[float, float] = (0.1, 0.1)
    scale: tuple[float, float] = (0.85, 1.15)
    shear: int = 10

    random_affine_prob: float = 0.3
    random_rotation_degrees_prob: float = 0.5
    random_crop_prob: float = 0.5
    occlusion_prob: float = 0.4

    mean: tuple[float, float, float] = (0.24, 0.24, 0.24)
    std: tuple[float, float, float] = (0.07, 0.07, 0.07)

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for name in ("mask_width_ratio", "mask_height_ratio"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "erase_prob",
            "horizontal_flip_prob",
            "vertical_flip_prob",
            "random_affine_prob",
            "random_rotation_degrees_prob",
            "random_crop_prob",
            "occlusion_prob",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("resized_crop_scale", "resized_crop_ratio", "scale"):
            low, high = getattr(self, name)
            if low <= 0 or high < low:
                raise ValueError(f"{name} must contain positive ordered bounds")
        if any(value < 0 for value in self.mean):
            raise ValueError("normalization means must be non-negative")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization standard deviations must be positive")

