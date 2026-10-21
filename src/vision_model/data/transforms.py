"""Official Hugging Face V-JEPA 2 preprocessing boundary."""

from __future__ import annotations

from typing import Any

import torch

from .config import ThermalAugConfig


class ThermalAugmentor:
    """Full-support thermal-video views adapted from the supplied augmentor.

    One flip, brightness gain and contrast factor apply to the entire clip.
    Positive gains <= 1 and convex contrast mixing avoid clipping hot/cold
    extrema. Contrast uses each frame's mean, so future frames cannot affect
    an earlier augmented frame. No crop, erase or spatial deformation is used:
    without a spill mask even a peripheral eraser could remove the whole event.
    """

    def __init__(self, *, horizontal_flip_prob: float = .5,
                 brightness_range: tuple[float, float] = (.9, 1.),
                 contrast_range: tuple[float, float] = (.85, 1.)) -> None:
        if not 0 <= horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be in [0,1]")
        for name, limits in (("brightness", brightness_range), ("contrast", contrast_range)):
            if len(limits) != 2 or not 0 < limits[0] <= limits[1] <= 1:
                raise ValueError(f"{name} range must satisfy 0 < low <= high <= 1 to avoid clipping")
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    @property
    def policy(self) -> dict[str, Any]:
        return dict(name="full_frame_thermal_v1", horizontal_flip_prob=self.horizontal_flip_prob,
                    brightness_range=self.brightness_range, contrast_range=self.contrast_range,
                    crop=False, erase=False, elastic=False, clip_consistent=True)

    def sample(self, generator: torch.Generator | None = None) -> dict[str, Any]:
        draws = torch.rand(3, generator=generator).tolist()
        low_b, high_b = self.brightness_range
        low_c, high_c = self.contrast_range
        return dict(horizontal_flip=draws[0] < self.horizontal_flip_prob,
                    brightness=low_b + draws[1] * (high_b - low_b),
                    contrast=low_c + draws[2] * (high_c - low_c))

    @staticmethod
    def apply(frames: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
            raise ValueError("Thermal augmentation expects uint8 video [T,3,H,W]")
        brightness, contrast = float(parameters["brightness"]), float(parameters["contrast"])
        if not 0 < brightness <= 1 or not 0 < contrast <= 1:
            raise ValueError("Brightness and contrast must lie in (0,1]")
        value = frames.flip(-1) if parameters["horizontal_flip"] else frames
        if brightness == 1 and contrast == 1:
            return value.clone()
        value = value.float()
        mean = value.mean(dim=(-2, -1), keepdim=True)
        # Identity at brightness=contrast=1; monotone, no polarity inversion.
        value = brightness * (contrast * value + (1 - contrast) * mean)
        return value.round().clamp(0, 255).to(torch.uint8)

    def __call__(self, frames: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.apply(frames, self.sample(generator))


class OfficialVJEPA2Transform:
    """Apply the checkpoint's `AutoVideoProcessor` to a `[T,C,H,W]` clip."""

    def __init__(
        self,
        model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        *,
        thermal_config: ThermalAugConfig | None = None,
        preserve_field_of_view: bool = True,
        augmentation: ThermalAugmentor | None = None,
    ) -> None:
        try:
            from transformers import AutoVideoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers with V-JEPA 2 support is required; install the project model extra"
            ) from exc
        self.model_id = model_id
        self.preserve_field_of_view = preserve_field_of_view
        self.augmentation = augmentation
        if augmentation is not None and not preserve_field_of_view:
            raise ValueError("Spill-safe thermal augmentation requires full-frame preservation")
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.thermal_config = thermal_config
        if thermal_config is not None:
            crop_size = getattr(self.processor, "crop_size", None) or {}
            crop_height = crop_size.get("height") if isinstance(crop_size, dict) else None
            crop_width = crop_size.get("width") if isinstance(crop_size, dict) else None
            expected = thermal_config.image_size
            if (crop_height, crop_width) != (expected, expected):
                raise ValueError(
                    f"Thermal image_size={expected} does not match checkpoint crop "
                    f"{crop_height}x{crop_width}"
                )

    @property
    def normalization(self) -> dict[str, tuple[float, float, float]]:
        if self.thermal_config is not None:
            return {"mean": self.thermal_config.mean, "std": self.thermal_config.std}
        return {
            "mean": tuple(float(value) for value in self.processor.image_mean),
            "std": tuple(float(value) for value in self.processor.image_std),
        }

    def __call__(self, frames: Any, *, generator: torch.Generator | None = None) -> Any:
        if self.augmentation is not None:
            frames = self.augmentation(frames, generator=generator)
        processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.preserve_field_of_view:
            # Resize into the checkpoint grid without cropping away a possible
            # spill. Mean-valued uint8 padding is approximately zero after normalization.
            import torch
            import torch.nn.functional as F
            if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.dtype != torch.uint8:
                raise ValueError("Field-of-view preservation expects uint8 frames [T,C,H,W]")
            crop = self.processor.crop_size
            height, width = int(crop["height"]), int(crop["width"])
            source_height, source_width = frames.shape[-2:]
            scale = min(height / source_height, width / source_width)
            resized_height = max(1, min(height, round(source_height * scale)))
            resized_width = max(1, min(width, round(source_width * scale)))
            resized = F.interpolate(frames.float(), size=(resized_height, resized_width),
                                    mode="bilinear", align_corners=False, antialias=True)
            mean = torch.tensor(self.normalization["mean"], device=frames.device).view(1, 3, 1, 1) * 255
            canvas = mean.expand(len(frames), 3, height, width).clone()
            top, left = (height - resized_height) // 2, (width - resized_width) // 2
            canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
            frames = canvas.round().clamp(0, 255).to(torch.uint8)
            processor_kwargs.update(do_resize=False, do_center_crop=False)
        if self.thermal_config is not None:
            # The processor first rescales uint8 pixels to [0,1], then applies
            # these statistics. Passing them here replaces (not compounds)
            # the checkpoint's ImageNet normalization.
            processor_kwargs.update(
                do_normalize=True,
                image_mean=list(self.thermal_config.mean),
                image_std=list(self.thermal_config.std),
            )
        values = self.processor(frames, **processor_kwargs)["pixel_values_videos"]
        if values.ndim != 5 or values.shape[0] != 1:
            raise RuntimeError(f"Unexpected processor output shape: {tuple(values.shape)}")
        return values[0]
