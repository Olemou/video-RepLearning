from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from vision_model.data.config import ThermalAugConfig
from vision_model.data.transforms import OfficialVJEPA2Transform, ThermalAugmentor


class FakeProcessor:
    crop_size = {"height": 256, "width": 256}
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, frames, **kwargs):
        self.kwargs = kwargs
        self.frames = frames
        return {"pixel_values_videos": torch.zeros(1, len(frames), 3, 256, 256)}


class TransformTests(unittest.TestCase):
    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_thermal_statistics_require_explicit_override(self, load_processor) -> None:
        processor = FakeProcessor()
        load_processor.return_value = processor
        transform = OfficialVJEPA2Transform(thermal_config=ThermalAugConfig())
        output = transform(torch.zeros(4, 3, 32, 32, dtype=torch.uint8))

        self.assertEqual(tuple(output.shape), (4, 3, 256, 256))
        self.assertEqual(processor.kwargs["image_mean"], [0.24, 0.24, 0.24])
        self.assertEqual(processor.kwargs["image_std"], [0.07, 0.07, 0.07])
        self.assertTrue(processor.kwargs["do_normalize"])
        self.assertEqual(transform.normalization["mean"], (0.24, 0.24, 0.24))

    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_checkpoint_normalization_is_the_default(self, load_processor) -> None:
        processor = FakeProcessor()
        load_processor.return_value = processor
        transform = OfficialVJEPA2Transform()
        transform(torch.zeros(2, 3, 32, 32, dtype=torch.uint8))

        self.assertNotIn("image_mean", processor.kwargs)
        self.assertEqual(transform.normalization["mean"], (0.485, 0.456, 0.406))

    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_letterbox_preserves_frame_edges_and_disables_center_crop(self, load_processor) -> None:
        processor = FakeProcessor()
        load_processor.return_value = processor
        transform = OfficialVJEPA2Transform(preserve_field_of_view=True)
        frames = torch.zeros(2, 3, 128, 256, dtype=torch.uint8)
        frames[:, :, :, :8] = 255
        frames[:, :, :, -8:] = 255
        transform(frames)
        self.assertEqual(tuple(processor.frames.shape), (2, 3, 256, 256))
        self.assertTrue((processor.frames[:, :, 64:192, :8] == 255).all())
        self.assertTrue((processor.frames[:, :, 64:192, -8:] == 255).all())
        self.assertEqual(processor.kwargs["do_resize"], False)
        self.assertEqual(processor.kwargs["do_center_crop"], False)
        self.assertTrue(torch.equal(processor.frames[0, :, 0, 0], torch.tensor([124, 116, 104], dtype=torch.uint8)))

    def test_rejects_invalid_standard_deviation(self) -> None:
        with self.assertRaisesRegex(ValueError, "standard deviations"):
            ThermalAugConfig(std=(0.07, 0.0, 0.07))

    def test_thermal_identity_and_clip_consistency(self) -> None:
        frame = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(1, 3, -1, -1)
        clip = frame.repeat(5, 1, 1, 1)
        augment = ThermalAugmentor()
        parameters = dict(horizontal_flip=False, brightness=1., contrast=1.)
        self.assertTrue(torch.equal(augment.apply(clip, parameters), clip))
        output = augment(clip, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(output, output[:1].expand_as(output)))
        self.assertEqual(output.shape, clip.shape)
        self.assertEqual(output.dtype, torch.uint8)

    def test_thermal_augmentation_keeps_floor_and_all_edges(self) -> None:
        clip = torch.full((2, 3, 16, 24), 100, dtype=torch.uint8)
        clip[:, :, -2:, 8:16] = 220  # floor spill, never protected by a center-only eraser
        clip[:, :, 0, :] = 30
        parameters = dict(horizontal_flip=True, brightness=.9, contrast=.85)
        result = ThermalAugmentor.apply(clip, parameters)
        self.assertTrue((result[:, :, -2:, 8:16] > result[:, :, 8:10, 8:16]).all())
        self.assertTrue((result[:, :, 0] < result[:, :, 1]).all())
        self.assertTrue((result > 0).all())
        self.assertTrue((result < 255).all())

    def test_thermal_prefix_does_not_depend_on_future_frames(self) -> None:
        clip = torch.randint(0, 256, (8, 3, 12, 20), dtype=torch.uint8)
        augment = ThermalAugmentor()
        short = augment(clip[:2], generator=torch.Generator().manual_seed(11))
        long = augment(clip, generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(short, long[:2]))

    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_thermal_transform_default_full_frame_and_replay(self, load_processor) -> None:
        processor = FakeProcessor()
        load_processor.return_value = processor
        augment = ThermalAugmentor(horizontal_flip_prob=1, brightness_range=(1, 1), contrast_range=(1, 1))
        transform = OfficialVJEPA2Transform(augmentation=augment)
        clip = torch.zeros(2, 3, 128, 256, dtype=torch.uint8)
        clip[:, :, -8:, :8] = 255
        transform(clip, generator=torch.Generator().manual_seed(1))
        self.assertTrue((processor.frames[:, :, 184:192, -8:] == 255).all())
        self.assertFalse(processor.kwargs["do_center_crop"])
        with self.assertRaisesRegex(ValueError, "full-frame"):
            OfficialVJEPA2Transform(augmentation=augment, preserve_field_of_view=False)

    def test_thermal_augmentation_rejects_polarity_inversion_and_clipping(self) -> None:
        for kwargs in (dict(contrast_range=(-1, 1)), dict(brightness_range=(.8, 1.4)),
                       dict(horizontal_flip_prob=1.5)):
            with self.assertRaises(ValueError):
                ThermalAugmentor(**kwargs)

    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_rejects_crop_size_that_changes_token_grid(self, load_processor) -> None:
        load_processor.return_value = FakeProcessor()
        with self.assertRaisesRegex(ValueError, "does not match checkpoint crop"):
            OfficialVJEPA2Transform(thermal_config=ThermalAugConfig(image_size=224))


if __name__ == "__main__":
    unittest.main()
