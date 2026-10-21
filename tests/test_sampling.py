from __future__ import annotations

import unittest

import numpy as np

from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig


class TemporalSamplerTests(unittest.TestCase):
    def test_exact_64_frame_sampling(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(clip_length=64, target_fps=4, training=False)
        )
        indices = sampler.sample(num_frames=200, source_fps=10)
        self.assertEqual(indices.shape, (64,))
        self.assertTrue(np.all(indices[1:] >= indices[:-1]))
        self.assertGreater(indices[-1], indices[0])

    def test_repeat_last_short_video(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(
                clip_length=64,
                target_fps=4,
                training=False,
                short_video_policy="repeat_last",
            )
        )
        indices = sampler.sample(num_frames=50, source_fps=10)
        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[-1], 49)
        self.assertGreater(np.count_nonzero(indices == 49), 1)

    def test_loop_short_video(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(
                clip_length=64,
                target_fps=10,
                training=False,
                short_video_policy="loop",
            )
        )
        indices = sampler.sample(num_frames=10, source_fps=10)
        np.testing.assert_array_equal(indices[:12], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]))

    def test_error_short_video(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(short_video_policy="error")
        )
        with self.assertRaisesRegex(ValueError, "require"):
            sampler.sample(num_frames=50, source_fps=10)

    def test_validation_is_deterministic(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(training=False), seed=100
        )
        first = sampler.sample(num_frames=300, source_fps=10, sample_index=5)
        sampler.set_epoch(9)
        second = sampler.sample(num_frames=300, source_fps=10, sample_index=5)
        np.testing.assert_array_equal(first, second)

    def test_training_changes_by_epoch_but_repeats_within_epoch(self) -> None:
        sampler = TemporalSampler(
            TemporalSamplingConfig(training=True), seed=100
        )
        first = sampler.sample(num_frames=300, source_fps=10, sample_index=5)
        repeat = sampler.sample(num_frames=300, source_fps=10, sample_index=5)
        np.testing.assert_array_equal(first, repeat)
        sampler.set_epoch(1)
        next_epoch = sampler.sample(num_frames=300, source_fps=10, sample_index=5)
        self.assertFalse(np.array_equal(first, next_epoch))


if __name__ == "__main__":
    unittest.main()
