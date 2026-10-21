from __future__ import annotations

import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.video_dataset import VideoManifestDataset, collate_video_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealManifestDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = PROJECT_ROOT / "dataset" / "manifest.jsonl"
        if not manifest.exists():
            raise unittest.SkipTest("Generated dataset is unavailable")
        cls.dataset = VideoManifestDataset(
            manifest,
            sampling=TemporalSamplingConfig(
                clip_length=64,
                target_fps=4,
                training=False,
                short_video_policy="repeat_last",
            ),
        )

    def test_item_shape_and_metadata(self) -> None:
        item = self.dataset[0]
        self.assertEqual(tuple(item["video"].shape[:2]), (64, 3))
        self.assertEqual(tuple(item["frame_indices"].shape), (64,))
        self.assertEqual(tuple(item["temporal_valid_mask"].shape), (64,))
        self.assertLess(int(item["temporal_valid_mask"].sum()), 64)
        self.assertIn(item["category"], {"clean", "cold", "warm"})
        self.assertEqual(item["label"], 0 if item["category"] == "clean" else 1)

    def test_batch_collation_with_variable_source_sizes(self) -> None:
        loader = DataLoader(
            self.dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_video_batch,
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch["video"]), 3)
        self.assertEqual(tuple(batch["frame_indices"].shape), (3, 64))
        self.assertEqual(tuple(batch["temporal_valid_mask"].shape), (3, 64))
        self.assertEqual(tuple(batch["labels"].shape), (3,))

    def test_worker_compatibility(self) -> None:
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            multiprocessing_context="spawn",
            collate_fn=collate_video_batch,
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch["video"]), 2)


if __name__ == "__main__":
    unittest.main()
