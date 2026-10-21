from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vision_model.data.conversion import (
    ConversionError,
    convert_dataset,
    discover_flat_sequences,
    discover_sequences,
    find_empty_leaf_directories,
    image_files,
    natural_key,
    validate_video,
)
from vision_model.data.validation import validate_dataset


def make_image(path: Path, size: tuple[int, int] = (32, 24), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class DiscoveryTests(unittest.TestCase):
    def test_natural_sort_keeps_numeric_frame_order(self) -> None:
        names = ["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]
        self.assertEqual(sorted(names, key=natural_key), ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"])

    def test_missing_numeric_names_are_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            sequence = Path(name)
            make_image(sequence / "frame_1.jpg")
            make_image(sequence / "frame_3.jpg")
            self.assertEqual([p.name for p in image_files(sequence)], ["frame_1.jpg", "frame_3.jpg"])

    def test_empty_leaf_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            empty = root / "clean" / "empty_sequence"
            empty.mkdir(parents=True)
            self.assertEqual(find_empty_leaf_directories(root), [empty])

    def test_nested_duplicate_names_preserve_parent_path(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            output = Path(output_name)
            make_image(source / "clean" / "camera_a" / "sequence" / "1.png")
            make_image(source / "clean" / "camera_b" / "sequence" / "1.png")
            specs = discover_sequences(source, output)
            outputs = {spec.output_path.relative_to(output).as_posix() for spec in specs}
            self.assertEqual(
                outputs,
                {"clean/camera_a/sequence.mp4", "clean/camera_b/sequence.mp4"},
            )

    def test_flat_sequences_infer_category_from_frame_names(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            clean = source / "videoFrame1"
            spill = source / "folderImage2"
            make_image(clean / "no_spill_000002.jpg")
            make_image(clean / "no_spill_000001.jpg")
            make_image(spill / "spill_000002.png")
            make_image(spill / "spill_000001.png")

            specs = discover_flat_sequences(source, Path(output_name))

            self.assertEqual(
                [(spec.category, spec.source_sequence.as_posix()) for spec in specs],
                [("cold", "folderImage2"), ("clean", "videoFrame1")],
            )
            self.assertEqual(
                {spec.output_path.relative_to(Path(output_name)).as_posix() for spec in specs},
                {"cold/folderImage2.mp4", "clean/videoFrame1.mp4"},
            )

    def test_flat_sequence_rejects_mixed_frame_labels(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            make_image(source / "sequence" / "spill_000001.png")
            make_image(source / "sequence" / "no_spill_000002.png")

            with self.assertRaisesRegex(ConversionError, "Mixed clean and spill"):
                discover_flat_sequences(source, Path(output_name))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class EncodingTests(unittest.TestCase):
    def test_corrupt_image_is_a_recorded_failure(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            sequence = source / "warm" / "broken"
            sequence.mkdir(parents=True)
            (sequence / "frame_1.jpg").write_bytes(b"not an image")
            records, failures = convert_dataset(source, Path(output_name), fps=5)
            self.assertEqual(records, [])
            self.assertEqual(len(failures), 1)
            self.assertIn("Unreadable image", failures[0].error)

    def test_inconsistent_dimensions_fail_under_error_policy(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            make_image(source / "cold" / "mixed" / "1.png", (32, 24))
            make_image(source / "cold" / "mixed" / "2.png", (40, 30))
            records, failures = convert_dataset(
                source, Path(output_name), fps=5, resize_policy="error"
            )
            self.assertEqual(records, [])
            self.assertEqual(len(failures), 1)
            self.assertIn("Inconsistent sequence dimensions", failures[0].error)

    def test_encode_manifest_decode_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            output = Path(output_name)
            for index in (1, 2, 10):
                make_image(
                    source / "clean" / "nested" / "sequence_1" / f"frame_{index}.jpg",
                    (31 + index % 2, 23 + index % 2),
                    (index, 20, 30),
                )
            records, failures = convert_dataset(
                source,
                output,
                fps=6,
                resize_policy="letterbox",
                width=32,
                height=24,
            )
            self.assertFalse(failures)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.num_frames, 3)
            self.assertEqual(record.label, 0)
            self.assertEqual(record.video_path, "clean/nested/sequence_1.mp4")
            metadata = validate_video(
                output / record.video_path,
                expected_frames=3,
                expected_fps=6,
                expected_size=(32, 24),
            )
            self.assertEqual(metadata["codec"], "h264")
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["source_sequence"], "clean/nested/sequence_1")
            self.assertEqual(rows[0]["label"], 0)
            summary = validate_dataset(
                output,
                output / "manifest.jsonl",
                source_root=source,
                short_threshold=2,
            )
            self.assertTrue(summary.ok)
            self.assertEqual(summary.valid_videos, 1)

    def test_convert_flat_folders_and_validate_original_sources(self) -> None:
        with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
            source = Path(source_name)
            output = Path(output_name)
            for index in (1, 2):
                make_image(source / "videoFrame1" / f"no_spill_{index:06d}.jpg")
                make_image(source / "folderImage2" / f"spill_{index:06d}.png")

            records, failures = convert_dataset(source, output, fps=5)

            self.assertFalse(failures)
            self.assertEqual(
                {(record.category, record.label, record.source_sequence) for record in records},
                {
                    ("clean", 0, "videoFrame1"),
                    ("cold", 1, "folderImage2"),
                },
            )
            summary = validate_dataset(
                output,
                output / "manifest.jsonl",
                source_root=source,
                short_threshold=0,
            )
            self.assertTrue(summary.ok)
            self.assertEqual(summary.expected_sequences, 2)

if __name__ == "__main__":
    unittest.main()
