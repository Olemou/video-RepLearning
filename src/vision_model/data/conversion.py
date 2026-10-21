"""Deterministic conversion of image-sequence directories to H.264 MP4."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError
from .labels import LABEL_MAP

LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("clean", "cold", "warm")
DEFAULT_LABEL_MAP = LABEL_MAP
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
_NATURAL_PART = re.compile(r"(\d+)")


class ConversionError(RuntimeError):
    """Raised when a sequence cannot be converted without losing frames."""


@dataclass(frozen=True)
class SequenceSpec:
    category: str
    directory: Path
    relative_directory: Path
    frames: tuple[Path, ...]
    output_path: Path
    source_sequence: Path | None = None


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    category: str
    label: int
    video_path: str
    source_sequence: str
    num_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class ConversionFailure:
    category: str
    source_sequence: str
    output_path: str | None
    error: str


def natural_key(value: str | Path) -> tuple[object, ...]:
    """Return a stable, case-insensitive key with numeric runs as integers."""

    text = str(value).replace(os.sep, "/")
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART.split(text)
    )


def image_files(directory: Path) -> tuple[Path, ...]:
    """List direct child image files in natural temporal order."""

    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: natural_key(path.name),
        )
    )


def discover_sequences(
    input_root: Path,
    output_root: Path,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    recursive: bool = True,
) -> list[SequenceSpec]:
    """Discover every directory containing direct image children.

    Relative subdirectories are preserved beneath each category, preventing
    two equally named nested sequences from mapping to the same output.
    """

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    discovered: list[SequenceSpec] = []
    output_owners: dict[str, Path] = {}

    for category in categories:
        category_root = input_root / category
        if not category_root.is_dir():
            LOGGER.warning("Category directory is missing: %s", category_root)
            continue
        candidates: Iterable[Path]
        candidates = category_root.rglob("*") if recursive else category_root.iterdir()
        directories = [path for path in candidates if path.is_dir()]
        directories.sort(key=lambda path: natural_key(path.relative_to(category_root)))

        for directory in directories:
            frames = image_files(directory)
            if not frames:
                continue
            relative = directory.relative_to(category_root)
            output = output_root / category / relative.parent / f"{relative.name}.mp4"
            collision_key = str(output).casefold()
            previous = output_owners.get(collision_key)
            if previous is not None:
                raise ConversionError(
                    f"Output collision: {previous} and {directory} both map to {output}"
                )
            output_owners[collision_key] = directory
            discovered.append(
                SequenceSpec(category, directory, relative, frames, output)
            )

    return discovered


def discover_flat_sequences(
    input_root: Path,
    output_root: Path,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    label_map: dict[str, int] | None = None,
) -> list[SequenceSpec]:
    """Discover flat sequence folders labeled by ``spill_``/``no_spill_`` filenames."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    labels = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
    category_for_label: dict[int, str] = {}
    for category in categories:
        if category in labels:
            category_for_label.setdefault(labels[category], category)
    if set(category_for_label) != {0, 1}:
        raise ConversionError("Flat image folders require categories mapped to both labels 0 and 1")

    discovered: list[SequenceSpec] = []
    output_owners: dict[str, Path] = {}
    directories = sorted(
        (path for path in input_root.rglob("*") if path.is_dir()),
        key=lambda path: natural_key(path.relative_to(input_root)),
    )
    for directory in directories:
        if directory == output_root or output_root in directory.parents:
            continue
        frames = image_files(directory)
        if not frames:
            continue

        frame_labels: set[int] = set()
        for frame in frames:
            name = frame.stem.casefold()
            if name.startswith("no_spill_"):
                frame_labels.add(0)
            elif name.startswith("spill_"):
                frame_labels.add(1)
            else:
                raise ConversionError(
                    f"Cannot infer clean/spill label from image filename {frame.name}; "
                    "expected no_spill_ or spill_ prefix"
                )
        if len(frame_labels) != 1:
            raise ConversionError(
                f"Mixed clean and spill images in {directory}; split the sequence folders "
                "before conversion"
            )

        category = category_for_label[frame_labels.pop()]
        relative = directory.relative_to(input_root)
        output = output_root / category / relative.parent / f"{relative.name}.mp4"
        collision_key = str(output).casefold()
        previous = output_owners.get(collision_key)
        if previous is not None:
            raise ConversionError(
                f"Output collision: {previous} and {directory} both map to {output}"
            )
        output_owners[collision_key] = directory
        discovered.append(
            SequenceSpec(category, directory, relative, frames, output, relative)
        )
    return discovered


def find_empty_leaf_directories(
    input_root: Path, categories: Sequence[str] = DEFAULT_CATEGORIES
) -> list[Path]:
    """Return leaf directories without supported image frames."""

    empty: list[Path] = []
    for category in categories:
        category_root = input_root / category
        if not category_root.is_dir():
            continue
        for directory in sorted(category_root.rglob("*"), key=natural_key):
            if directory.is_dir() and not any(child.is_dir() for child in directory.iterdir()):
                if not image_files(directory):
                    empty.append(directory)
    return empty


def _checked_run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ConversionError(f"Required executable is missing: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no subprocess output").strip()
        raise ConversionError(f"{description} failed: {detail}") from exc


def _probe(path: Path, ffprobe: str = "ffprobe") -> dict[str, object]:
    result = _checked_run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        f"Probing {path}",
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ConversionError(f"Expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def _parse_rate(value: object) -> float:
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as exc:
        raise ConversionError(f"Invalid video frame rate: {value!r}") from exc


def validate_video(
    path: Path,
    *,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
    expected_size: tuple[int, int] | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    decode_all: bool = True,
) -> dict[str, object]:
    """Probe and optionally decode a video, enforcing expected metadata."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ConversionError(f"Video is missing or empty: {path}")
    stream = _probe(path, ffprobe=ffprobe)
    codec = str(stream.get("codec_name", ""))
    if codec != "h264":
        raise ConversionError(f"Expected H.264 video, got {codec!r}: {path}")

    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(str(raw_count))
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"Unable to determine frame count for {path}") from exc
    fps = _parse_rate(stream.get("avg_frame_rate"))
    width = int(stream["width"])
    height = int(stream["height"])

    if frame_count <= 0:
        raise ConversionError(f"Video has no frames: {path}")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ConversionError(f"Invalid H.264 dimensions {width}x{height}: {path}")
    if expected_frames is not None and frame_count != expected_frames:
        raise ConversionError(
            f"Frame-count mismatch for {path}: expected {expected_frames}, got {frame_count}"
        )
    if expected_fps is not None and abs(fps - expected_fps) > 1e-6:
        raise ConversionError(f"FPS mismatch for {path}: expected {expected_fps}, got {fps}")
    if expected_size is not None and (width, height) != expected_size:
        raise ConversionError(
            f"Dimension mismatch for {path}: expected {expected_size}, got {(width, height)}"
        )
    if decode_all:
        _checked_run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
            f"Decoding {path}",
        )

    duration = frame_count / fps
    return {
        "num_frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "codec": codec,
    }


def _inspect_images(frames: Sequence[Path]) -> list[tuple[int, int]]:
    dimensions: list[tuple[int, int]] = []
    for frame in frames:
        try:
            with Image.open(frame) as image:
                image.verify()
            with Image.open(frame) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                dimensions.append(image.size)
        except (OSError, UnidentifiedImageError) as exc:
            raise ConversionError(f"Unreadable image {frame}: {exc}") from exc
    return dimensions


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _target_size(
    dimensions: Sequence[tuple[int, int]], width: int | None, height: int | None
) -> tuple[int, int]:
    if (width is None) != (height is None):
        raise ConversionError("--width and --height must be supplied together")
    if width is None:
        width, height = dimensions[0]
        width, height = _even(width), _even(height)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ConversionError(
            f"Output dimensions must be positive and even for yuv420p, got {width}x{height}"
        )
    return width, height


def _normalize_image(
    source: Path,
    destination: Path,
    target: tuple[int, int],
    policy: str,
) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if policy == "error":
            if image.size != target:
                raise ConversionError(
                    f"Inconsistent dimensions for {source}: {image.size}, expected {target}"
                )
            normalized = image
        elif policy == "stretch":
            normalized = image.resize(target, Image.Resampling.LANCZOS)
        elif policy == "letterbox":
            normalized = ImageOps.pad(
                image, target, method=Image.Resampling.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5)
            )
        elif policy == "crop":
            normalized = ImageOps.fit(
                image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
        else:
            raise ConversionError(f"Unknown resize policy: {policy}")
        normalized.save(destination, format="PNG", compress_level=1)


def convert_sequence(
    spec: SequenceSpec,
    output_root: Path,
    *,
    fps: float,
    codec: str = "libx264",
    resize_policy: str = "letterbox",
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
    validate: bool = True,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    label_map: dict[str, int] | None = None,
) -> VideoRecord:
    """Convert exactly one sequence and return validated manifest metadata."""

    if not spec.frames:
        raise ConversionError(f"No supported image frames in {spec.directory}")
    if fps <= 0:
        raise ConversionError(f"FPS must be positive, got {fps}")

    dimensions = _inspect_images(spec.frames)
    target = _target_size(dimensions, width, height)
    if resize_policy == "error" and any(size != target for size in dimensions):
        unique = sorted(set(dimensions))
        raise ConversionError(f"Inconsistent sequence dimensions: {unique}")

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.output_path.exists() and not overwrite:
        metadata = validate_video(
            spec.output_path,
            expected_frames=len(spec.frames),
            expected_fps=fps,
            expected_size=target,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            decode_all=validate,
        )
        LOGGER.info("Keeping validated existing video: %s", spec.output_path)
    else:
        with tempfile.TemporaryDirectory(prefix="vjepa-frames-") as temp_name:
            temp = Path(temp_name)
            for index, source in enumerate(spec.frames):
                _normalize_image(
                    source,
                    temp / f"{index:08d}.png",
                    target,
                    resize_policy,
                )
            temporary_output = spec.output_path.with_name(
                f".{spec.output_path.name}.partial.mp4"
            )
            temporary_output.unlink(missing_ok=True)
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                str(temp / "%08d.png"),
                "-frames:v",
                str(len(spec.frames)),
                "-an",
                "-c:v",
                codec,
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-map_metadata",
                "-1",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            try:
                _checked_run(command, f"Encoding {spec.directory}")
                metadata = validate_video(
                    temporary_output,
                    expected_frames=len(spec.frames),
                    expected_fps=fps,
                    expected_size=target,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    decode_all=validate,
                )
                os.replace(temporary_output, spec.output_path)
            finally:
                temporary_output.unlink(missing_ok=True)

    relative_video = spec.output_path.resolve().relative_to(output_root.resolve())
    source_sequence = spec.source_sequence or Path(spec.category) / spec.relative_directory
    video_id = "_".join((Path(spec.category) / spec.relative_directory).parts)
    return VideoRecord(
        video_id=video_id,
        category=spec.category,
        label=(label_map or DEFAULT_LABEL_MAP)[spec.category],
        video_path=relative_video.as_posix(),
        source_sequence=source_sequence.as_posix(),
        num_frames=int(metadata["num_frames"]),
        fps=float(metadata["fps"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        duration_seconds=float(metadata["duration_seconds"]),
    )


def _atomic_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def convert_dataset(
    input_root: Path,
    output_root: Path,
    *,
    fps: float,
    codec: str = "libx264",
    recursive: bool = True,
    resize_policy: str = "letterbox",
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
    validate: bool = True,
    manifest_path: Path | None = None,
    errors_path: Path | None = None,
    fail_fast: bool = False,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    label_map: dict[str, int] | None = None,
) -> tuple[list[VideoRecord], list[ConversionFailure]]:
    """Convert a category tree and atomically write manifest and failures."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise ConversionError(f"Input root does not exist: {input_root}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ConversionError("FFmpeg and FFprobe must both be available on PATH")

    categories = tuple(dict.fromkeys(categories))
    label_map = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
    if not categories:
        raise ConversionError("At least one category must be configured")
    missing_labels = set(categories) - set(label_map)
    if missing_labels:
        raise ConversionError(f"Missing labels for categories: {sorted(missing_labels)}")
    if any(type(label_map[name]) is not int or label_map[name] not in (0, 1) for name in categories):
        raise ConversionError("Every category label must be integer 0 or 1")
    sequences = (
        discover_sequences(
            input_root, output_root, categories=categories, recursive=recursive
        )
        if any((input_root / category).is_dir() for category in categories)
        else []
    )
    if not sequences:
        sequences = discover_flat_sequences(
            input_root, output_root, categories=categories, label_map=label_map
        )
        if sequences:
            LOGGER.info(
                "Using flat image-folder layout; inferred labels from spill/no_spill filenames"
            )
    if not sequences:
        raise ConversionError(
            f"No category-based or labeled flat image-sequence directories found in {input_root}"
        )
    empty = find_empty_leaf_directories(input_root, categories=categories)
    for directory in empty:
        LOGGER.warning("Empty or unsupported leaf sequence skipped: %s", directory)

    records: list[VideoRecord] = []
    failures: list[ConversionFailure] = []
    total = len(sequences)
    for index, spec in enumerate(sequences, start=1):
        LOGGER.info("[%d/%d] Converting %s", index, total, spec.directory)
        try:
            records.append(
                convert_sequence(
                    spec,
                    output_root,
                    fps=fps,
                    codec=codec,
                    resize_policy=resize_policy,
                    width=width,
                    height=height,
                    overwrite=overwrite,
                    validate=validate,
                    label_map=label_map,
                )
            )
        except Exception as exc:  # record each failed sequence without hiding it
            failure = ConversionFailure(
                category=spec.category,
                source_sequence=(Path(spec.category) / spec.relative_directory).as_posix(),
                output_path=str(spec.output_path),
                error=f"{type(exc).__name__}: {exc}",
            )
            failures.append(failure)
            LOGGER.error("Failed %s: %s", spec.directory, exc)
            if fail_fast:
                break

    manifest_path = manifest_path or output_root / "manifest.jsonl"
    errors_path = errors_path or output_root / "conversion_errors.jsonl"
    video_ids = [record.video_id for record in records]
    if len(video_ids) != len(set(video_ids)):
        duplicates = sorted({value for value in video_ids if video_ids.count(value) > 1})
        raise ConversionError(f"Duplicate generated video IDs: {duplicates}")
    _atomic_jsonl(manifest_path, records)
    _atomic_jsonl(errors_path, failures)
    return records, failures
