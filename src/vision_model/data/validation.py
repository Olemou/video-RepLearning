"""Manifest-driven MP4 dataset validation and Markdown reporting."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from .labels import LABEL_MAP

from .conversion import (
    DEFAULT_CATEGORIES,
    ConversionError,
    discover_flat_sequences,
    discover_sequences,
    validate_video,
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    subject: str
    message: str


@dataclass(frozen=True)
class ValidationSummary:
    expected_sequences: int | None
    manifest_records: int
    valid_videos: int
    failed_videos: int
    short_videos: int
    category_counts: dict[str, int]
    resolution_counts: dict[str, int]
    fps_counts: dict[str, int]
    frame_counts: tuple[int, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            required = {
                "video_id",
                "category",
                "label",
                "video_path",
                "source_sequence",
                "num_frames",
                "fps",
                "width",
                "height",
                "duration_seconds",
            }
            missing = required - row.keys()
            if missing:
                raise ConversionError(f"Missing fields at {path}:{line_number}: {sorted(missing)}")
            video_id = str(row["video_id"])
            video_path = str(row["video_path"])
            if video_id in seen_ids:
                raise ConversionError(f"Duplicate video_id in manifest: {video_id}")
            if video_path.casefold() in seen_paths:
                raise ConversionError(f"Duplicate video_path in manifest: {video_path}")
            seen_ids.add(video_id)
            seen_paths.add(video_path.casefold())
            rows.append(row)
    return rows


def validate_dataset(
    dataset_root: Path,
    manifest_path: Path,
    *,
    source_root: Path | None = None,
    short_threshold: int = 64,
    decode_all: bool = True,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
) -> ValidationSummary:
    dataset_root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    rows = load_manifest(manifest_path)
    issues: list[ValidationIssue] = []
    counts: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    frame_rates: Counter[str] = Counter()
    frame_counts: list[int] = []
    short_count = 0
    valid_count = 0

    expected_sources: set[str] | None = None
    if source_root is not None:
        specs = (
            discover_sequences(
                source_root.resolve(), dataset_root, categories=categories, recursive=True
            )
            if any((source_root / category).is_dir() for category in categories)
            else []
        )
        if not specs:
            specs = discover_flat_sequences(
                source_root.resolve(),
                dataset_root,
                categories=categories,
                label_map=LABEL_MAP,
            )
        expected_sources = {
            (spec.source_sequence or Path(spec.category) / spec.relative_directory).as_posix()
            for spec in specs
        }
        manifested_sources = {str(row["source_sequence"]) for row in rows}
        for source in sorted(expected_sources - manifested_sources):
            issues.append(ValidationIssue("error", source, "Source sequence is absent from manifest"))
        for source in sorted(manifested_sources - expected_sources):
            issues.append(ValidationIssue("error", source, "Manifest source does not exist in input tree"))

    for row in rows:
        relative = Path(str(row["video_path"]))
        category = str(row["category"])
        expected_label = LABEL_MAP.get(category)
        subject = str(row["video_id"])
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(ValidationIssue("error", subject, "video_path escapes dataset root"))
            continue
        if category not in categories:
            issues.append(ValidationIssue("error", subject, f"Unknown category: {category}"))
        if expected_label is not None and row["label"] != expected_label:
            issues.append(ValidationIssue("error", subject, f"Wrong label: expected {expected_label}, got {row['label']}"))
        if not relative.parts or relative.parts[0] != category:
            issues.append(
                ValidationIssue("error", subject, "video_path category does not match record category")
            )
        path = dataset_root / relative
        try:
            metadata = validate_video(
                path,
                expected_frames=int(row["num_frames"]),
                expected_fps=float(row["fps"]),
                expected_size=(int(row["width"]), int(row["height"])),
                decode_all=decode_all,
            )
        except Exception as exc:
            issues.append(ValidationIssue("error", subject, str(exc)))
            continue
        actual_duration = float(metadata["duration_seconds"])
        if abs(actual_duration - float(row["duration_seconds"])) > 1e-6:
            issues.append(
                ValidationIssue(
                    "error",
                    subject,
                    f"Duration mismatch: manifest={row['duration_seconds']} actual={actual_duration}",
                )
            )
            continue
        frames = int(metadata["num_frames"])
        if frames < short_threshold:
            short_count += 1
            issues.append(
                ValidationIssue("warning", subject, f"Short video: {frames} < {short_threshold} frames")
            )
        counts[category] += 1
        resolutions[f"{metadata['width']}x{metadata['height']}"] += 1
        frame_rates[f"{float(metadata['fps']):g}"] += 1
        frame_counts.append(frames)
        valid_count += 1

    return ValidationSummary(
        expected_sequences=None if expected_sources is None else len(expected_sources),
        manifest_records=len(rows),
        valid_videos=valid_count,
        failed_videos=len(rows) - valid_count,
        short_videos=short_count,
        category_counts=dict(sorted(counts.items())),
        resolution_counts=dict(sorted(resolutions.items())),
        fps_counts=dict(sorted(frame_rates.items())),
        frame_counts=tuple(frame_counts),
        issues=tuple(issues),
    )


def render_report(summary: ValidationSummary, dataset_root: Path, manifest_path: Path) -> str:
    counts = summary.frame_counts
    lines = [
        "# Dataset validation report",
        "",
        f"Dataset root: `{dataset_root.resolve()}`  ",
        f"Manifest: `{manifest_path.resolve()}`  ",
        f"Status: **{'PASS' if summary.ok else 'FAIL'}**",
        "",
        "Label definition: **clean = 0; warm/cold = 1**.",
        "",
        "## Summary",
        "",
        f"- Expected source sequences: {summary.expected_sequences if summary.expected_sequences is not None else 'not checked'}",
        f"- Manifest records: {summary.manifest_records}",
        f"- Valid videos: {summary.valid_videos}",
        f"- Failed videos: {summary.failed_videos}",
        f"- Class 0 (clean): {summary.category_counts.get('clean', 0)}",
        f"- Class 1 (warm/cold): {summary.category_counts.get('warm', 0) + summary.category_counts.get('cold', 0)}",
        f"- Videos shorter than configured threshold: {summary.short_videos}",
    ]
    if counts:
        ordered = sorted(counts)
        lines.extend(
            [
                f"- Frame count min/median/max: {ordered[0]} / {statistics.median(ordered):g} / {ordered[-1]}",
                f"- Frame count mean: {statistics.mean(ordered):.2f}",
            ]
        )
    lines.extend(["", "## Category statistics", ""])
    if summary.category_counts:
        lines.extend(
            [f"- `{category}`: {count}" for category, count in summary.category_counts.items()]
        )
    else:
        lines.append("No valid videos.")
    lines.extend(["", "## Encoding statistics", ""])
    lines.append(
        "- FPS values: "
        + (", ".join(f"{fps} ({count} videos)" for fps, count in summary.fps_counts.items()) or "none")
    )
    lines.append(
        "- Resolutions: "
        + (
            ", ".join(
                f"{resolution} ({count} videos)"
                for resolution, count in summary.resolution_counts.items()
            )
            or "none"
        )
    )
    lines.extend(["", "## Issues", ""])
    if summary.issues:
        lines.extend(
            f"- **{issue.severity.upper()}** `{issue.subject}`: {issue.message}"
            for issue in summary.issues
        )
    else:
        lines.append("No issues detected.")
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
