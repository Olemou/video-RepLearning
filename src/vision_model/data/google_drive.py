"""Read-only downloader for publicly shared Google Drive folder trees.

This intentionally supports public folders only. It does not accept cookies,
OAuth tokens, or private account credentials.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
FOLDER_MIME = "application/vnd.google-apps.folder"
_IVD = re.compile(r"window\['_DRIVE_ivd'\]\s*=\s*'(.*?)';", re.DOTALL)
_ENTRY = re.compile(
    r'\["(?P<id>[A-Za-z0-9_-]+)",\["(?P<parent>[A-Za-z0-9_-]+)"\],'
    r'"(?P<name>[^"]+)","(?P<mime>[^"]+)"'
)


class DriveDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DriveEntry:
    file_id: str
    parent_id: str
    name: str
    mime_type: str


@dataclass(frozen=True)
class DownloadRecord:
    file_id: str
    relative_path: str
    mime_type: str
    size_bytes: int
    sha256: str


def _safe_name(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name or "\x00" in name:
        raise DriveDownloadError(f"Unsafe Drive entry name: {name!r}")
    return name


def _request(url: str, attempts: int = 5) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == attempts:
                raise DriveDownloadError(f"Request failed after {attempts} attempts: {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_public_folder(folder_id: str) -> list[DriveEntry]:
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    html = _request(url).decode("utf-8", errors="replace")
    match = _IVD.search(html)
    if match is None:
        raise DriveDownloadError(
            f"Public folder metadata was not found for {folder_id}; verify sharing permissions"
        )
    # Drive embeds its JSON-like payload in a JS single-quoted string using
    # \xNN escapes. Decoding that layer exposes ordinary quoted metadata.
    payload = codecs.decode(match.group(1), "unicode_escape").replace("\\/", "/")
    entries: dict[str, DriveEntry] = {}
    for item in _ENTRY.finditer(payload):
        entry = DriveEntry(
            file_id=item.group("id"),
            parent_id=item.group("parent"),
            name=_safe_name(item.group("name")),
            mime_type=item.group("mime"),
        )
        if entry.parent_id == folder_id:
            entries[entry.file_id] = entry
    return list(entries.values())


def _download_file(entry: DriveEntry, target: Path, overwrite: bool) -> DownloadRecord:
    if target.exists() and not overwrite:
        if not target.is_file() or target.stat().st_size == 0:
            raise DriveDownloadError(f"Existing target is not a valid non-empty file: {target}")
        content = target.read_bytes()
    else:
        url = (
            "https://drive.usercontent.google.com/download"
            f"?id={entry.file_id}&export=download&authuser=0"
        )
        content = _request(url)
        if not content:
            raise DriveDownloadError(f"Downloaded empty file: {entry.name} ({entry.file_id})")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return DownloadRecord(
        file_id=entry.file_id,
        relative_path="",  # populated by caller relative to its root
        mime_type=entry.mime_type,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def download_public_tree(
    folder_id: str,
    output_root: Path,
    *,
    categories: tuple[str, ...] = ("clean", "cold", "warm"),
    overwrite: bool = False,
    workers: int = 8,
) -> list[DownloadRecord]:
    output_root = output_root.resolve()
    root_entries = list_public_folder(folder_id)
    by_name = {entry.name: entry for entry in root_entries}
    missing = set(categories) - by_name.keys()
    if missing:
        raise DriveDownloadError(f"Missing configured categories in Drive folder: {sorted(missing)}")
    unexpected = sorted(
        entry.name for entry in root_entries if entry.mime_type == FOLDER_MIME and entry.name not in categories
    )
    if unexpected:
        LOGGER.warning("Ignoring unconfigured root folders: %s", unexpected)

    records: list[DownloadRecord] = []
    for category in categories:
        category_entry = by_name[category]
        if category_entry.mime_type != FOLDER_MIME:
            raise DriveDownloadError(f"Configured category is not a folder: {category}")
        sequence_entries = list_public_folder(category_entry.file_id)
        for sequence_index, sequence in enumerate(sequence_entries, start=1):
            if sequence.mime_type != FOLDER_MIME:
                raise DriveDownloadError(f"Expected sequence folder, found file: {category}/{sequence.name}")
            files = list_public_folder(sequence.file_id)
            if not files:
                LOGGER.warning("Empty sequence: %s/%s", category, sequence.name)
                continue
            names = [entry.name.casefold() for entry in files]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise DriveDownloadError(
                    f"Duplicate filenames in {category}/{sequence.name}: {duplicates}"
                )
            LOGGER.info(
                "[%s %d/%d] Downloading %d frames from %s",
                category,
                sequence_index,
                len(sequence_entries),
                len(files),
                sequence.name,
            )
            download_jobs: list[tuple[DriveEntry, Path]] = []
            for entry in files:
                if entry.mime_type == FOLDER_MIME:
                    raise DriveDownloadError(
                        f"Unexpected nested folder below sequence: {category}/{sequence.name}/{entry.name}"
                    )
                relative = Path(category) / _safe_name(sequence.name) / _safe_name(entry.name)
                download_jobs.append((entry, relative))

            def download_job(job: tuple[DriveEntry, Path]) -> DownloadRecord:
                entry, relative = job
                record = _download_file(entry, output_root / relative, overwrite)
                return DownloadRecord(
                    file_id=record.file_id,
                    relative_path=relative.as_posix(),
                    mime_type=record.mime_type,
                    size_bytes=record.size_bytes,
                    sha256=record.sha256,
                )

            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                sequence_records = list(executor.map(download_job, download_jobs))
            for file_index, record in enumerate(sequence_records, start=1):
                records.append(
                    record
                )
                if file_index % 25 == 0 or file_index == len(files):
                    LOGGER.info("  %d/%d frames", file_index, len(files))

    manifest = output_root / "drive_source_manifest.jsonl"
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.relative_path.casefold()):
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    return records
