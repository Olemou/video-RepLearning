"""Authoritative project labels: clean versus warm/cold."""
import hashlib
from pathlib import Path

LABEL_MAP = {"clean": 0, "cold": 1, "warm": 1}
LABEL_SCHEMA = "clean0_warm_cold1_v1"


def label_for(category: str) -> int:
    try:
        return LABEL_MAP[category]
    except KeyError as exc:
        raise ValueError(f"Unknown category: {category!r}") from exc


def validate_feature_cache(cache, manifest: Path, *, encoder_padding_policy: str | None = None) -> None:
    """Reject stale labels, reordered samples, or a cache from another manifest."""
    from .video_dataset import read_manifest

    records = read_manifest(manifest)
    if encoder_padding_policy is not None and cache.get("encoder_padding_policy") != encoder_padding_policy:
        raise ValueError("Feature cache uses an obsolete encoder padding policy; rebuild it at a new path")
    if cache.get("label_schema") != LABEL_SCHEMA or cache.get("label_map") != LABEL_MAP:
        raise ValueError("Feature cache uses an obsolete label schema; rebuild it")
    if cache.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Feature cache manifest hash mismatch; rebuild it")
    if cache.get("ids") != [r.video_id for r in records]:
        raise ValueError("Feature cache video IDs/order do not match the manifest")
    if cache.get("categories") != [r.category for r in records]:
        raise ValueError("Feature cache categories do not match the manifest")
    if cache["labels"].tolist() != [r.label for r in records]:
        raise ValueError("Feature cache labels do not match the manifest")
    if cache["features"].shape[0] != len(records) or cache["valid"].shape != cache["features"].shape[:2]:
        raise ValueError("Feature cache tensor shapes do not match the manifest")
