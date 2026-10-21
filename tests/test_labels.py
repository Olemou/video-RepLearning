"""The user's class definition must survive conversion, loading and caching."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from vision_model.data.labels import LABEL_MAP, LABEL_SCHEMA, label_for, validate_feature_cache
from vision_model.data.video_dataset import read_manifest
from vision_model.data.conversion import convert_dataset
from vision_model.data.validation import validate_dataset
from tests.test_conversion import make_image

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_mapping_and_configuration_and_cli(monkeypatch):
    assert LABEL_MAP == {"clean": 0, "cold": 1, "warm": 1}
    assert yaml.safe_load((ROOT / "configs/data.yaml").read_text())["dataset"]["label_map"] == LABEL_MAP
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from images_to_mp4 import parser
    args = parser().parse_args(["--input-root", "unused", "--output-root", "unused"])
    assert dict((k, int(v)) for k, v in (s.split("=") for s in args.label_map)) == LABEL_MAP


def test_real_manifest_has_correct_labels_for_every_record():
    records = read_manifest(ROOT / "dataset/manifest.jsonl")
    assert len(records) == 24
    assert sum(r.label == 0 for r in records) == 17
    assert sum(r.label == 1 for r in records) == 7
    for r in records:
        assert r.label == (0 if r.category == "clean" else 1)


def test_loader_rejects_old_cold_label(tmp_path):
    rows = [json.loads(line) for line in (ROOT / "dataset/manifest.jsonl").read_text().splitlines()]
    cold = next(r for r in rows if r["category"] == "cold")
    cold["label"] = 0
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(cold) + "\n")
    with pytest.raises(ValueError, match="cold requires 1"):
        read_manifest(manifest)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_conversion_writes_correct_labels_for_all_three_categories(tmp_path):
    source, output = tmp_path / "source", tmp_path / "dataset"
    for category in ("clean", "cold", "warm"):
        for index in (1, 2):
            make_image(source / category / "sample" / f"{index}.png")
    records, failures = convert_dataset(source, output, fps=5)
    assert not failures
    assert {r.category: r.label for r in records} == {"clean": 0, "cold": 1, "warm": 1}
    assert {r.category: r.label for r in read_manifest(output / "manifest.jsonl")} == LABEL_MAP
    assert validate_dataset(output, output / "manifest.jsonl", source_root=source).ok


@pytest.mark.parametrize("tamper", ["labels", "schema", "hash", "ids"])
def test_feature_cache_rejects_stale_or_misaligned_labels(tamper):
    manifest = ROOT / "dataset/manifest.jsonl"
    records = read_manifest(manifest)
    cache = dict(label_schema=LABEL_SCHEMA, label_map=LABEL_MAP.copy(),
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 ids=[r.video_id for r in records], categories=[r.category for r in records],
                 labels=torch.tensor([r.label for r in records]),
                 features=torch.zeros(24, 2, 4), valid=torch.ones(24, 2, dtype=torch.bool))
    validate_feature_cache(cache, manifest)
    if tamper == "labels":
        cache["labels"][17] = 0
    elif tamper == "schema":
        cache["label_schema"] = "old"
    elif tamper == "hash":
        cache["manifest_sha256"] = "old"
    else:
        cache["ids"] = list(reversed(cache["ids"]))
    with pytest.raises(ValueError, match="cache"):
        validate_feature_cache(cache, manifest)


@pytest.mark.parametrize("counts", [(17, 7), (7, 17)])
def test_balanced_batches_cover_both_classes_without_duplicate_videos(counts, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from simulate_loss_training import balanced_batches
    labels = torch.tensor([0] * counts[0] + [1] * counts[1])
    batches = balanced_batches(labels, epoch=0)
    assert len(batches) == 6
    assert set(torch.cat(batches).tolist()) == set(range(24))
    for indices in batches:
        assert len(indices.unique()) == 6
        assert int(labels[indices].sum()) == 3


def test_balanced_batches_reject_too_few_distinct_positives(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from simulate_loss_training import balanced_batches
    with pytest.raises(RuntimeError, match="distinct"):
        balanced_batches(torch.tensor([0, 0, 1, 1, 1]), epoch=0)


def test_training_rejects_features_encoded_without_padding_mask():
    manifest = ROOT / "dataset/manifest.jsonl"
    with pytest.raises(ValueError, match="encoder padding policy"):
        validate_feature_cache({}, manifest, encoder_padding_policy="masked_attention_v1")
