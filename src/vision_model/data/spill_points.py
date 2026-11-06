"""Verified source-frame points mapped to actual identity/mirror tubelet cells."""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

SCHEMA = "spill_points_v1"


def frame_geometry(height, width, size=256):
    if min(height, width, size) <= 0:
        raise ValueError("Invalid frame dimensions")
    factor = min(size / height, size / width)
    h, w = max(1, min(size, round(height * factor))), max(1, min(size, round(width * factor)))
    return (size - h) // 2, (size - w) // 2, h, w


def content_cells(height, width, *, size=256, grid=(16, 16)):
    """Content occupies >50% of the cell, matching the existing patch audit."""
    if size % grid[0] or size % grid[1]:
        raise ValueError("Grid must divide the checkpoint image")
    top, left, h, w = frame_geometry(height, width, size)
    pixels = torch.zeros(1, 1, size, size)
    pixels[:, :, top:top+h, left:left+w] = 1
    return (torch.nn.functional.avg_pool2d(pixels, (size//grid[0], size//grid[1])) > .5).flatten()


def point_cell(x, y, height, width, *, mirror=False, size=256, grid=(16, 16)):
    """Coordinates are zero-indexed source pixel centers; mirror before resize.

    Mapping the source point explicitly also handles asymmetric letterboxing,
    where simply flipping a label grid would be incorrect.
    """
    if type(x) not in (int,float) or type(y) not in (int,float) or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Points need finite pixel-center coordinates")
    if not 0 <= x <= width-1 or not 0 <= y <= height-1:
        raise ValueError("Point outside original image")
    top, left, h, w = frame_geometry(height, width, size)
    if mirror:
        x = width-1-x
    column = int(((x+.5)*w/width+left) // (size/grid[1]))
    row = int(((y+.5)*h/height+top) // (size/grid[0]))
    return row*grid[1]+column


def annotation_template(raw, records):
    """An empty, unverified template contains no generated foreground labels."""
    rows = []
    for i, name in enumerate(raw['ids']):
        record = records[name]
        for frame in raw['frame_indices'][i][raw['valid'][i]].tolist():
            rows.append(dict(source_id=name, source_sha256=raw['video_sha256'][i],
                frame_index=frame, width=record['width'], height=record['height'],
                verified=False, verified_by="", points=[]))
    return dict(schema=SCHEMA, coordinate_system="source_pixel_centres",
                temporal_scope="tubelet_endpoint", records=rows)


def load_spill_points(path, raw, records, *, size=256, grid=(16, 16)):
    """Validate identity, provenance, endpoint time and labels; unknown=-1.

    A verified flag/reviewer is an annotation assertion, not automatic proof of
    truth. Reviewers must check point centers against the original image.
    Both views derive only from the same verified source point.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != SCHEMA or payload.get('coordinate_system') != 'source_pixel_centres' or payload.get('temporal_scope') != 'tubelet_endpoint':
        raise ValueError("Expected spill_points_v1, source_pixel_centres, tubelet_endpoint")
    if not isinstance(payload.get('records'), list):
        raise ValueError("Annotation records must be a list")
    ids = raw['ids'];lookup = {name: i for i, name in enumerate(ids)}
    if len(lookup) != len(ids) or len(set(raw['video_sha256'])) != len(ids):
        raise ValueError("Duplicate source IDs or byte-identical videos")
    if raw['valid'].dtype != torch.bool or raw['frame_indices'].shape != raw['valid'].shape:
        raise ValueError("Invalid cache endpoint masks")
    labels = torch.full((len(ids), 2, raw['valid'].shape[1], grid[0]*grid[1]), -1, dtype=torch.long)
    content = torch.stack([content_cells(records[name]['height'], records[name]['width'], size=size, grid=grid) for name in ids])
    verified_records, unverified_records = 0, 0
    for record in payload['records']:
        name = record.get('source_id')
        if name not in lookup:
            raise ValueError(f"Unknown annotated source: {name}")
        i = lookup[name];source = records[name]
        if record.get('source_sha256') != raw['video_sha256'][i]:
            raise ValueError(f"Annotation source hash mismatch: {name}")
        if (record.get('height'), record.get('width')) != (source['height'], source['width']):
            raise ValueError(f"Annotation dimensions mismatch: {name}")
        frame = record.get('frame_index')
        if type(frame) is not int:
            raise ValueError("Annotation frame_index must be an integer")
        times = ((raw['frame_indices'][i] == frame) & raw['valid'][i]).nonzero().flatten()
        if len(times) != 1:
            raise ValueError(f"Frame must match one valid sampled tubelet endpoint: {name}, {frame}")
        t = int(times[0]);points = record.get('points')
        if not isinstance(points, list):
            raise ValueError("Point list is required")
        if record.get('verified') is not True:
            if points:
                raise ValueError("Unverified points cannot be used as training labels")
            unverified_records += 1
            continue
        if not isinstance(record.get('verified_by'), str) or not record['verified_by'].strip():
            raise ValueError("Verified annotations require a reviewer")
        verified_records += 1
        for point in points:
            label = point.get('label')
            if type(label) is int and label == -1:
                continue  # Reviewer explicitly marked an ambiguous point.
            if type(label) is not int or label not in (0, 1):
                raise ValueError("Point label must be 0 (non-liquid) or 1 (liquid)")
            if label == 1 and int(raw['labels'][i]) == 0:
                raise ValueError("Liquid annotation contradicts clean source label; review the labels")
            for view in (0, 1):
                cell = point_cell(point['x'], point['y'], source['height'], source['width'],
                                  mirror=bool(view), size=size, grid=grid)
                if not content[i, cell]:
                    raise ValueError("Annotated point maps to a padding-dominated cell; use a clear content center")
                existing = int(labels[i, view, t, cell])
                if existing not in (-1, label):
                    raise ValueError("Conflicting foreground/background points in one tubelet cell")
                labels[i, view, t, cell] = label
    coverage = dict(verified_records=verified_records, unverified_records=unverified_records,
        identity_foreground_cells=int((labels[:, 0] == 1).sum()),
        identity_background_cells=int((labels[:, 0] == 0).sum()),
        annotated_sources=int((labels[:, 0] >= 0).flatten(1).any(1).sum()),
        all_other_positions="ignored; video labels never become patch labels")
    return labels, content, coverage


def require_training_points(labels, train_indices):
    """Validation labels cannot satisfy a missing training class."""
    selected = labels[train_indices, 0]
    counts = dict(foreground=int((selected == 1).sum()), background=int((selected == 0).sum()))
    if min(counts.values()) == 0:
        raise ValueError(f"Need verified training liquid AND non-liquid points; found {counts}. No episode-label fallback.")
    return counts
