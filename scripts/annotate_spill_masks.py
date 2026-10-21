#!/usr/bin/env python3
"""Export exact model-input frames and interactively draw spill masks.

Masks are saved only when explicitly accepted.  Missing files mean
"unannotated"; an accepted all-black PNG means "annotated, no visible spill".
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image
import torch

from vision_model.data.config import ThermalAugConfig
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.video_dataset import VideoManifestDataset
from train_spill_localizer import local_model_source, validate_cache


def export_frames(cache: dict, manifest: Path, dataset_root: Path | None,
                  output: Path, video_ids: set[str]) -> list[dict]:
    transform = OfficialVJEPA2Transform(
        local_model_source(cache["model_id"]),
        thermal_config=ThermalAugConfig(image_size=256),
    )
    dataset = VideoManifestDataset(
        manifest,
        dataset_root=dataset_root,
        sampling=TemporalSamplingConfig(**cache["sampling"]),
        transform=transform,
    )
    mean = torch.tensor(cache["normalization"]["mean"])[None, :, None, None]
    std = torch.tensor(cache["normalization"]["std"])[None, :, None, None]
    records = []
    for index, item in enumerate(dataset):
        if item["label"] == 0 or (video_ids and item["video_id"] not in video_ids):
            continue
        frames = ((item["pixel_values_videos"].float() * std + mean).clamp(0, 1) * 255)
        frames = frames.permute(0, 2, 3, 1).round().byte().numpy()
        valid_frames = cache["valid"][index].repeat_interleave(2)
        frame_dir = output / item["video_id"] / "frames"
        mask_dir = output / item["video_id"] / "masks"
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for slot in torch.where(valid_frames)[0].tolist():
            source = int(item["frame_indices"][slot])
            name = f"slot_{slot:03d}_source_{source:06d}.png"
            Image.fromarray(frames[slot]).save(frame_dir / name)
            entries.append({"slot": slot, "source_frame": source, "file": name})
        record = {
            "video_id": item["video_id"],
            "category": item["category"],
            "label": item["label"],
            "coordinate_space": "exact 256x256 normalized-model input after inverse normalization",
            "frames": entries,
        }
        (output / item["video_id"] / "annotation.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"exported {item['video_id']}: {len(entries)} frames")
    return records


ANNOTATOR_HTML = Path(__file__).with_name("spill_editor.html").read_text(encoding="utf-8")


def annotate(output: Path, records: list[dict], host: str, port: int) -> None:
    allowed: dict[tuple[str, str], dict] = {}
    items = []
    for record in records:
        for entry in record["frames"]:
            key = (record["video_id"], entry["file"])
            allowed[key] = entry
            items.append({**entry, "video_id": record["video_id"],
                          "url": f"/frame/{record['video_id']}/{entry['file']}",
                          "mask_url": f"/mask/{record['video_id']}/{entry['file']}",
                          "annotated": (output / record["video_id"] / "masks" / entry["file"]).exists()})

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/": return self.send_bytes(ANNOTATOR_HTML.encode(), "text/html; charset=utf-8")
            if path == "/api/items": return self.send_bytes(json.dumps(items).encode(), "application/json")
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) == 3 and parts[0] == "frame" and (parts[1], parts[2]) in allowed:
                source = output / parts[1] / "frames" / parts[2]
                return self.send_bytes(source.read_bytes(), mimetypes.guess_type(source)[0] or "image/png")
            if len(parts) == 3 and parts[0] == "mask" and (parts[1], parts[2]) in allowed:
                source = output / parts[1] / "masks" / parts[2]
                if source.is_file(): return self.send_bytes(source.read_bytes(), "image/png")
            return self.send_bytes(b"not found", "text/plain", 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/save": return self.send_bytes(b"not found", "text/plain", 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                key = (payload["video_id"], payload["file"])
                if key not in allowed: raise ValueError("Unknown frame")
                prefix, encoded = payload["data"].split(",", 1)
                if prefix != "data:image/png;base64": raise ValueError("Expected a PNG data URL")
                raw = base64.b64decode(encoded, validate=True)
                image = Image.open(__import__("io").BytesIO(raw)).convert("L")
                if image.size != (256, 256): raise ValueError("Mask must be 256x256")
                array = np.asarray(image)
                binary = Image.fromarray(np.where(array >= 128, 255, 0).astype(np.uint8))
                destination = output / key[0] / "masks" / key[1]
                temporary = destination.with_suffix(".tmp.png"); binary.save(temporary); temporary.replace(destination)
                for item in items:
                    if (item["video_id"], item["file"]) == key: item["annotated"] = True
                return self.send_bytes(b'{"saved":true}', "application/json")
            except Exception as exc:
                return self.send_bytes(str(exc).encode(), "text/plain", 400)

        def log_message(self, format, *args):
            print(f"annotation server: {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"\nOpen this URL in your browser: http://{host}:{port}")
    print("Keep this terminal running. Press Ctrl+C here when annotation is finished.\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nAnnotation server stopped.")
    finally: server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path,
                        help="directory used to resolve video_path entries (default: manifest directory)")
    parser.add_argument("--token-cache", type=Path, default=ROOT / "checkpoints/local_spatial_video_tokens.pt")
    parser.add_argument("--output", "--path", "--annotations", dest="output", type=Path,
                        default=ROOT / "dataset/spill_annotations",
                        help="folder in which frames and masks are stored")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--annotate", action="store_true", help="start the browser brush editor after export")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    cache = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    validate_cache(cache, args.manifest)
    if cache["token_layer"] != "patch_embed":
        raise ValueError("Mask supervision must align with the local patch-token cache")
    records = export_frames(cache, args.manifest, args.dataset_root, args.output, set(args.video_id))
    if args.annotate:
        annotate(args.output, records, args.host, args.port)


if __name__ == "__main__":
    main()
