#!/usr/bin/env python3
"""Read-only pretrained encoder forward parity against installed HF reference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from transformers import AutoModel
from vision_model.models.encoder import load_pretrained_encoder
from prepare_experiment81 import normalize_bank, digest


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU required")
    out = a.root / "encoder_reference_parity.json"
    if out.exists():
        raise FileExistsError(out)
    c = torch.load(a.root / "prepared/prepared.pt", map_location="cpu", weights_only=False, mmap=True)
    path = c["metadata"]["encoder_dir"]
    custom = load_pretrained_encoder(path, local_files_only=True).encoder.to("cuda").eval()
    ref_wrapper = AutoModel.from_pretrained(path, attn_implementation="sdpa", local_files_only=True)
    reference = ref_wrapper.encoder.to("cuda").eval()
    del ref_wrapper
    results = []
    for source in (int(c["train_indices"][0]), c["ids"].index("cold_folderImage56")):
        pixels = normalize_bank(c["pixels"][source, :1], c["normalization"]).float().to("cuda")
        for frames in (2, 12):
            clip = pixels[:, :frames]
            actual = custom(clip.permute(0, 2, 1, 3, 4))
            expected = reference(clip).last_hidden_state
            delta = actual - expected
            relative_rms = float(delta.square().mean().sqrt() / expected.square().mean().sqrt())
            result = dict(id=c["ids"][source], frames=frames, token_shape=list(actual.shape),
                token_max_abs_error=float(delta.abs().max()), relative_rms_error=relative_rms,
                last_tubelet_pooled_max_abs_error=float((actual[:, -256:].mean(1) - expected[:, -256:].mean(1)).abs().max()),
                structural_parity_passed=bool(delta.abs().max() < .01 and relative_rms < 1e-4))
            results.append(result); print(json.dumps(result), flush=True)
    out.write_text(json.dumps(dict(reference="Installed transformers.models.vjepa2 encoder, same immutable checkpoint",
        encoder_sha256=digest(Path(path) / "model.safetensors"), precision="FP32, CUDA SDPA, identical normalized input values",
        thresholds=dict(max_abs=.01, relative_rms=1e-4), results=results,
        all_passed=all(r["structural_parity_passed"] for r in results)), indent=2) + "\n")


if __name__ == "__main__":
    main()
