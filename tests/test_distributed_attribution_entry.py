"""Run the real attribution entry point on uneven distributed sample shards."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import torch
import torch.multiprocessing as mp

from vision_model.training.trajectory import TrajectoryModel

ROOT = Path(__file__).resolve().parents[1]


def _worker(rank, port, checkpoint, cache, output):
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    sys.path.insert(0, str(ROOT / "scripts"))
    import attribute_trajectory as entry
    torch.set_num_threads(1)
    # TorchCodec is not available in the CPU test environment. Its real frame
    # decoding is exercised by the dataset's integration tests where installed.
    entry.save_temporal_figures = lambda result, output, **_: []
    entry.main(["--checkpoint", checkpoint, "--cache", cache, "--output", output,
                "--all-videos", "--target", "state", "--timestep", "2", "--component", "0"])


def test_distributed_attribution_accounts_for_each_source_once(tmp_path):
    records = [dict(video_id=f"v{i}", category="clean" if i < 2 else "cold",
                    label=0 if i < 2 else 1, video_path=f"v{i}.mp4",
                    source_sequence=str(i), num_frames=10, fps=4.0,
                    width=16, height=16, duration_seconds=2.5) for i in range(4)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    for row in records:
        (tmp_path / row["video_path"]).touch()
    cache = dict(features=torch.randn(4, 2, 5, 16),
                 valid=torch.tensor([[1, 1, 1, 1, 0]] * 4, dtype=torch.bool),
                 timestamps=torch.tensor([[.25, .75, 1.25, 1.75, 0]] * 4),
                 frame_indices=torch.tensor([[1, 3, 5, 7, -1]] * 4),
                 labels=torch.tensor([0, 0, 1, 1]), ids=[r["video_id"] for r in records],
                 temporal_context_policy="past_only_prefix_last_tubelet_v1",
                 manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 dataset_root=str(tmp_path))
    cache_path = tmp_path / "features.pt"
    torch.save(cache, cache_path)
    model = TrajectoryModel(16)
    checkpoint = dict(format="trajectory_resumable_v1", model=model.state_dict(),
                      width=16, hidden=128, embedding_dim=128,
                      resume_settings={"cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()})
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = tmp_path / "attribution"
    mp.spawn(_worker, args=(port, str(checkpoint_path), str(cache_path), str(output)), nprocs=3)
    report = json.loads((output / "manifest.json").read_text())
    assert report["sample_count"] == 4
    assert report["world_size"] == 3
    assert [row["index"] for row in report["results"]] == [0, 1, 2, 3]
    assert len(list(output.glob("*/attribution.json"))) == 4
