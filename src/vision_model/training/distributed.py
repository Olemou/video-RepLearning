"""Environment-driven torch.distributed initialization for torchrun and SLURM."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from types import SimpleNamespace


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(backend: str | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        selected_backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        selected_backend = backend or "gloo"

    if world_size > 1 and not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            raise RuntimeError(
                "Multi-node/multi-GPU launch requires MASTER_ADDR and MASTER_PORT. "
                "Set MASTER_ADDR to the first SLURM node (or launch with torchrun)."
            )
        dist.init_process_group(
            backend=selected_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    return DistributedContext(rank, world_size, local_rank, device)


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def unpadded_sampler_indices(sampler: DistributedSampler, dataset_size: int) -> tuple[list[int], int]:
    """Remove only DistributedSampler's repeated tail entries.

    With drop_last=False, the sampler pads its global index list to a multiple
    of world size, then gives rank r positions r, r+world, ... . The padded
    positions begin at dataset_size even when the original list is shuffled.
    """
    if sampler.drop_last or len(sampler.dataset) != dataset_size:
        raise ValueError("Expected a full, non-dropping DistributedSampler")
    assigned = list(sampler)
    unique = [index for position, index in enumerate(assigned)
              if sampler.rank + position * sampler.num_replicas < dataset_size]
    if not unique:
        raise ValueError("Every rank must receive at least one real source video")
    return unique, len(assigned) - len(unique)


def audit_distributed_indices(local_indices: list[int], dataset_size: int) -> tuple[list[int], list[int]]:
    """Check globally that each source is assigned exactly once, in rank order."""
    if dist.is_available() and dist.is_initialized():
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local_indices)
    else:
        shards = [local_indices]
    flattened = [index for shard in shards for index in shard]
    if sorted(flattened) != list(range(dataset_size)):
        raise RuntimeError("Distributed sampler omitted or duplicated source videos")
    return flattened, [len(shard) for shard in shards]


def gather_contrastive_batch(output, labels, original_ids):
    """Gather candidates with gradients, but assign each anchor to one rank.

    Padding supports unequal local batch sizes. Temporal shapes must agree;
    fixed-length video sampling supplies that contract. All ranks participate
    even when one rank owns no valid anchors in the global candidate set.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return output, labels, original_ids, None
    from torch.distributed.nn.functional import all_gather

    world, rank = dist.get_world_size(), dist.get_rank()
    metadata = [None] * world
    ids = original_ids.detach().cpu().tolist() if torch.is_tensor(original_ids) else list(original_ids)
    dist.all_gather_object(metadata, (len(labels), ids))
    counts = [m[0] for m in metadata]
    maximum = max(counts)

    def gather(value, differentiable):
        pad = value.new_zeros((maximum - value.shape[0], *value.shape[1:]))
        padded = torch.cat((value, pad), dim=0)
        if differentiable:
            pieces = all_gather(padded)
        else:
            pieces = [torch.empty_like(padded) for _ in range(world)]
            dist.all_gather(pieces, padded)
        return torch.cat([p[:n] for p, n in zip(pieces, counts)], dim=0)

    fields = {name: gather(getattr(output, name), True) for name in
              ("temporal_mean", "temporal_variance", "video_mean", "video_variance")}
    fields["temporal_valid_mask"] = gather(output.temporal_valid_mask, False)
    mask = torch.zeros(sum(counts), device=labels.device, dtype=torch.bool)
    offset = sum(counts[:rank])
    mask[offset:offset + counts[rank]] = True
    return SimpleNamespace(**fields), gather(labels, False), [i for m in metadata for i in m[1]], mask
