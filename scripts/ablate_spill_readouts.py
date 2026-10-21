#!/usr/bin/env python3
"""Bounded DDP spatial-readout ablations using verified causal patch caches.

B/C use global ViCo on a patch-preserving readout. D adds latent-instance MIL;
E additionally trains a rank-16 residual adapter before final encoder norm.
The original encoder weights never change. These are spatial branch studies,
not controlled reproductions of A's four-term temporal optimization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from vision_model.training.vico import vico_pairwise_contrastive
from vision_model.training.lr_search import warmup_cosine
from diagnose_spill_representation import recognition, digest


class PatchReadout(nn.Module):
    """Shared patch MLP, global attention pooling and optional latent MIL."""
    def __init__(self, norm_state, *, mil=False, adapter=False):
        super().__init__()
        self.mil, self.use_adapter = mil, adapter
        self.norm = nn.LayerNorm(1024, eps=1e-6)
        self.norm.load_state_dict(norm_state, strict=True)
        self.norm.requires_grad_(False)
        if adapter:
            self.adapter_down = nn.Linear(1024, 16, bias=False)
            self.adapter_up = nn.Linear(16, 1024, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        self.patch = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 64), nn.GELU())
        self.pool = nn.Linear(64, 1)
        self.semantic = nn.Linear(64, 128)
        self.local = nn.Linear(64, 1)
        self.local.requires_grad_(mil)

    def forward(self, tokens, valid):
        # [sources*views,time,patch,width], mask includes letterbox and time.
        if self.use_adapter:
            tokens = self.norm(tokens + self.adapter_up(F.gelu(self.adapter_down(F.layer_norm(tokens, (1024,))))))
        h = self.patch(tokens)
        score = self.pool(h).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = score.flatten(1).softmax(1).reshape_as(score)
        mean = (h * valid[..., None]).sum((1, 2)) / valid.sum((1, 2))[:, None]
        pooled = .5 * mean + .5 * (h * weights[..., None]).sum((1, 2))
        embedding = F.normalize(self.semantic(pooled), dim=-1)
        logits = self.local(h).squeeze(-1)
        masked = logits.masked_fill(~valid, -torch.inf).flatten(1)
        count = max(1, math.ceil(int(valid[0].sum()) * .05))
        # All sources in this cache have six valid tubelets and the same 4:3
        # content grid; check rather than silently varying instance counts.
        if not torch.all(valid.sum((1, 2)) == valid[0].sum()):
            raise ValueError("This controlled top-k readout requires equal valid patch counts")
        bag = masked.topk(count, dim=1).values.mean(1)
        return embedding, logits, bag, weights


def local_mil(logits, bag, labels, valid):
    video = F.binary_cross_entropy_with_logits(bag, labels.float())
    clean = F.softplus(logits[labels == 0][valid[labels == 0]]).mean()
    sparse = logits[labels == 1][valid[labels == 1]].sigmoid().mean()
    return video + .25 * clean + .05 * sparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spatial-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(4)
    ctx = initialize_distributed()
    if ctx.is_main:
        a.output.mkdir(parents=True, exist_ok=False)
    if ctx.world_size > 1:
        dist.barrier()
    c = torch.load(a.spatial_cache, map_location="cpu", weights_only=False)
    if c["cache_feature_max_error"] > .08:
        raise ValueError("Rejecting spatial cache with unmatched teacher precision")
    y, ids = c["labels"], c["ids"]
    train, val = c["train_indices"], c["validation_indices"]
    assigned = torch.arange(ctx.rank, len(ids), ctx.world_size)
    _, rank_counts = audit_distributed_indices(assigned.tolist(), len(ids))
    lookup = {int(global_i): local_i for local_i, global_i in enumerate(assigned)}
    # All source frames are 4:3, giving 32 letterbox pixels at top and bottom.
    # Derive masks from manifest metadata, rather than assuming a spill ROI.
    manifest = ROOT / "reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/balanced/manifest.jsonl"
    records = {r["video_id"]:r for r in (json.loads(line) for line in manifest.read_text().splitlines())}
    masks = torch.zeros(len(ids), c["valid"].shape[1], 16, 16, dtype=torch.bool)
    for i, name in enumerate(ids):
        r = records[name]
        scale = min(256/r["height"], 256/r["width"])
        h, w = round(r["height"]*scale), round(r["width"]*scale)
        pixel = torch.zeros(256,256)
        top, left = (256-h)//2, (256-w)//2
        pixel[top:top+h,left:left+w] = 1
        spatial = F.avg_pool2d(pixel[None,None],16,16)[0,0] > .5
        masks[i] = c["valid"][i,:,None,None] & spatial[None]
    masks = masks.flatten(2)
    summaries = []
    cases = [("B_final_global",23,False,False), ("C_block5_global",5,False,False),
             ("C_block11_global",11,False,False), ("C_block17_global",17,False,False),
             ("D_final_global_plus_MIL",23,True,False), ("E_final_MIL_rank16_adapter",23,True,True)]
    for name, layer, mil, adapter in cases:
        torch.manual_seed(239)
        model = PatchReadout(c["norm_state"], mil=mil, adapter=adapter).to(ctx.device)
        # Keep shared readout initialization identical when adding the adapter.
        if adapter:
            torch.manual_seed(239)
            base = PatchReadout(c["norm_state"], mil=True).state_dict()
            model.load_state_dict({**model.state_dict(), **base}, strict=True)
        wrapped = DistributedDataParallel(model, device_ids=[ctx.local_rank], broadcast_buffers=False) if ctx.world_size>1 else model
        slot = c["layers"].index(layer)
        bank = (c["final_raw"][assigned] if adapter else c["tokens"][assigned,:,:,slot]).to(ctx.device)
        valid_bank = masks[assigned].to(ctx.device)
        optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=5e-4, eps=1e-12, weight_decay=0.)
        history = []
        exposure = torch.zeros(len(ids), dtype=torch.long)
        for step in range(1, a.updates+1):
            g = torch.Generator().manual_seed(239 + step * 104729)
            if 16 % ctx.world_size:
                raise ValueError("Each class must divide across ranks")
            shards=[]
            for rank in range(ctx.world_size):
                for label in (0,1):
                    candidates=train[(y[train]==label) & (train.remainder(ctx.world_size)==rank)]
                    if len(candidates)<16//ctx.world_size:
                        raise ValueError("Insufficient distinct sources on a class/rank shard")
                    shards.append(candidates[torch.randperm(len(candidates),generator=g)[:16//ctx.world_size]])
            selected=torch.cat(shards)
            local_count=32//ctx.world_size
            chosen=selected[ctx.rank*local_count:(ctx.rank+1)*local_count]
            positions = torch.tensor([lookup[int(i)] for i in chosen],device=ctx.device)
            if not set(chosen.tolist()).issubset(set(assigned.tolist())):
                raise RuntimeError("Source ownership mismatch")
            exposure[chosen] += 1
            data = bank[positions].flatten(0,1).float()
            valid = valid_bank[positions,None].expand(-1,2,-1,-1).flatten(0,1)
            labels = y[chosen].repeat_interleave(2).to(ctx.device)
            source_ids = chosen.repeat_interleave(2).to(ctx.device)
            if step==21:
                optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = warmup_cosine(step,a.updates,5e-4,warmup_fraction=.1)
            optimizer.zero_grad(set_to_none=True)
            embedding, logits, bag, weights = wrapped(data,valid)
            if ctx.world_size>1:
                from torch.distributed.nn.functional import all_gather
                embeddings = torch.cat(all_gather(embedding))
                all_labels, all_ids = [torch.empty_like(labels) for _ in range(ctx.world_size)], [torch.empty_like(source_ids) for _ in range(ctx.world_size)]
                dist.all_gather(all_labels,labels)
                dist.all_gather(all_ids,source_ids)
                labels_global, ids_global = torch.cat(all_labels), torch.cat(all_ids)
            else:
                embeddings, labels_global, ids_global = embedding, labels, source_ids
            if ids_global.unique().numel()!=32 or torch.bincount(labels_global,minlength=2).tolist()!=[32,32]:
                raise RuntimeError("Distributed update duplicated sources or lost class balance")
            semantic,_ = vico_pairwise_contrastive(embeddings, labels_global, source_ids=ids_global,epoch=step,curriculum_epochs=20)
            weak = local_mil(logits,bag,labels,valid) if mil else semantic.new_zeros(())
            loss = semantic + weak
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(),5.)
            optimizer.step()
            values=torch.tensor([float(loss.detach()),float(semantic.detach()),float(weak.detach())],device=ctx.device)
            if ctx.world_size>1:
                dist.all_reduce(values)
                values /= ctx.world_size
            history.append(dict(update=step,total=float(values[0]),global_vico=float(values[1]),mil=float(values[2]),gradient_norm=float(norm)))
        model.eval()
        local = []
        with torch.no_grad():
            for pos, global_i in enumerate(assigned):
                valid = valid_bank[pos:pos+1].expand(2,-1,-1)
                emb,logit,bag,weights = model(bank[pos].float(),valid)
                local.append(dict(index=int(global_i),embedding=emb.cpu(),maps=logit.cpu(),bag=bag.cpu(),pool=weights.cpu()))
        gathered=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:
            dist.gather_object(local,gathered,dst=0)
        else:
            gathered=[local]
        exposures=exposure.to(ctx.device)
        if ctx.world_size>1:
            dist.all_reduce(exposures)
        if ctx.is_main:
            rows=sorted([r for shard in gathered for r in shard],key=lambda r:r["index"])
            embedding=torch.stack([r["embedding"] for r in rows])[:,0]
            metrics=recognition(embedding,y,train,val)
            metrics.update(case=name,layer=layer,mil=mil,adapter=adapter,optimizer_updates=a.updates,
                           trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),
                           rank_source_counts=rank_counts,source_exposures=exposures.cpu().tolist(),
                           optimizer_state_reset_before_update=21,localization_metrics=None,
                           localization_status="No masks; patch scores and perturbations do not establish localization accuracy")
            summaries.append(metrics)
            torch.save(dict(model=model.cpu().state_dict(),optimizer=optimizer.state_dict(),history=history,
                            metrics=metrics,rows=rows,seed=239,spatial_cache_sha256=digest(a.spatial_cache),
                            train_source_ids=[ids[i] for i in train],validation_source_ids=[ids[i] for i in val]),a.output/f"{name}.pt")
            print(json.dumps(dict(case=name,accuracy=metrics["accuracy"],global_vico=history[-1]["global_vico"],mil=history[-1]["mil"])),flush=True)
        if ctx.world_size>1:
            dist.barrier()
        del wrapped,model,bank,optimizer
        torch.cuda.empty_cache()
    if ctx.is_main:
        (a.output/"results.json").write_text(json.dumps(dict(seed=239,updates=a.updates,global_source_batch=32,
            views=["identity","mirror"],protocol="Fixed-budget spatial branch probe; selection sources excluded from fitting; no independent test or localization ground truth",results=summaries),indent=2)+"\n")
    if ctx.world_size>1:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
