#!/usr/bin/env python3
"""Relative-contrast component ablation, sharing the same raw local cache."""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import torch
import torch.distributed as dist
from vision_model.training.local_thermal import LocalThermalSettings,local_relative_features,region_pairs,compact_pair
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.dense_encoder import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);ctx=initialize_distributed();path=a.root/'prepared/cache.pt';out=a.root/'relative_pairs.pt'
    if out.exists():raise FileExistsError(out)
    c=torch.load(path,map_location='cpu',mmap=True,weights_only=False);settings=LocalThermalSettings()
    owned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(owned,len(c['ids']))
    rows=[]
    for i in owned:
        with torch.no_grad():
            field=local_relative_features(c['gray'][i].to(ctx.device).float()/255,replace(settings,stride=1))
            relative=field[:,::2].clamp_min(0).mean(1).cpu()
        pairs=[[compact_pair(p) for p in region_pairs(-frame.numpy(),settings)] for frame in relative]
        rows.append((i,pairs));print(json.dumps(dict(rank=ctx.rank,id=c['ids'][i],pairs=sum(map(len,pairs)))),flush=True)
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
    else:pieces=[rows]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r[0])
        if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Duplicate/missing source')
        pairs=[r[1] for r in rows]
        data=dict(ids=c['ids'],pairs=pairs,cache_sha256=file_sha256(path),geometry='Components of negative mean positive local relative contrast; raw-intensity components retained in prepared cache',
            settings=settings.state_dict(),world_size=ctx.world_size,rank_source_counts=counts,code_sha256=file_sha256(__file__))
        torch.save(data,out)
        coverage=[dict(id=c['ids'][i],label=int(c['labels'][i]),split='train' if i in c['train_indices'] else 'selection',
            pairs=sum(map(len,pairs[i])),frames_with_pairs=sum(bool(v) for v in pairs[i])) for i in range(len(pairs))]
        (a.root/'relative_pairs.json').write_text(json.dumps({**{k:v for k,v in data.items() if k not in ['pairs']},'coverage':coverage},indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
