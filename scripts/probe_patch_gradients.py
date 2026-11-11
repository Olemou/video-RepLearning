#!/usr/bin/env python3
"""Measure global final-state loss gradients without changing checkpoints."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.distributed as dist
from vision_model.training.patch_vico import PatchProjector,patch_vico,local_feature_retention
from vision_model.training.distributed import initialize_distributed
from train_spatial_semantic import DEFAULT_SPATIAL,DEFAULT_TEACHER,content_masks
from control_patch_vico import case_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--additional-run',type=Path,nargs='+',required=True)
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    destination=a.run/'gradient_probes.json'
    if destination.exists():raise FileExistsError(destination)
    c=torch.load(DEFAULT_SPATIAL,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    content=content_masks(raw['manifest'],c['ids'],(16,16)).flatten(1)
    reference_checkpoint=torch.load(a.additional_run[-1]/'all_patch/checkpoint.pt',map_location='cpu',weights_only=False)
    teacher=PatchProjector(**reference_checkpoint['metadata']['architecture']).to(ctx.device).eval().requires_grad_(False)
    teacher.load_state_dict(reference_checkpoint['local_teacher'],strict=True)
    result={}
    for run in [a.run,*a.additional_run]:
        report=json.loads((run/'report.json').read_text())
        case=next(r for r in report['cases'] if r['case']=='all_patch')
        ck=torch.load(run/'all_patch/checkpoint.pt',map_location='cpu',weights_only=False)
        model=PatchProjector(**ck['metadata']['architecture']).to(ctx.device).eval()
        model.load_state_dict(ck['model'],strict=True)
        indices=torch.tensor([c['ids'].index(i) for i in case['train_source_ids']])
        source=indices[ctx.rank::ctx.world_size]
        steps=(source+case['seed']+case['updates']-1)%c['valid'][source].sum(-1)
        patches=c['tokens'][source,:,steps,c['layers'].index(case['layer'])].float().to(ctx.device)
        mask=content[source,None].expand(-1,2,-1).to(ctx.device)
        patches=patches.masked_fill(~mask[...,None],0)
        q=model(patches)
        labels=c['labels'][source,None,None].to(ctx.device).expand_as(mask)[mask]
        images=(source*c['valid'].shape[-1]+steps)[:,None,None].to(ctx.device).expand_as(mask)[mask]
        sources=source[:,None,None].to(ctx.device).expand_as(mask)[mask]
        semantic,_=patch_vico(q[mask],labels,images,sources,epoch=20)
        with torch.no_grad():target=teacher(patches)
        retention=local_feature_retention(q,target,mask)*ctx.world_size*len(source)/len(indices)
        parameters=tuple(model.parameters());vectors=[]
        for loss in (semantic,retention):
            grads=torch.autograd.grad(loss,parameters,retain_graph=True)
            vector=torch.cat([g.flatten() for g in grads])
            if ctx.world_size>1:dist.all_reduce(vector);vector/=ctx.world_size
            vectors.append(vector)
        sem,keep=vectors;weight=case.get('local_keep_weight',0.)
        result[case_name(case)]=dict(semantic_gradient_norm=float(sem.norm()),local_keep_gradient_norm=float(keep.norm()),
            weighted_local_keep_gradient_norm=float(weight*keep.norm()),gradient_cosine=float((sem@keep)/(sem.norm()*keep.norm()).clamp_min(1e-30)),
            optimized_sum_gradient_norm=float((sem+weight*keep).norm()),local_keep_weight=weight,
            comparison='Final checkpoint, same training-only tubelet schedule as update 100; global parameter gradients; no optimizer step',
            temporal_gradient_to_patch=0.)
    if ctx.is_main:destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
