#!/usr/bin/env python3
"""Training-only actual-attention gradient calibration; no optimizer steps."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from vision_model.training.distributed import initialize_distributed
from vision_model.training.trajectory import TrajectoryModel,DistributedTrajectoryStep
from vision_model.training.spill_localization import SpillPatchHead,foreground_background_loss
from train_spill_localization import build_student,DEFAULT_PREFIX
from train_spatial_semantic import DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--pseudo-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    ck=torch.load(EXPERIMENT/'training/trial_01_checkpoint.pt',map_location='cpu',weights_only=False)
    cache=torch.load(DEFAULT_PREFIX,map_location='cpu',weights_only=False,mmap=True)
    probe=torch.load(a.probe,map_location='cpu',weights_only=False);candidate=torch.load(a.pseudo_cache,map_location='cpu',weights_only=False)
    if probe['metadata']['supervision_kind']!='automatic_unverified' or probe['metadata']['supervision_sha256']!=digest(a.pseudo_cache):raise ValueError('Probe/candidate mismatch')
    head=SpillPatchHead(**probe['metadata']['head_configuration']);head.load_state_dict(probe['local_head'],strict=True)
    student=build_student(cache,head,train_attention=True).to(ctx.device).eval()
    temporal=TrajectoryModel(ck['width'],ck['hidden'],ck['embedding_dim']).to(ctx.device).eval().requires_grad_(False)
    temporal.load_state_dict(ck['model'],strict=True)
    train=[raw['ids'].index(v) for v in ck['train_source_ids']];local=train[ctx.rank::ctx.world_size]
    targets=candidate['pseudo_targets'][probe['metadata']['pseudo_case']][local].to(ctx.device)
    with torch.no_grad():
        means=torch.zeros(len(local),2,raw['valid'].shape[1],ck['width'],device=ctx.device);logits=torch.zeros(*means.shape[:-1],256,device=ctx.device)
        for j,i in enumerate(local):
            for t in range(int(raw['valid'][i].sum())):
                hidden=torch.stack([cache['rows'][i]['prefixes'][v][t] for v in (0,1)]).to(ctx.device)
                result=student(hidden,frames=t+1);means[j,:,t]=result['student_features'];logits[j,:,t]=result['local_logits']
    means.requires_grad_(True);logits.requires_grad_(True)
    trajectory,_,_=DistributedTrajectoryStep(temporal)(raw['features'][local].to(ctx.device),raw['valid'][local].to(ctx.device),raw['timestamps'][local].to(ctx.device),raw['labels'][local].to(ctx.device),epoch=20,student_features=means)
    foreground,_=foreground_background_loss(logits,targets)
    gm=torch.autograd.grad(trajectory,means)[0];gl=torch.autograd.grad(foreground,logits)[0]
    params=tuple(student.blocks[0].attn.parameters());gt=[torch.zeros_like(p) for p in params];gf=[torch.zeros_like(p) for p in params]
    for j,i in enumerate(local):
        for t in range(int(raw['valid'][i].sum())):
            hidden=torch.stack([cache['rows'][i]['prefixes'][v][t] for v in (0,1)]).to(ctx.device)
            result=student(hidden,frames=t+1)
            first=torch.autograd.grad((result['student_features']*gm[j,:,t]).sum(),params,retain_graph=True)
            second=torch.autograd.grad((result['local_logits']*gl[j,:,t]).sum(),params)
            for accum,value in zip(gt,first):accum.add_(value)
            for accum,value in zip(gf,second):accum.add_(value)
    gt=torch.cat([v.flatten() for v in gt]);gf=torch.cat([v.flatten() for v in gf])
    if ctx.world_size>1:
        dist.all_reduce(gt);dist.all_reduce(gf);gt/=ctx.world_size;gf/=ctx.world_size
    ratio=.25*float(gt.norm()/gf.norm().clamp_min(1e-20));weight=min(10.,max(.001,ratio))
    if ctx.is_main:
        result=dict(fg_weight=weight,unclipped_weight=ratio,rule='New local gradient norm = 25% of original-four-term norm on actual attention at the trained probe; clamp [.001,10]',
            trajectory_gradient_norm=float(gt.norm()),automatic_local_gradient_norm=float(gf.norm()),gradient_cosine=float((gt@gf)/(gt.norm()*gf.norm()).clamp_min(1e-20)),
            training_sources=len(train),validation_sources_used=0,optimizer_updates=0,world_size=ctx.world_size,
            probe_sha256=digest(a.probe),pseudo_cache_sha256=digest(a.pseudo_cache),code_sha256=digest(Path(__file__)))
        with a.output.open('x') as stream:json.dump(result,stream,indent=2)
        print(json.dumps(result),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
