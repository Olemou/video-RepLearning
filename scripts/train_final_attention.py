#!/usr/bin/env python3
"""Matched frozen versus final-attention continuation, without MIL or decoder.

Memory-bounded exact first-order VJP: differentiate the global patch objective
with respect to detached student outputs, then replay each source with those
cotangents through the real student. DDP averages accumulated parameter grads.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vision_model.models.encoder import Block
from vision_model.training.final_attention import FinalAttentionStudent
from vision_model.training.patch_vico import patch_vico,local_feature_retention
from vision_model.training.vico import vico_pairwise_contrastive
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from train_spatial_semantic import DEFAULT_TEACHER,content_masks
from train_patch_vico import patch_diagnostics
from diagnose_spill_representation import digest,recognition


def make_student(cache,train_attention,head_state):
    block=Block(1024,16,4.,16)
    model=FinalAttentionStudent(block,torch.nn.LayerNorm(1024,eps=1e-6),train_attention=train_attention)
    model.load_state_dict({**cache['suffix_state'],**{'projector.'+k:v for k,v in head_state.items()}},strict=True)
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--initial-projector',type=Path,default=ROOT/'reports/experiments/all_patch_vico_local_keep_1_20261021/all_patch/checkpoint.pt')
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--attention-lr',type=float,default=3e-6)
    p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--keep-weight',type=float,default=1.)
    p.add_argument('--seed',type=int,default=239)
    p.add_argument('--reuse-frozen',type=Path,help='Reuse a verified identical frozen control instead of rerunning it')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed)
    if a.updates<1 or a.eval_every<1 or a.keep_weight<0:raise ValueError('Invalid budget')
    ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    head=torch.load(a.initial_projector,map_location='cpu',weights_only=False)
    if digest(a.teacher_cache)!=c['teacher_cache_sha256'] or c['ids']!=raw['ids']:raise ValueError('Cache provenance mismatch')
    if head['metadata']['train_source_ids']!=[c['ids'][i] for i in c['train_indices']]:raise ValueError('Initialization trained a different split')
    if c['temporal_context_policy']!='past_only_prefix_last_tubelet_v1':raise ValueError('Invalid causal cache')
    train,val=c['train_indices'],c['validation_indices'];y=c['labels'];valid=c['valid']
    content=content_masks(raw['manifest'],c['ids'],(16,16)).flatten(1)
    # Partition each split, so each rank has 32 real training and 8 real
    # validation sources. No repeated/padded evaluation or training sources.
    local_train=train[ctx.rank::ctx.world_size].tolist()
    assigned=sorted(local_train+val[ctx.rank::ctx.world_size].tolist())
    _,evalcounts=audit_distributed_indices(assigned,len(c['ids']))
    counts=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(counts,len(local_train))
    else:counts=[len(local_train)]
    if min(counts)<1 or len(set(counts))!=1:raise ValueError('VJP synchronization requires equal distinct training source counts')
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    # Immutable continuation anchor; no updated student output becomes a teacher.
    teacher=make_student(c,False,head['model']).to(ctx.device).eval().requires_grad_(False)
    reference={}
    with torch.no_grad():
        for i in assigned:reference[i]=teacher.projector(c['rows'][i]['reference'].to(ctx.device))

    def input_for(i,step):
        return torch.stack([c['rows'][i]['prefixes'][v][step] for v in range(2)]).to(ctx.device).float()

    @torch.no_grad()
    def evaluate(model,update,folder=None):
        model.eval();rows=[]
        for i in assigned:
            q=torch.stack([model(input_for(i,p),frames=p+1).cpu() for p in range(int(valid[i].sum()))],1)
            rows.append((i,q))
        gathered=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
        else:gathered=[rows]
        if not ctx.is_main:return None
        ordered=sorted([r for shard in gathered for r in shard],key=lambda r:r[0])
        if [r[0] for r in ordered]!=list(range(len(c['ids']))):raise RuntimeError('Evaluation duplication')
        q=torch.stack([r[1] for r in ordered]);support=valid[:,None,:,None]&content[:,None,None]
        mean=q.masked_fill(~support[...,None],0).sum((2,3))/support.expand(-1,2,-1,-1).sum((2,3))[...,None]
        z=F.normalize(mean,dim=-1)
        proto=torch.stack([F.normalize(z[train,0][y[train]==j].mean(0),dim=0) for j in (0,1)])
        z,maps,diag,_,reconstruction=patch_diagnostics(q,valid,content,proto)
        result=dict(update=update,margin_reconstruction_max_error=reconstruction)
        for name,indices in (('train',train),('validation',val)):
            metrics=recognition(z[:,0],y,train,indices,exclude_self=name=='train')
            _,detail=vico_pairwise_contrastive(z[indices].flatten(0,1),y[indices].repeat_interleave(2),source_ids=indices.repeat_interleave(2),epoch=20,curriculum_epochs=20)
            metrics.update({k:float(v) for k,v in detail.items()})
            metrics.update({k:float(v[indices].mean()) for k,v in diag.items()})
            # Source-excluded retrieval is recognition's leave-one-source-out probe.
            result[name]=metrics
        if folder:
            torch.save(dict(ids=c['ids'],labels=y,valid=valid,content=content,train_indices=train,validation_indices=val,
                embeddings=z,signed_maps=maps,patch_features=q,prototypes=proto,diagnostics=diag),folder/'representations.pt')
        return result

    cases=[]
    for case,adapt in (('FT0_frozen',False),('FT1_attention',True)):
        if not adapt and a.reuse_frozen:
            prior=torch.load(a.reuse_frozen/'checkpoint.pt',map_location='cpu',weights_only=False)
            meta=prior['metadata']
            for key,expected in dict(train_attention=False,updates=a.updates,seed=a.seed,head_lr=a.head_lr,
                local_keep_weight=a.keep_weight,cache_sha256=digest(a.cache),initial_projector_sha256=digest(a.initial_projector)).items():
                if meta[key]!=expected:raise ValueError(f'Reused frozen control differs in {key}')
            if ctx.is_main:
                (a.output/case).symlink_to(a.reuse_frozen.resolve(),target_is_directory=True)
                cases.append(json.loads((a.reuse_frozen/'metrics.json').read_text()))
            if ctx.world_size>1:dist.barrier()
            continue
        model=make_student(c,adapt,head['model']).to(ctx.device).eval()
        frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        initial_attention={n:p.detach().clone() for n,p in model.blocks[-1].attn.named_parameters()}
        audit=model.parameter_audit()
        if adapt and sum(v['count'] for k,v in audit.items() if '.attn.' in k)!=4198400:raise RuntimeError('Unexpected trainable attention set')
        wrapped=DDP(model,device_ids=[ctx.device.index],broadcast_buffers=False) if ctx.world_size>1 else model
        groups=[dict(params=model.projector.parameters(),lr=a.head_lr)]
        if adapt:groups.insert(0,dict(params=model.blocks[-1].attn.parameters(),lr=a.attention_lr))
        def optimizer_new():return torch.optim.AdamW(groups,weight_decay=0.,eps=1e-12)
        optimizer=optimizer_new()
        folder=a.output/case
        if ctx.is_main:folder.mkdir(exist_ok=False)
        evaluations=[];initial=evaluate(model,0)
        if ctx.is_main:evaluations.append(initial)
        history=[];exposures=torch.zeros(len(c['ids']),2,valid.shape[1],dtype=torch.long,device=ctx.device)
        start=time.monotonic()
        for update in range(1,a.updates+1):
            if update==21:optimizer=optimizer_new()
            steps=[int((i+a.seed+update-1)%int(valid[i].sum())) for i in local_train]
            with torch.no_grad():q=torch.stack([model(input_for(i,p),frames=p+1) for i,p in zip(local_train,steps)])
            leaf=q.detach().requires_grad_(True)
            source_ids=torch.tensor(local_train,device=ctx.device)
            mask=content[local_train].to(ctx.device)[:,None].expand(-1,2,-1)
            labels=y[local_train].to(ctx.device)[:,None,None].expand_as(mask)[mask]
            imageids=(source_ids*valid.shape[1]+torch.tensor(steps,device=ctx.device))[:,None,None].expand_as(mask)[mask]
            sources=source_ids[:,None,None].expand_as(mask)[mask]
            semantic,detail=patch_vico(leaf[mask],labels,imageids,sources,epoch=update,chunk_size=128)
            target=torch.stack([reference[i][:,p] for i,p in zip(local_train,steps)])
            keep=local_feature_retention(leaf,target,mask)*ctx.world_size*len(local_train)/len(train)
            loss=semantic+a.keep_weight*keep
            loss.backward();cotangent=leaf.grad.detach()
            optimizer.zero_grad(set_to_none=True)
            # Exactly the chain rule, without retaining all 64 source graphs.
            for j,(i,prefix) in enumerate(zip(local_train,steps)):
                sync=contextlib.nullcontext() if j==len(local_train)-1 or ctx.world_size==1 else wrapped.no_sync()
                with sync:
                    replay=wrapped(input_for(i,prefix),frames=prefix+1)
                    (replay*cotangent[j]).sum().backward()
            def norm(parameters):
                values=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
                return float(torch.stack(values).sum().sqrt()) if values else 0.
            attn_gradient=norm(model.blocks[-1].attn.parameters());head_gradient=norm(model.projector.parameters())
            if adapt and (not attn_gradient>0 or not torch.isfinite(torch.tensor(attn_gradient))):raise RuntimeError('Attention gradient is missing/nonfinite')
            gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
            optimizer.step()
            components=torch.stack((loss.detach(),semantic.detach(),keep.detach()))
            if ctx.world_size>1:dist.all_reduce(components);components/=ctx.world_size
            for i,prefix in zip(local_train,steps):exposures[i,:,prefix]+=1
            drift=max(float((p.detach()-initial_attention[n]).abs().max()) for n,p in model.blocks[-1].attn.named_parameters())
            row=dict(update=update,loss=float(components[0]),semantic_loss=float(components[1]),local_feature_retention=float(components[2]),
                attention_gradient_norm=attn_gradient,projector_gradient_norm=head_gradient,gradient_norm=float(gradient),attention_parameter_max_change=drift,
                seconds=time.monotonic()-start,source_exposures=len(train),patch_exposures=int(mask.sum())*ctx.world_size,
                same_image_weight_max_error=float(detail['same_image_weight_max_error']))
            history.append(row)
            if update%a.eval_every==0 or update==a.updates:
                result=evaluate(model,update,folder if update==a.updates else None)
                if ctx.is_main:
                    evaluations.append(result)
                    print(json.dumps(dict(case=case,**row,validation_accuracy=result['validation']['accuracy'])),flush=True)
                    (folder/'progress.json').write_text(json.dumps(dict(history=history,evaluations=evaluations),indent=2)+'\n')
        frozen_drift=max(float((p.cpu()-frozen[n]).abs().max()) for n,p in model.named_parameters() if n in frozen)
        if frozen_drift:raise RuntimeError('A frozen tensor changed')
        if ctx.world_size>1:dist.all_reduce(exposures)
        if ctx.is_main:
            metadata=dict(case=case,train_attention=adapt,trainable_parameters=audit,updates=a.updates,
                train_source_ids=[c['ids'][i] for i in train],validation_source_ids=[c['ids'][i] for i in val],
                frozen_parameter_max_drift=frozen_drift,attention_parameter_max_change=drift,seed=a.seed,
                attention_lr=a.attention_lr if adapt else 0.,head_lr=a.head_lr,local_keep_weight=a.keep_weight,
                optimizer_reset_before_update=21 if a.updates>=21 else None,world_size=ctx.world_size,
                train_rank_sources=counts,eval_rank_sources=evalcounts,initial_projector_sha256=digest(a.initial_projector),
                cache_sha256=digest(a.cache),precision=c['precision'],seconds=time.monotonic()-start,
                objective='Direct all-content-patch ViCo + local retention to immutable continuation-start student',
                temporal_branch='Original trajectory teacher/cache/checkpoint untouched; temporal objectives unchanged constants for this spatial-only continuation',
                mil=False,decoder=False,mirror_loss=False,verified_masks=False,independent_scene_test=False)
            torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                metadata=metadata,view_time_exposures=exposures.cpu(),history=history,configuration=vars(a)),folder/'checkpoint.pt')
            (folder/'evaluation_history.json').write_text(json.dumps(evaluations,indent=2)+'\n')
            (folder/'metrics.json').write_text(json.dumps(dict(metadata=metadata,final=evaluations[-1]),indent=2)+'\n')
            with (folder/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            cases.append(dict(metadata=metadata,final=evaluations[-1]))
        if ctx.world_size>1:dist.barrier()
        del wrapped,model,optimizer
    if ctx.is_main:
        paths=[Path(__file__),ROOT/'scripts/cache_final_attention.py',ROOT/'src/vision_model/training/final_attention.py',ROOT/'src/vision_model/models/final_block_maps.py',ROOT/'src/vision_model/training/patch_vico.py']
        (a.output/'report.json').write_text(json.dumps(dict(cases=cases,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch=torch.__version__),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
