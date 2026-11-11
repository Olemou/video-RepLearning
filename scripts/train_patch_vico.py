#!/usr/bin/env python3
"""Controlled all-patch ViCo versus fixed-mean and positive-weight controls.

All content patches participate; one deterministic causal tubelet per source
per update bounds cost. No MIL, decoder, mask labels or attention regularizer.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from vision_model.training.patch_vico import PatchProjector,patch_vico,local_feature_retention
from vision_model.training.vico import vico_pairwise_contrastive
from vision_model.training.trajectory import TrajectoryModel
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from train_spatial_semantic import read_inputs,content_masks,DEFAULT_SPATIAL,DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import recognition,digest


def gather_embeddings(value):
    if not dist.is_initialized():return value
    from torch.distributed.nn.functional import all_gather
    counts=[None]*dist.get_world_size();dist.all_gather_object(counts,len(value))
    padded=torch.cat((value,value.new_zeros(max(counts)-len(value),*value.shape[1:])))
    return torch.cat([v[:n] for v,n in zip(all_gather(padded),counts)])


def patch_diagnostics(q,valid,content,prototypes):
    """Class-score diagnostics connected exactly to fixed-mean classification."""
    b,v,t,n,d=q.shape
    support=valid[:,None,:,None] & content[:,None,None]
    q=q.masked_fill(~support[...,None],0)
    counts=content.sum(-1)
    image_mean=q.sum(-2)/counts[:,None,None,None]
    mean=(image_mean*valid[:,None,:,None]).sum(2)/valid.sum(-1)[:,None,None]
    embeddings=F.normalize(mean,dim=-1)
    # The masked mean of these signed maps equals the episode prototype margin.
    maps=(q@(prototypes[1]-prototypes[0]))/mean.norm(dim=-1).clamp_min(1e-12)[:,:,None,None]
    maps=maps.masked_fill(~support,0)
    patch_predictions=(q@prototypes.T).argmax(-1)
    mean_norm=image_mean.square().sum(-1)
    variance=(1-mean_norm).clamp_min(0)
    offdiag=(counts[:,None,None]*mean_norm-1)/(counts[:,None,None]-1).clamp_min(1)
    logits=maps/.1
    attention=logits.masked_fill(~support,-torch.inf)
    # Invalid time rows get a dummy key, then are zeroed.
    attention=attention.masked_fill(~valid[:,None,:,None],0).softmax(-1).masked_fill(~support,0)
    entropy=-(attention*attention.clamp_min(1e-30).log()).sum(-1)/counts.log()[:,None,None]
    def average(x):return (x*valid[:,None]).sum((1,2))/(v*valid.sum(1))
    centered=q-image_mean[...,None,:]
    centered=centered.masked_fill(~support[...,None],0)
    # Gram and feature covariance have the same nonzero eigenvalues. The
    # spatial Gram avoids a 1024x1024 matrix per raw-backbone tubelet.
    gram=centered@centered.transpose(-1,-2)/counts[:,None,None,None,None]
    trace=gram.diagonal(dim1=-2,dim2=-1).sum(-1)
    participation=trace.square()/gram.square().sum((-2,-1)).clamp_min(1e-30)
    side=int(n**.5)
    mirror=q[:,1].reshape(b,t,side,side,d).flip(-2).reshape(b,t,n,d)
    matched=(q[:,0]*mirror).sum(-1)
    cosine=q[:,0]@mirror.transpose(-1,-2)
    cosine=cosine.masked_fill(~content[:,None,None],-torch.inf)
    predicted=cosine.argmax(-1)
    correspondence=(predicted==torch.arange(n)[None,None]) & content[:,None]
    denom=valid.sum(1)*counts
    diag=dict(within_image_feature_variance=average(variance),within_image_patch_cosine=average(offdiag),
        participation_rank=average(participation),normalized_class_map_entropy=average(entropy),
        mirror_matched_cosine=(matched*support[:,0]).sum((1,2))/denom,
        mirror_correspondence_top1=(correspondence*valid[:,:,None]).sum((1,2))/denom,
        class_map_spatial_std=average(((maps-(maps.sum(-1)/counts[:,None,None])[...,None]).square().masked_fill(~support,0).sum(-1)/counts[:,None,None]).sqrt()),
        signed_map_mirror_MAE=((maps[:,0]-maps[:,1].reshape(b,t,side,side).flip(-1).reshape(b,t,n)).abs()*support[:,0]).sum((1,2))/denom)
    margin=embeddings@(prototypes[1]-prototypes[0])
    replay=(maps.sum((-2,-1))/(valid.sum(1)*counts)[:,None]-margin).abs().max()
    return embeddings,maps,diag,patch_predictions,float(replay)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spatial-cache',type=Path,default=DEFAULT_SPATIAL)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--temporal-checkpoint',type=Path,default=EXPERIMENT/'training/trial_01_checkpoint.pt')
    p.add_argument('--cases',nargs='+',choices=['global_mean','all_patch','uniform_positive'],default=['global_mean','all_patch','uniform_positive'])
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--seed',type=int,default=239)
    p.add_argument('--layer',type=int,default=23)
    p.add_argument('--chunk-size',type=int,default=128)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--local-keep-weight',type=float,default=0.,help='Proposed local feature preservation; distinct from temporal L_keep')
    a=p.parse_args()
    if min(a.updates,a.eval_every,a.chunk_size)<1 or len(set(a.cases))!=len(a.cases) or a.local_keep_weight<0:raise ValueError('Invalid budget/cases')
    if a.local_keep_weight and a.cases!=['all_patch']:raise ValueError('Local-retention experiment requires --cases all_patch')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    ctx=initialize_distributed()
    c,raw,ck,train,val=read_inputs(a)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=torch.arange(ctx.rank,len(c['ids']),ctx.world_size)
    _,evaluation_counts=audit_distributed_indices(assigned.tolist(),len(c['ids']))
    content=content_masks(raw['manifest'],c['ids'],(16,16)).flatten(1)
    slot=c['layers'].index(a.layer)
    bank=c['tokens'][assigned,:,:,slot].to(ctx.device)
    local_content=content[assigned].to(ctx.device)
    local_valid=c['valid'][assigned].to(ctx.device)
    local_train=torch.tensor([j for j,i in enumerate(assigned.tolist()) if i in train.tolist()],device=ctx.device)
    source_ids=assigned[local_train.cpu()].to(ctx.device)
    shards=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(shards,source_ids.tolist())
    else:shards=[source_ids.tolist()]
    if sorted(i for shard in shards for i in shard)!=sorted(train.tolist()):raise RuntimeError('Training source ownership mismatch')
    if min(map(len,shards))==0:raise ValueError('Every rank requires training sources')
    architecture=dict(width=1024,hidden=256,embedding_dim=256)
    teacher=TrajectoryModel(ck['width'],ck['hidden'],ck['embedding_dim']).to(ctx.device).eval().requires_grad_(False)
    teacher.load_state_dict(ck['model'],strict=True)
    tx=raw['features'][assigned].to(ctx.device)
    tv=raw['valid'][assigned].to(ctx.device)
    tt=raw['timestamps'][assigned].to(ctx.device)
    @torch.no_grad()
    def temporal_probe():
        out=teacher(tx.flatten(0,1),tv[:,None].expand(-1,2,-1).flatten(0,1),tt[:,None].expand(-1,2,-1).flatten(0,1))
        stamps=tt[:,None].expand(-1,2,-1).flatten(0,1)
        for horizon in (1,2,4):
            if horizon<stamps.shape[1]:
                state=F.layer_norm(out['states'],(ck['width'],))[:,:-horizon]
                out[f'future_{horizon}']=state+teacher.future_delta(torch.cat((out['dynamics'][:,:-horizon],(stamps[:,horizon:]-stamps[:,:-horizon])[...,None]),-1))
        return {k:v.clone() for k,v in out.items()}
    teacher_before=temporal_probe()
    torch.manual_seed(a.seed)
    initial=PatchProjector(**architecture).to(ctx.device)
    initial_state=copy.deepcopy(initial.state_dict())
    local_teacher=copy.deepcopy(initial).eval().requires_grad_(False)
    @torch.no_grad()
    def evaluate(model,step,save_dir=None,raw_baseline=False):
        model.eval()
        rows=[]
        for j,i in enumerate(assigned.tolist()):
            values=bank[j].float()
            support=local_valid[j][None,:,None] & local_content[j][None,None]
            values=values.masked_fill(~support[...,None],0)
            q=F.normalize(values,dim=-1) if raw_baseline else model(values)
            q=q.masked_fill(~support[...,None],0)
            rows.append((i,q.cpu()))
        gathered=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
        else:gathered=[rows]
        if not ctx.is_main:return None
        rows=sorted([row for shard in gathered for row in shard],key=lambda row:row[0])
        if [i for i,_ in rows]!=list(range(len(c['ids']))):raise RuntimeError('Duplicate/missing evaluation source')
        q=torch.stack([value for _,value in rows])
        support=c['valid'][:,None,:,None] & content[:,None,None]
        mean=q.sum((2,3))/(c['valid'].sum(1)*content.sum(1))[:,None,None]
        z=F.normalize(mean,dim=-1)
        prototypes=torch.stack([F.normalize(z[train,0][c['labels'][train]==label].mean(0),dim=0) for label in (0,1)])
        z,maps,diag,patch_predictions,replay=patch_diagnostics(q,c['valid'],content,prototypes)
        result=dict(update=step,map_mean_margin_replay_error=replay)
        for split,indices in (('train',train),('validation',val)):
            metrics=recognition(z[:,0],c['labels'],train,indices,exclude_self=split=='train',prototypes=prototypes)
            _,detail=vico_pairwise_contrastive(z[indices].flatten(0,1),c['labels'][indices].repeat_interleave(2),source_ids=indices.repeat_interleave(2),epoch=20,curriculum_epochs=20)
            metrics.update({key:float(value) for key,value in detail.items()})
            metrics.update({key:float(value[indices].mean()) for key,value in diag.items()})
            patch_correct=(patch_predictions[indices]==c['labels'][indices,None,None,None]) & support[indices]
            metrics['weak_patch_class_accuracy']=float(patch_correct.sum()/support[indices].expand(-1,2,-1,-1).sum())
            metrics['mirrored_accuracy']=float(((z[indices,1]@prototypes.T).argmax(-1)==c['labels'][indices]).float().mean())
            metrics['patch_label_interpretation']='Source-image association, not verified region correctness'
            result[split]=metrics
        result['per_video']=[dict(id=name,label=int(c['labels'][i]),split='train' if i in train else 'validation',**{key:float(value[i]) for key,value in diag.items()}) for i,name in enumerate(c['ids'])]
        if save_dir:
            torch.save(dict(ids=c['ids'],labels=c['labels'],valid=c['valid'],content=content,timestamps=c['timestamps'],
                q=q,embeddings=z,prototypes=prototypes,signed_maps=maps,diagnostics=diag),save_dir/'representations.pt')
        return result
    raw_metrics=evaluate(initial,0,raw_baseline=True)
    initial_metrics=evaluate(initial,0)
    if ctx.is_main:
        (a.output/'initial_metrics.json').write_text(json.dumps(dict(frozen_raw=raw_metrics,untrained_projector=initial_metrics),indent=2)+'\n')
    del initial
    cases=[]
    for name in a.cases:
        folder=a.output/name
        if ctx.is_main:folder.mkdir()
        torch.manual_seed(a.seed)
        model=PatchProjector(**architecture).to(ctx.device)
        model.load_state_dict(initial_state,strict=True)
        wrapped=DDP(model,device_ids=[ctx.local_rank] if ctx.device.type=='cuda' else None,broadcast_buffers=False) if ctx.world_size>1 else model
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,eps=1e-12,weight_decay=0.)
        history=[];evaluations=[]
        exposures=torch.zeros(len(c['ids']),2,bank.shape[2],dtype=torch.long,device=ctx.device)
        started=time.monotonic()
        if ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for update in range(1,a.updates+1):
            if update==21:optimizer.state.clear()
            wrapped.train();optimizer.zero_grad(set_to_none=True)
            steps=(source_ids+a.seed+update-1)%local_valid[local_train].sum(1)
            original_images=source_ids*bank.shape[2]+steps
            patches=bank[local_train,:,steps].float()
            mask=local_content[local_train,None].expand(-1,2,-1)
            q=wrapped(patches.masked_fill(~mask[...,None],0))
            if name=='global_mean':
                episode=F.normalize(q.masked_fill(~mask[...,None],0).sum(-2)/mask.sum(-1)[...,None],dim=-1)
                z=gather_embeddings(episode.flatten(0,1))
                labels=gather_embeddings(c['labels'][source_ids.cpu()].to(ctx.device).repeat_interleave(2))
                images=gather_embeddings(original_images.repeat_interleave(2))
                loss,details=vico_pairwise_contrastive(z,labels,source_ids=images,epoch=update,curriculum_epochs=20)
            else:
                labels=c['labels'][source_ids.cpu()].to(ctx.device)[:,None,None].expand_as(mask)[mask]
                images=original_images[:,None,None].expand_as(mask)[mask]
                sources=source_ids[:,None,None].expand_as(mask)[mask]
                loss,details=patch_vico(q[mask],labels,images,sources,epoch=update,chunk_size=a.chunk_size,
                    positive_weighting=name=='all_patch')
            semantic=loss
            local_keep=loss.new_zeros(())
            if a.local_keep_weight:
                with torch.no_grad():target=local_teacher(patches.masked_fill(~mask[...,None],0))
                local_keep=local_feature_retention(q,target,mask)*ctx.world_size*len(source_ids)/len(train)
                loss=semantic+a.local_keep_weight*local_keep
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            components=torch.stack((loss.detach(),semantic.detach(),local_keep.detach()))
            if ctx.world_size>1:dist.all_reduce(components);components/=ctx.world_size
            reported=components[0]
            exposures[source_ids,:,steps]+=1
            history.append(dict(update=update,loss=float(reported),gradient_norm=float(gradient),
                semantic_loss=float(components[1]),local_feature_retention=float(components[2]),local_keep_weight=a.local_keep_weight,
                seconds=time.monotonic()-started,valid_patch_exposures=int(mask.sum())*ctx.world_size,
                source_exposures=len(train),same_image_weight_max_error=float(details.get('same_image_weight_max_error',0.))))
            if update%a.eval_every==0 or update==a.updates:
                result=evaluate(model,update,folder if update==a.updates else None)
                if ctx.is_main:
                    evaluations.append(result)
                    print(json.dumps(dict(case=name,update=update,loss=float(reported),validation_accuracy=result['validation']['accuracy'],
                        within_image_cosine=result['validation']['within_image_patch_cosine'],
                        correspondence=result['validation']['mirror_correspondence_top1'],elapsed=time.monotonic()-started)),flush=True)
        after=temporal_probe()
        drift=max(float((after[k]-teacher_before[k]).abs().max()) for k in after)
        parameter_drift=max(float((v.cpu()-ck['model'][k]).abs().max()) for k,v in teacher.state_dict().items())
        checked=torch.tensor([drift,parameter_drift],device=ctx.device)
        if ctx.world_size>1:dist.all_reduce(checked,op=dist.ReduceOp.MAX);dist.all_reduce(exposures)
        if checked.any():raise RuntimeError('Frozen temporal baseline changed')
        if ctx.is_main:
            metadata=dict(case=name,seed=a.seed,updates=a.updates,optimizer_state_reset_before_update=21 if a.updates>=21 else None,
                architecture=architecture,layer=a.layer,learning_rate=a.learning_rate,
                local_keep_weight=a.local_keep_weight,local_teacher='Immutable initial projector on frozen V-JEPA patches',
                temporal_output_and_future_max_drift=float(checked[0]),temporal_parameter_max_drift=float(checked[1]),
                seconds=time.monotonic()-started,world_size=ctx.world_size,train_rank_sources=list(map(len,shards)),eval_rank_sources=evaluation_counts,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if ctx.device.type=='cuda' else 0,
                spatial_patches='All content-valid patches, both actual cached views',tubelet_schedule='(global_source_index+seed+update-1) modulo valid prefix count',
                train_source_ids=ck['train_source_ids'],validation_source_ids=ck['validation_source_ids'],
                image_id='source_index * max_prefix_count + selected_prefix_index',
                weak_patch_labels=True,verified_masks=False,mil=False,decoder=False,mirror_attention_loss=False,
                optimized_loss='Original episode ViCo on fixed mean' if name=='global_mean' else 'Source-balanced direct patch ViCo; same-image positives weight one; optional weighted local feature retention')
            torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                metadata=metadata,view_time_exposures=exposures.cpu(),history=history,configuration=vars(a),
                local_teacher={k:v.cpu() for k,v in local_teacher.state_dict().items()} if a.local_keep_weight else None,
                cache_provenance={k:c[k] for k in ('cache_sha256','checkpoint_sha256','encoder_sha256','temporal_context_policy','normalization','views')}),folder/'checkpoint.pt')
            (folder/'evaluation_history.json').write_text(json.dumps(evaluations,indent=2)+'\n')
            (folder/'metrics.json').write_text(json.dumps(dict(metadata=metadata,final=evaluations[-1]),indent=2)+'\n')
            with (folder/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            cases.append(dict(**metadata,metrics=evaluations[-1]))
        if ctx.world_size>1:dist.barrier()
        del wrapped,model,optimizer
    if ctx.is_main:
        paths=[Path(__file__),ROOT/'src/vision_model/training/patch_vico.py',ROOT/'src/vision_model/training/vico.py',ROOT/'src/vision_model/training/trajectory.py',ROOT/'vico.md']
        (a.output/'report.json').write_text(json.dumps(dict(cases=cases,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            initial=initial_metrics,frozen_raw=raw_metrics,source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch=torch.__version__,
            selection_split=True,independent_scene_test=False,spatial_annotation=False),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
