#!/usr/bin/env python3
"""Matched normal-only projection versus spatial correspondence on two GPUs."""
import argparse,contextlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vision_model.training.normal_spatial_student import NormalSpatialProjector,correspondence_loss,patch_inputs
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from train_spatial_semantic import read_inputs,content_masks,DEFAULT_SPATIAL,DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import digest
from sklearn.metrics import roc_auc_score,average_precision_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=100)
    p.add_argument('--spatial-cache',type=Path,default=DEFAULT_SPATIAL);p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--temporal-checkpoint',type=Path,default=EXPERIMENT/'training/trial_01_checkpoint.pt')
    a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');ctx=initialize_distributed()
    if ctx.device.type!='cuda':raise RuntimeError('Run the matched real-data study on GPUs')
    c,raw,ck,train,val=read_inputs(a)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    normal=train[c['labels'][train]==0];assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(c['ids']))
    local=normal[ctx.rank::ctx.world_size].tolist()
    if len(normal)%ctx.world_size:raise ValueError('Matched normal-source gradient averaging needs equal real-source counts')
    support=content_masks(raw['manifest'],c['ids'],(16,16)).flatten(1)
    qslot,tslot=c['layers'].index(5),c['layers'].index(11)
    # Fit target geometry exclusively on training clean sources, without labels
    # from spill/validation clips. Same immutable transform in both arms.
    normal_values=[]
    for i in normal:
        x=patch_inputs(c['tokens'][i,:,c['valid'][i],tslot],support[i])
        normal_values.append(x[...,support[i],:].reshape(-1,x.shape[-1]))
    values=torch.cat(normal_values);mean=values.mean(0);std=values.std(0,unbiased=False)
    std=std.clamp_min(max(float(std.mean())*.05,1e-6))
    generator=torch.Generator().manual_seed(239)
    projection=torch.linalg.qr(torch.randn(1024,128,generator=generator),mode='reduced').Q
    # Cache only owned inputs/targets on each GPU.
    inputs={};targets={}
    for i in sorted(set(assigned+local)):
        inputs[i]=patch_inputs(c['tokens'][i,:,:,qslot].to(ctx.device),support[i].to(ctx.device))
        targets[i]=patch_inputs(c['tokens'][i,:,:,tslot].to(ctx.device),support[i].to(ctx.device),mean=mean,std=std,projection=projection)
    outcomes={};begin=time.monotonic()
    for arm,contrast in [('normal_cosine',False),('normal_contrast',True)]:
        torch.manual_seed(239)
        model=NormalSpatialProjector().to(ctx.device)
        wrapped=DDP(model,device_ids=[ctx.device.index]) if ctx.world_size>1 else model
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,eps=1e-12,weight_decay=0)
        history=[];evaluations=[];arm_dir=a.output/arm
        if ctx.is_main:arm_dir.mkdir()
        @torch.no_grad()
        def evaluate(update):
            rows=[]
            for i in assigned:
                q=model(inputs[i]);score=(1-(q*targets[i]).sum(-1)).clamp_min(0)
                score[:,:,~support[i]]=0;score[:,~c['valid'][i]]=0
                rows.append((i,score.cpu()))
            gathered=[None]*ctx.world_size if ctx.is_main else None
            if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
            else:gathered=[rows]
            if not ctx.is_main:return
            rows=sorted([r for part in gathered for r in part],key=lambda r:r[0])
            if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Repeated evaluation sources')
            maps=torch.stack([r[1] for r in rows])
            norm=torch.cat([maps[i,0,c['valid'][i]][:,support[i]].flatten() for i in normal])
            quantiles={str(v):float(torch.quantile(norm,v)) for v in (.5,.95,.99,.999)}
            rows_metrics=[]
            for i in range(len(c['ids'])):
                v=maps[i,0,c['valid'][i]][:,support[i]]
                rows_metrics.append(dict(id=c['ids'][i],mean_error=float(v.mean()),
                    mean_excess=float((v-quantiles['0.95']).clamp_min(0).mean()),above_clean_q95_fraction=float((v>quantiles['0.95']).float().mean())))
            metrics={}
            for split,idx in [('train',train),('validation',val)]:
                scores=[rows_metrics[i]['mean_excess'] for i in idx]
                metrics[split]=dict(sources=len(idx),roc_auc=float(roc_auc_score(c['labels'][idx].numpy(),scores)),
                    average_precision=float(average_precision_score(c['labels'][idx].numpy(),scores)),
                    clean_patch_false_alarm_rate=float(torch.cat([maps[i,0,c['valid'][i]][:,support[i]].flatten()>quantiles['0.95'] for i in idx if c['labels'][i]==0]).float().mean()))
            evaluation=dict(update=update,calibration=quantiles,metrics=metrics,sources=rows_metrics)
            evaluations.append(evaluation)
            torch.save(dict(maps=maps,ids=c['ids'],labels=c['labels'],valid=c['valid'],timestamps=c['timestamps'],content=support,
                train_indices=train,validation_indices=val,scope='Normal cross-layer feature disagreement; NOT original trajectory attention'),arm_dir/'maps.pt')
            (arm_dir/'evaluation_history.json').write_text(json.dumps(evaluations,indent=2)+'\n')
            print(json.dumps(dict(arm=arm,update=update,metrics=metrics)),flush=True)
        evaluate(0)
        for update in range(1,a.updates+1):
            if update==21:optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,eps=1e-12,weight_decay=0)
            optimizer.zero_grad(set_to_none=True);sums=torch.zeros(4,device=ctx.device)
            for position,i in enumerate(local):
                sync=wrapped.no_sync() if ctx.world_size>1 and position<len(local)-1 else contextlib.nullcontext()
                with sync:
                    q=wrapped(inputs[i])[:,c['valid'][i]][:,:,support[i]].flatten(0,1)
                    t=targets[i][:,c['valid'][i]][:,:,support[i]].flatten(0,1)
                    loss,details=correspondence_loss(q,t,contrast=contrast)
                    (loss/len(local)).backward()
                sums+=torch.stack([loss.detach(),*[details[k].detach() for k in ('cosine_loss','spatial_contrast_loss','margin_loss')]])/len(local)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            if ctx.world_size>1:dist.all_reduce(sums);sums/=ctx.world_size
            history.append(dict(update=update,loss=float(sums[0]),cosine_loss=float(sums[1]),spatial_contrast_loss=float(sums[2]),margin_loss=float(sums[3]),gradient_norm=float(norm)))
            if update%10==0 or update==a.updates:evaluate(update)
        if ctx.is_main:
            metadata=dict(arm=arm,updates=a.updates,seed=239,query_layer=5,target_layer=11,world_size=ctx.world_size,
                rank_eval_source_counts=counts,normal_train_sources=[c['ids'][i] for i in normal],fit_spill_sources=0,
                manual_annotations=0,decoder=False,mil=False,trajectory_optimizer_updates=0,backbone_optimizer_updates=0,
                objective='.7*cosine + .3*spatial_correspondence_CE + relu(.05-p_ii)' if contrast else 'mean same-position cosine error',
                scope='UniNet-inspired normal-only correspondence adaptation, not a UniNet reproduction or trajectory-decision localization',
                teacher_geometry='Per-current-tubelet content centering; training-clean channel whitening; frozen orthogonal 128-D projection',
                learning_rate=1e-3,temperature=.1,margin=.05,optimizer_reset_before_update=21,independent_scene_test=False,
                spatial_cache_sha256=digest(a.spatial_cache),trajectory_checkpoint_sha256=digest(a.temporal_checkpoint),
                source_code_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                    'src/vision_model/training/normal_spatial_student.py':digest(ROOT/'src/vision_model/training/normal_spatial_student.py')})
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,projection=projection,
                metadata=metadata,history=history,torch_rng_state=torch.get_rng_state()),arm_dir/'checkpoint.pt')
            (arm_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');outcomes[arm]=dict(metadata=metadata,final=evaluations[-1])
        del wrapped,model
        if ctx.world_size>1:dist.barrier()
    if ctx.is_main:
        (a.output/'results.json').write_text(json.dumps(dict(arms=outcomes,elapsed_seconds=time.monotonic()-begin),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
