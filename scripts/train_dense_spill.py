#!/usr/bin/env python3
"""Paired dense readout pilots; original four-loss trajectory stays immutable."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn.functional import all_gather
from vision_model.models.dense_encoder import file_sha256
from vision_model.training.dense_spill import DenseSpillReadout,require_candidate_coverage,source_balanced_partial_loss
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.vico import vico_pairwise_contrastive
from diagnose_spill_representation import recognition


def gather(value,grad=False):
    if not dist.is_initialized():return value
    if grad:return torch.cat(all_gather(value),0)
    out=[torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(out,value);return torch.cat(out,0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=100);p.add_argument('--control',action='store_true')
    p.add_argument('--cold-pixel-input',action='store_true')
    p.add_argument('--unnormalized-patches',action='store_true',help='CAM-style feature amplitudes; normalize only the episode embedding')
    a=p.parse_args(argv);torch.set_num_threads(4);torch.manual_seed(239)
    ctx=initialize_distributed()
    if a.updates<1:raise ValueError('Need positive update budget')
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    pseudo=torch.load(a.candidates,map_location='cpu',weights_only=False)
    meta=json.loads(a.candidates.with_name('report.json').read_text())
    if meta['metadata']['cache_sha256']!=file_sha256(a.cache) or pseudo['ids']!=c['ids']:
        raise ValueError('Candidate cache provenance mismatch')
    train,val=c['train_indices'],c['validation_indices'];labels=c['labels']
    if not torch.equal(pseudo['train_indices'],train) or not torch.equal(pseudo['validation_indices'],val):raise ValueError('Split mismatch')
    if (not torch.equal(pseudo['labels'],labels) or not torch.equal(pseudo['valid'],c['valid'])
        or not torch.equal(pseudo['content'],c['content']) or pseudo['targets'].shape!=c['tokens'].shape[:-1]):
        raise ValueError('Candidate labels/grid/validity mismatch')
    eligible=c['valid'][:,None,:,None]&c['content'].flatten(1)[:,None,None,:]
    if (pseudo['targets'].masked_select(~eligible.expand_as(pseudo['targets']))!=-1).any():
        raise ValueError('Automatic labels cannot supervise temporal/spatial padding')
    expected_clean={c['ids'][i] for i in train if labels[i]==0}
    if set(meta['metadata']['reference_sources'])!=expected_clean:raise ValueError('References must be exactly training clean sources')
    require_candidate_coverage(pseudo['targets'],labels,train)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(c['ids']))
    owned=train[ctx.rank::ctx.world_size]
    audit_distributed_indices(list(range(ctx.rank,len(train),ctx.world_size)),len(train))
    sizes=gather(torch.tensor([len(owned)],device=ctx.device)).tolist()
    if len(set(sizes))!=1:raise ValueError('Training requires equal unpadded ownership')
    x=c['tokens'][owned].flatten(0,1).to(ctx.device).float()
    cold_scale=None
    cold_input=None
    if a.cold_pixel_input:
        if meta['metadata'].get('input_alignment')!='encoder_input_uint8_reconstruction_v1':
            raise ValueError('Pixel input requires encoder-aligned descriptors for fresh-input replay')
        cold_scale=meta['calibration']['high_q999']
        if not cold_scale>0:raise ValueError('Degenerate cold-input scale')
        cold=pseudo['cold_contrast'];side=c['content'].shape[-1]
        cold_input=torch.stack((cold,cold.reshape(*cold.shape[:-1],side,side).flip(-1).flatten(2)),1)/cold_scale
        x=torch.cat((x,cold_input[owned].flatten(0,1).to(ctx.device)[...,None]),-1)
    valid=c['valid'][owned,None].expand(-1,2,-1).flatten(0,1).to(ctx.device)
    support=c['content'][owned].flatten(1)[:,None].expand(-1,2,-1).flatten(0,1).to(ctx.device)
    target=pseudo['targets'][owned].to(ctx.device);y=labels[owned].to(ctx.device)
    global_y=gather(y).repeat_interleave(2)
    global_ids=gather(owned.to(ctx.device)).repeat_interleave(2)
    model=DenseSpillReadout(width=x.shape[-1],normalize_patches=not a.unnormalized_patches).to(ctx.device)

    @torch.no_grad()
    def refresh():
        z=gather(model(x,valid,support)['embedding'])
        model.set_prototypes(torch.stack([z[global_y==label].mean(0) for label in (0,1)]))
    refresh()
    out=model(x,valid,support)
    sem,_=vico_pairwise_contrastive(gather(out['embedding'],True),global_y,source_ids=global_ids,epoch=20,curriculum_epochs=20)
    local,_=source_balanced_partial_loss(out['local_logits'].reshape_as(target),target,y)
    parameters=tuple(model.parameters());vectors=[]
    for loss in (sem,local):
        gradients=torch.autograd.grad(loss,parameters,retain_graph=True)
        v=torch.cat([g.flatten() for g in gradients])
        if ctx.world_size>1:dist.all_reduce(v);v/=ctx.world_size
        vectors.append(v)
    semantic_norm,local_norm=[float(v.norm()) for v in vectors]
    weight=min(10.,max(.01,.5*semantic_norm/max(local_norm,1e-12)))
    calibrated=weight
    if a.control:weight=0.
    del out,sem,local,vectors
    wrapped=DDP(model,device_ids=[ctx.local_rank] if ctx.device.type=='cuda' else None,
                broadcast_buffers=False) if ctx.world_size>1 else model
    def optimizer():return torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.,eps=1e-12)
    opt=optimizer();history=[];evaluations=[]

    @torch.no_grad()
    def evaluate(update):
        rows=[]
        for i in assigned:
            vv=c['valid'][i:i+1].expand(2,-1).to(ctx.device)
            cc=c['content'][i].flatten()[None].expand(2,-1).to(ctx.device)
            patches=c['tokens'][i].to(ctx.device).float()
            if cold_input is not None:patches=torch.cat((patches,cold_input[i].to(ctx.device)[...,None]),-1)
            output=model(patches,vv,cc)
            error=float(((output['contributions']*output['valid']).sum((1,2))/output['valid'].sum((1,2))-output['margin']).abs().max())
            if error>2e-6:raise RuntimeError('Contribution/decision mismatch')
            rows.append((i,{k:v.cpu() for k,v in output.items()},error))
        pieces=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
        else:pieces=[rows]
        if ctx.is_main:
            rows=sorted([r for part in pieces for r in part],key=lambda r:r[0])
            if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Repeated source')
            z=torch.stack([r[1]['embedding'][0] for r in rows])
            margin=torch.stack([r[1]['margin'][0] for r in rows])
            detail=dict(update=update,classification_correct=int(((margin[val]>0)==labels[val]).sum()),sources=len(val),
                recognition=recognition(z,labels,train,val),contribution_replay_max_error=max(r[2] for r in rows))
            evaluations.append(detail)
            torch.save(dict(ids=c['ids'],labels=labels,valid=c['valid'],content=c['content'],
                train_indices=train,validation_indices=val,prototypes=model.prototypes.cpu(),
                embeddings=z,rows=rows),a.output/'representations.pt')
            print(json.dumps(detail),flush=True)

    evaluate(0)
    for update in range(1,a.updates+1):
        if update==21:opt=optimizer()
        opt.zero_grad(set_to_none=True)
        out=wrapped(x,valid,support)
        sem,_=vico_pairwise_contrastive(gather(out['embedding'],True),global_y,source_ids=global_ids,epoch=update,curriculum_epochs=20)
        local,detail=source_balanced_partial_loss(out['local_logits'].reshape_as(target),target,y)
        loss=sem+weight*local;loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        values=torch.stack([loss.detach(),sem.detach(),local.detach()])
        if ctx.world_size>1:dist.all_reduce(values);values/=ctx.world_size
        history.append(dict(update=update,total_loss=float(values[0]),semantic_loss=float(values[1]),
            local_loss=float(values[2]),weighted_local_loss=weight*float(values[2]),gradient_norm=float(norm)))
        refresh()
        if update%10==0 or update==a.updates:evaluate(update)
    if ctx.is_main:
        metadata=dict(seed=239,optimizer_updates=a.updates,world_size=ctx.world_size,rank_source_counts=counts,
            training_rank_source_counts=sizes,train_source_ids=[c['ids'][i] for i in train],
            validation_source_ids=[c['ids'][i] for i in val],
            optimizer_reset_before_update=21,model_configuration=model.configuration,
            supervision='automatic_unverified',fg_bg_weight=weight,calibrated_weight=calibrated,
            calibration=dict(semantic_gradient_norm=semantic_norm,local_gradient_norm=local_norm,target_ratio=.5),
            objective='L_ViCo(dense_episode) + weight * source_balanced_within_frame_partial_BCE',
            original_trajectory='Immutable original four-loss recognition/temporal control; not retrained in this dense pilot',
            cold_pixel_input=a.cold_pixel_input,cold_input_scale=cold_scale,
            prototype_policy='Recomputed only from training embeddings after every update; detached selection statistics',
            cache_sha256=file_sha256(a.cache),candidate_sha256=file_sha256(a.candidates),code_sha256=file_sha256(__file__),
            manual_annotations=0,decoder=False,mil=False,independent_scene_test=False,localization_validated=False)
        exposures=torch.zeros(len(c['ids']),2,c['valid'].shape[1],dtype=torch.long)
        exposures[train]=c['valid'][train,None].expand(-1,2,-1).long()*a.updates
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),metadata=metadata,history=history,
                        view_time_exposures=exposures,torch_rng_state=torch.get_rng_state()),a.output/'checkpoint.pt')
        (a.output/'report.json').write_text(json.dumps(dict(metadata=metadata,evaluations=evaluations),indent=2)+'\n')
        with (a.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
