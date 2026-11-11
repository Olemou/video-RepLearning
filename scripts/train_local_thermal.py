#!/usr/bin/env python3
"""Train finite-field local outputs; preserve original four-loss trajectory."""
import argparse,json,sys,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vision_model.training.local_thermal import LocalThermalReadout,packed_local_pairs,packed_pair_terms
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.dense_encoder import file_sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=100);p.add_argument('--control',action='store_true')
    p.add_argument('--local-conv',action='store_true');a=p.parse_args(argv)
    torch.set_num_threads(4);torch.manual_seed(239);ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',mmap=True,weights_only=False)
    pairdata=torch.load(a.pairs,map_location='cpu',weights_only=False)
    if pairdata['cache_sha256']!=file_sha256(a.cache) or pairdata['ids']!=c['ids']:raise ValueError('Pair/cache identity mismatch')
    if a.updates<1:raise ValueError('Need positive budget')
    train,val=c['train_indices'],c['validation_indices']
    if set(train.tolist())&set(val.tolist()) or sorted(train.tolist()+val.tolist())!=list(range(len(c['ids']))):raise ValueError('Bad split')
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,eval_counts=audit_distributed_indices(assigned,len(c['ids']))
    owned=train[ctx.rank::ctx.world_size];_,train_counts=audit_distributed_indices(list(range(ctx.rank,len(train),ctx.world_size)),len(train))
    x=c['features'][owned].to(ctx.device).float();valid=c['valid'][owned].to(ctx.device);labels=c['labels'][owned].to(ctx.device)
    b,v,t,ch,h,w=x.shape;flat=x.flatten(0,2)
    packed=packed_local_pairs(pairdata['pairs'],owned,c['valid'][owned],(h,w),ctx.device)
    model=LocalThermalReadout(channels=ch,kind='local_conv' if a.local_conv else 'pointwise').to(ctx.device)
    logits=model(flat).reshape(b,v,t,h,w)
    absence,ranking=packed_pair_terms(logits,labels,valid,packed)
    norms=[]
    for loss in (absence,ranking):
        gg=torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)])
        if ctx.world_size>1:dist.all_reduce(gg);gg/=ctx.world_size
        norms.append(float(gg.norm()))
    weight=min(10.,max(.01,.5*norms[0]/max(norms[1],1e-12)))
    if a.control:weight=0.
    wrapped=DDP(model,device_ids=[ctx.local_rank] if ctx.device.type=='cuda' else None) if ctx.world_size>1 else model
    def optimizer():return torch.optim.AdamW(model.parameters(),lr=1e-3,eps=1e-12,weight_decay=0)
    opt=optimizer();history=[]
    del logits,absence,ranking
    for update in range(1,a.updates+1):
        if update==21:opt=optimizer()
        opt.zero_grad(set_to_none=True)
        logits=wrapped(flat).reshape(b,v,t,h,w)
        absence,ranking=packed_pair_terms(logits,labels,valid,packed)
        total=absence+weight*ranking;total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        values=torch.stack([total.detach(),absence.detach(),ranking.detach()])
        if ctx.world_size>1:dist.all_reduce(values);values/=ctx.world_size
        history.append(dict(update=update,total=float(values[0]),absence=float(values[1]),pair=float(values[2]),weighted_pair=float(values[2])*weight,gradient_norm=float(norm)))
        if ctx.is_main and update%10==0:print(json.dumps(history[-1]),flush=True)
    local=[]
    with torch.no_grad():
        for i in assigned:
            z=c['features'][i].to(ctx.device).float();shape=z.shape
            scores=model(z.flatten(0,1)).reshape(*shape[:2],h,w).cpu()
            local.append((i,scores))
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(local,pieces,dst=0)
    else:pieces=[local]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r[0])
        if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Duplicated eval sources')
        maps=torch.stack([r[1] for r in rows])
        clean=train[c['labels'][train]==0]
        normal=maps[clean,0].flatten();threshold=float(torch.quantile(normal,.995))
        evidence=(maps[train,0]-threshold).clamp_min(0)
        scale=max(float(torch.quantile(evidence.flatten(),.999)),1e-6)
        scope=[dict(id=c['ids'][i],split='train' if i in train else 'selection',label=int(c['labels'][i]),
            mean_score=float(maps[i,0].mean()),above_clean_threshold_fraction=float((maps[i,0]>threshold).float().mean()),
            pairs=sum(map(len,pairdata['pairs'][i]))) for i in range(len(maps))]
        metadata=dict(seed=239,optimizer_updates=a.updates,world_size=ctx.world_size,rank_source_counts=eval_counts,
            training_rank_source_counts=train_counts,configuration=model.configuration,pair_weight=weight,
            initial_gradient_norms=dict(absence=norms[0],pair=norms[1]),optimizer_reset_before_update=21,
            loss='Training-clean absence + calibrated same-image region/ring ranking',
            temporal_consistency_weight=0.,temporal_consistency_reason='Registration/occlusion correspondence not validated',
            original_four_loss_trajectory='Unchanged; this is a separate local detector, not backbone attention',
            maximum_receptive_radius=39+model.extra_receptive_radius,
            code_sha256=file_sha256(__file__),module_sha256=file_sha256(ROOT/'src/vision_model/training/local_thermal.py'),
            cache_sha256=file_sha256(a.cache),pairs_sha256=file_sha256(a.pairs),manual_annotations=0,mil=False,decoder=False,
            localization_validated=False,independent_scene_test=False)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),metadata=metadata,history=history,
            train_indices=train,validation_indices=val,view_exposures=c['valid'][train,None].expand(-1,2,-1).long()*a.updates,
            torch_rng_state=torch.get_rng_state()),a.output/'checkpoint.pt')
        torch.save(maps,a.output/'maps.pt')
        (a.output/'report.json').write_text(json.dumps(dict(metadata=metadata,history=history,sources=scope,
            calibration=dict(clean_q995=threshold,evidence_scale=scale,fit='Only training maps; scaleq99.9 over all train evidence; no per-frame normalization')),
            indent=2)+'\n')
        with (a.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
