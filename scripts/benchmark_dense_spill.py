#!/usr/bin/env python3
"""Training-clean reference fitting and honest within-spill candidate audit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.normal_patch_reference import NormalPatchReference,automatic_candidates
from vision_model.training.dense_spill import candidate_coverage
from vision_model.models.dense_encoder import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.cache,map_location='cpu',mmap=True,weights_only=False)
    train,val=c['train_indices'],c['validation_indices'];normal=train[c['labels'][train]==0]
    if len(normal)<2:raise ValueError('Need independent clean sources')
    clean_ids=[c['ids'][i] for i in normal]
    support=c['content'].flatten(1);side=c['content'].shape[-1]
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(c['ids']))
    maximum=max(c['tokens'].shape[2]+len(c['rows'][i]['normal_extra']) for i in normal)
    bank=torch.zeros(len(normal),2,maximum,c['tokens'].shape[-2],c['tokens'].shape[-1],dtype=torch.float16)
    normal_valid=torch.zeros(len(normal),maximum,dtype=torch.bool)
    for slot,index in enumerate(normal.tolist()):
        n=int(c['valid'][index].sum());extra=c['rows'][index]['normal_extra']
        bank[slot,:,:n]=c['tokens'][index,:,:n]
        for j,tokens in enumerate(extra):bank[slot,:,n+j]=tokens
        normal_valid[slot,:n+len(extra)]=True
    bank=bank.to(ctx.device);allmaps={};calibration={}
    for mode in ['raw','residual']:
        ref=NormalPatchReference(mode=mode,per_source=1024).fit(bank,normal_valid.to(ctx.device),support[normal].to(ctx.device),clean_ids,c['labels'][normal])
        rows=[]
        for i in assigned:
            values=ref.score(c['tokens'][i].to(ctx.device),support[i].to(ctx.device),exclude_source=c['ids'][i]).cpu()
            values[:,~c['valid'][i]]=0
            rows.append((i,values))
        pieces=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
        else:pieces=[rows]
        if ctx.is_main:
            rows=sorted([r for part in pieces for r in part],key=lambda r:r[0])
            if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Duplicate/missing sources')
            maps=torch.stack([r[1] for r in rows]);allmaps[mode]=maps
            clean=torch.cat([maps[i,0,c['valid'][i]][:,support[i]].flatten() for i in normal])
            calibration[mode]={str(q):float(torch.quantile(clean,q)) for q in [.5,.95,.99,.999]}
            torch.save(ref.state_dict(),a.output/f'{mode}_reference.pt')
        del ref
    if ctx.is_main:
        candidates=[]
        for mode in ['raw','residual']:
            q=calibration[mode]
            candidates.append(automatic_candidates(allmaps[mode],c['valid'],support,low=q['0.95'],high=q['0.99'])
                              if q['0.99']>q['0.95'] else torch.full_like(allmaps[mode],-1,dtype=torch.long))
        targets=torch.full_like(candidates[0],-1)
        targets[(candidates[0]==0)&(candidates[1]==0)]=0
        targets[(candidates[0]==1)&(candidates[1]==1)]=1
        for i in torch.where(c['labels']==0)[0]:
            for t in torch.where(c['valid'][i])[0]:targets[i,:,t,support[i]]=0
        coverage=candidate_coverage(targets,c['labels'],train)
        rows=[]
        for i in range(len(targets)):
            values=allmaps['raw'][i,0,c['valid'][i]][:,support[i]]
            rows.append(dict(id=c['ids'][i],index=i,label=int(c['labels'][i]),split='train' if i in train else 'validation_selection',
                foreground=int((targets[i,0]==1).sum()),background=int((targets[i,0]==0).sum()),
                same_frame_fg_bg=int(((targets[i,0]==1).any(-1)&(targets[i,0]==0).any(-1)).sum()),
                mean_excess=float((values-calibration['raw']['0.95']).clamp_min(0).mean())))
        auc=float(roc_auc_score(c['labels'][val],[rows[i]['mean_excess'] for i in val]))
        result=dict(metadata=dict(cache_sha256=file_sha256(a.cache),cache=str(a.cache.resolve()),
            code_sha256=file_sha256(__file__),reference_sources=clean_ids,reference_views='identity/mirror',
            extra_training_clean_windows=sum(len(c['rows'][i]['normal_extra']) for i in normal),
            world_size=ctx.world_size,rank_source_counts=counts,manual_annotations=0,
            supervision_kind='automatic_unverified',grid=[side,side],
            policy='Raw AND content-centered normal matches, clean source-excluded q95 BG/q99 FG, mirror agreement, robust spatial FG excess; no forced area'),
            calibration=calibration,coverage=coverage,validation_novelty_auc=auc,sources=rows,
            interpretation='Novelty and unverified candidates only, not liquid localization accuracy')
        torch.save(dict(ids=c['ids'],labels=c['labels'],valid=c['valid'],content=c['content'],maps=allmaps,
            targets=targets,train_indices=train,validation_indices=val),a.output/'candidates.pt')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(coverage=coverage,validation_novelty_auc=auc)),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
