#!/usr/bin/env python3
"""Annotation-free, source-excluded normal patch diagnostics on the existing split.

Twelve fixed cases (4 blocks x 3 normal-feature geometries). No masks, decoder,
MIL, backbone update, or selection using a validation localization target.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score,average_precision_score
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.normal_patch_reference import NormalPatchReference,automatic_candidates
from vision_model.training.trajectory import TrajectoryModel
from train_spatial_semantic import read_inputs,content_masks,DEFAULT_SPATIAL,DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import digest,recognition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spatial-cache',type=Path,default=DEFAULT_SPATIAL)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--temporal-checkpoint',type=Path,default=EXPERIMENT/'training/trial_01_checkpoint.pt')
    p.add_argument('--references-per-source',type=int,default=512)
    p.add_argument('--modes',nargs='+',choices=('raw','residual','whitened_residual','selected_residual'),default=['raw','residual','whitened_residual'])
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(239)
    torch.set_float32_matmul_precision('high')
    ctx=initialize_distributed()
    if ctx.device.type!='cuda':raise RuntimeError('Real GPU benchmark required')
    c,raw,ck,train,val=read_inputs(a)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(c['ids']))
    support=content_masks(raw['manifest'],c['ids'],(16,16)).flatten(1)
    normal=train[c['labels'][train]==0]
    clean_ids=[c['ids'][i] for i in normal]
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    for i in assigned:
        if digest(Path(raw['dataset_root'])/records[c['ids'][i]]['video_path'])!=raw['video_sha256'][i]:raise ValueError('Source bytes changed')
    outcomes={};allmaps={};pseudo={}
    begin=time.monotonic()
    for layer in (5,11,17,23):
        slot=c['layers'].index(layer)
        normal_tokens=c['tokens'][normal,:,:,slot].to(ctx.device).float()
        for mode in a.modes:
            name=f'block{layer}_{mode}'
            ref=NormalPatchReference(mode=mode,per_source=a.references_per_source).fit(
                normal_tokens,c['valid'][normal].to(ctx.device),support[normal].to(ctx.device),clean_ids,c['labels'][normal])
            rows=[]
            for i in assigned:
                x=c['tokens'][i,:,:,slot].to(ctx.device).float()
                score=ref.score(x,support[i].to(ctx.device),exclude_source=c['ids'][i])
                score[:,~c['valid'][i]]=0
                rows.append((i,score.cpu()))
            gathered=[None]*ctx.world_size if ctx.is_main else None
            if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
            else:gathered=[rows]
            if ctx.is_main:
                rows=sorted([row for shard in gathered for row in shard],key=lambda r:r[0])
                if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Evaluation omitted/duplicated a source')
                maps=torch.stack([r[1] for r in rows]);allmaps[name]=maps
                clean_values=torch.cat([maps[i,0,c['valid'][i]][:,support[i]].flatten() for i in normal])
                q={f'q{v:g}':float(torch.quantile(clean_values,v)) for v in (.5,.95,.99,.999)}
                low,high=q['q0.5'],q['q0.99']
                if high>low:
                    targets=automatic_candidates(maps,c['valid'],support,low=low,high=high)
                else:targets=torch.full_like(maps,-1,dtype=torch.long)
                # Trusted video-level absence gives clean targets. No cold-video
                # label is expanded into patch foreground targets.
                for i in torch.where(c['labels']==0)[0]:
                    for t in torch.where(c['valid'][i])[0]:targets[i,:,t,support[i]]=0
                pseudo[name]=targets
                source=[]
                for i in range(len(maps)):
                    x=maps[i,0,c['valid'][i]][:,support[i]]
                    excess=(x-q['q0.95']).clamp_min(0)
                    mass=excess.sum(-1);prob=excess/mass[:,None].clamp_min(1e-20)
                    entropy=-(prob*prob.clamp_min(1e-20).log()).sum(-1)/torch.log(x.new_tensor(x.shape[-1]))
                    mirrored=maps[i,1,c['valid'][i]].reshape(-1,16,16).flip(-1).flatten(1)[:,support[i]]
                    source.append(dict(index=i,id=c['ids'][i],label=int(c['labels'][i]),
                        split='training_fit' if i in train else 'source_validation_selection',
                        mean_novelty=float(x.mean()),mean_excess=float(excess.mean()),
                        above_clean_q95_fraction=float((x>q['q0.95']).float().mean()),
                        entropy_when_nonempty=float(entropy[mass>0].mean()) if (mass>0).any() else None,
                        empty_map_frames=int((mass==0).sum()),
                        mirror_normalized_l1=float((x-mirrored).abs().mean()/(x.abs().mean()+1e-12)),
                        foreground_candidates=int((targets[i,0]==1).sum()),background_candidates=int((targets[i,0]==0).sum())))
                split_metrics={}
                for split,idx in (('train',train),('validation',val)):
                    selected=[source[i] for i in idx];truth=c['labels'][idx].numpy()
                    values=[r['mean_excess'] for r in selected]
                    split_metrics[split]=dict(sources=len(idx),mean_excess_roc_auc=float(roc_auc_score(truth,values)),
                        mean_excess_average_precision=float(average_precision_score(truth,values)),
                        clean_patch_false_alarm_rate=float(torch.cat([maps[i,0,c['valid'][i]][:,support[i]].flatten()>q['q0.95'] for i in idx if c['labels'][i]==0]).float().mean()),
                        foreground_candidate_sources=sum(r['foreground_candidates']>0 for r in selected),
                        foreground_candidate_cells=sum(r['foreground_candidates'] for r in selected),
                        background_candidate_cells=sum(r['background_candidates'] for r in selected),
                        scope='Novelty recognition and unverified candidate statistics; not spill localization accuracy')
                outcomes[name]=dict(layer=layer,mode=mode,normal_sources=clean_ids,bank_size=len(ref.bank),
                    selected_feature_indices=ref.feature_indices.cpu().tolist() if mode=='selected_residual' else None,
                    calibration=q,metrics=split_metrics,sources=source)
                print(json.dumps(dict(case=name,validation=split_metrics['validation'],elapsed=time.monotonic()-begin)),flush=True)
            del ref
            if ctx.world_size>1:dist.barrier()
        del normal_tokens
    if ctx.is_main:
        model=TrajectoryModel(ck['width'],ck['hidden'],ck['embedding_dim']).to(ctx.device).eval()
        model.load_state_dict(ck['model'],strict=True)
        with torch.no_grad():z=model(raw['features'][:,0].to(ctx.device),raw['valid'].to(ctx.device),raw['timestamps'].to(ctx.device))['embedding'].cpu()
        baseline={split:recognition(z,c['labels'],train,idx,exclude_self=split=='train') for split,idx in (('train',train),('validation',val))}
        torch.save(dict(ids=c['ids'],labels=c['labels'],valid=c['valid'],timestamps=c['timestamps'],content=support,
            maps=allmaps,pseudo_targets=pseudo,train_indices=train,validation_indices=val,trajectory_embeddings=z),a.output/'maps.pt')
        metadata=dict(world_size=ctx.world_size,rank_source_counts=counts,device=torch.cuda.get_device_name(ctx.device),
            real_sources=len(c['ids']),fit_clean_sources=len(normal),fit_spill_sources=0,seed=239,
            reference_sampling=f'{a.references_per_source} deterministic valid patch instances per training clean source, both actual views',
            calibration='Training clean identity cells with their own source excluded from the reference bank; whitening statistics still fit all training clean sources',
            temporal_context_policy=c['temporal_context_policy'],spatial_cache_sha256=digest(a.spatial_cache),
            teacher_cache_sha256=digest(a.teacher_cache),trajectory_checkpoint_sha256=digest(a.temporal_checkpoint),
            source_sha256=raw['video_sha256'],source_code_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                'src/vision_model/training/normal_patch_reference.py':digest(ROOT/'src/vision_model/training/normal_patch_reference.py')},
            manual_annotations_used=0,decoder=False,mil=False,backbone_optimizer_updates=0,trajectory_optimizer_updates=0,
            spatial_scope='Normal-reference novelty, not V-JEPA attention or original-decision attribution',
            independent_scene_test=False,elapsed_seconds=time.monotonic()-begin)
        (a.output/'results.json').write_text(json.dumps(dict(metadata=metadata,baseline=baseline,cases=outcomes),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
