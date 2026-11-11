#!/usr/bin/env python3
"""Separate all-source/all-block normalized-attention sensitivity experiment."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from attribute_experiment81 import load_case, margins_for_group, evaluation_groups
from prepare_experiment81 import normalize_bank,digest
from vision_model.training.distributed import initialize_distributed
from vision_model.interpretability.attention_competition import explain_attention_competition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); torch.set_num_threads(4)
    ctx=initialize_distributed()
    if ctx.device.type!='cuda':raise RuntimeError('Allocated GPU required')
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    if not c['valid'].all():raise ValueError('Variable-tail packing not supported by this all-prefix audit')
    source_sha=digest(ROOT/'src/vision_model/interpretability/attention_competition.py')
    if ctx.is_main:
        a.output.mkdir(parents=True,exist_ok=a.resume)
        (a.output/'settings.json').write_text(json.dumps(dict(baseline=str(a.baseline.resolve()),
            formula='dS/dlogit = A * (dS/dA - sum_key A*dS/dA)',method_sha256=source_sha,
            source_count=80,blocks=24,heads='all',query_selection='all',manual_annotations=0,
            trainable_parameters=0,training_updates=0,mil=False,decoder=False,
            target='Fixed training-prototype spill-minus-clean episode margin',
            claim='Normalized-attention sensitivity, not liquid segmentation or a reproduced named attribution method',
            host=os.uname().nodename,gpu=torch.cuda.get_device_name(ctx.device),world_size=ctx.world_size),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier()
    for case in ('frozen','full'):
        encoder,model,reps,proto,settings=load_case(a.baseline,case,ctx.device)
        checkpoint_sha=digest(a.baseline/case/'checkpoint.pt')
        folder=a.output/case
        if ctx.is_main:folder.mkdir(exist_ok=a.resume)
        if ctx.world_size>1:dist.barrier()
        summaries=[]
        groups=evaluation_groups(reps,settings)
        for round_index in range((len(groups)+ctx.world_size-1)//ctx.world_size):
            slot=round_index*ctx.world_size+ctx.rank
            row=None
            if slot<len(groups):
                ids=groups[slot]
                name='group_'+'_'.join(map(str,ids.tolist()))+'.pt'
                target=folder/name
                if target.exists() and a.resume:
                    old=torch.load(target,map_location='cpu',weights_only=False)
                    if old['checkpoint_sha256']!=checkpoint_sha or old['method_sha256']!=source_sha:
                        raise ValueError('Different method/checkpoint in resumed group')
                    row=old['summary']
                else:
                    old=torch.load(a.baseline/'spatial'/case/name,map_location='cpu',weights_only=False)
                    if old['checkpoint_sha256']!=checkpoint_sha:raise ValueError('Baseline map checkpoint mismatch')
                    pixels=torch.stack([normalize_bank(c['pixels'][int(i),:2],c['normalization']) for i in ids])
                    result=explain_attention_competition(encoder,pixels,old['explicit_features'],
                        lambda f:margins_for_group(model,reps,ids,f,proto,ctx.device),ctx.device)
                    difference=float((result['probability_positive']-old['positive_gradients']).abs().max())
                    score_error=float((result['scores']-old['explicit_scores']).abs().max())
                    if difference>1e-8 or score_error>1e-6:
                        raise RuntimeError(f'Original probability-gradient replay changed: {difference}, {score_error}')
                    row=dict(file=name,ids=ids.tolist(),old_gradient_max_error=difference,
                        old_score_max_error=score_error,explicit_replay_max_error=result['explicit_replay_max_error'])
                    result.update(ids=ids,summary=row,method_sha256=source_sha,checkpoint_sha256=checkpoint_sha)
                    row=dict(row,payload=result)
            gathered=[None]*ctx.world_size if ctx.is_main else None
            if ctx.world_size>1:dist.gather_object(row,gathered,dst=0)
            else:gathered=[row]
            if ctx.is_main:
                for entry in gathered:
                    if entry is None:continue
                    payload=entry.pop('payload',None)
                    if payload is not None:torch.save(payload,folder/entry['file'])
                    summaries.append(entry)
                    print(json.dumps(dict(case=case,**entry)),flush=True)
                (folder/'progress.json').write_text(json.dumps(summaries,indent=2)+'\n')
        if ctx.is_main:
            ids=[i for s in summaries for i in s['ids']]
            if sorted(ids)!=list(range(80)):raise RuntimeError('Duplicated or missing evaluation source')
            (folder/'completed.json').write_text(json.dumps(dict(source_count=80,checkpoint_sha256=checkpoint_sha,
                groups=len(summaries),method_sha256=source_sha,training_updates=0,localization_validated=False),indent=2)+'\n')
        del encoder,model
        torch.cuda.empty_cache()
        if ctx.world_size>1:dist.barrier()
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
