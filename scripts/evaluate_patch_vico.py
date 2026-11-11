#!/usr/bin/env python3
"""Fixed-final-curriculum direct patch losses at every valid cached prefix."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.distributed as dist
from vision_model.training.patch_vico import patch_vico
from vision_model.training.distributed import initialize_distributed
from control_patch_vico import case_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--additional-run',type=Path,nargs='*',default=[])
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    destination=a.run/'patch_evaluation.json'
    if destination.exists():raise FileExistsError(destination)
    result={}
    for run in [a.run,*a.additional_run]:
        report=json.loads((run/'report.json').read_text())
        for case in report['cases']:
            name=case_name(case)
            saved=torch.load(run/case['case']/'representations.pt',map_location='cpu',weights_only=False,mmap=True)
            result[name]={}
            for split,key in (('train','train_source_ids'),('validation','validation_source_ids')):
                indices=[saved['ids'].index(i) for i in case[key]]
                owned=torch.tensor(indices[ctx.rank::ctx.world_size])
                per_time=[]
                for time in range(saved['valid'].shape[1]):
                    source=owned[saved['valid'][owned,time]]
                    if len(source)==0:raise ValueError('Each rank requires sources at an evaluated time')
                    mask=saved['content'][source,None].expand(-1,2,-1)
                    q=saved['q'][source,:,time][mask].to(ctx.device)
                    labels=saved['labels'][source,None,None].expand_as(mask)[mask].to(ctx.device)
                    image=(source*saved['valid'].shape[1]+time)[:,None,None].expand_as(mask)[mask].to(ctx.device)
                    sources=source[:,None,None].expand_as(mask)[mask].to(ctx.device)
                    with torch.no_grad():
                        loss,statistics=patch_vico(q,labels,image,sources,epoch=20,positive_weighting=case['case']!='uniform_positive')
                    if ctx.world_size>1:dist.all_reduce(loss);loss/=ctx.world_size
                    stats={}
                    for name_stat,value in statistics.items():
                        if ctx.world_size>1:
                            op=dist.ReduceOp.SUM if name_stat=='owned_anchors' else (dist.ReduceOp.MIN if name_stat in ('minimum_positive_cosine','minimum_pair_separation','positive_weight_min') else dist.ReduceOp.MAX)
                            dist.all_reduce(value,op=op)
                        stats[name_stat]=float(value)
                    per_time.append(dict(prefix=time,patch_loss=float(loss),**stats))
                result[name][split]=dict(mean_prefix_patch_loss=sum(r['patch_loss'] for r in per_time)/len(per_time),per_time=per_time,
                    comparison='All content patches, both views, every valid prefix evaluated in separate global source batches; fixed final curriculum')
            if ctx.is_main:print(json.dumps(dict(case=name,train=result[name]['train']['mean_prefix_patch_loss'],validation=result[name]['validation']['mean_prefix_patch_loss'])),flush=True)
    if ctx.is_main:destination.write_text(json.dumps(result,indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
