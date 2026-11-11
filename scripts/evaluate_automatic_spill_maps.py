#!/usr/bin/env python3
"""Two-GPU all-source actual-decision attribution for matched automatic pilots."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
import torch.distributed as dist
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.spill_localization import SpillPatchHead
from vision_model.training.trajectory import TrajectoryModel
from vision_model.interpretability.spill_localization import explain_trajectory_spatial
from train_spill_localization import build_student,DEFAULT_PREFIX,DEFAULT_TEACHER
from train_spatial_semantic import EXPERIMENT
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);ctx=initialize_distributed()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False);cache=torch.load(DEFAULT_PREFIX,map_location='cpu',weights_only=False,mmap=True)
    original=torch.load(EXPERIMENT/'training/trial_01_checkpoint.pt',map_location='cpu',weights_only=False)
    assigned=list(range(ctx.rank,len(raw['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(raw['ids']))
    rows=[];checks=[];selected={'cold_folderImage10','cold_folderImage13','clean_videoFrame121','clean_videoFrame124'}
    for case in ('automatic_control','automatic_joint'):
        checkpoint=a.root/case/'checkpoint.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        rep=torch.load(checkpoint.parent/'representations.pt',map_location='cpu',weights_only=False)
        student=build_student(cache,SpillPatchHead(**saved['metadata']['head_configuration']),train_attention=False).to(ctx.device).eval().requires_grad_(False)
        student.load_state_dict(saved['model'],strict=True)
        trajectory=TrajectoryModel(original['width'],original['hidden'],original['embedding_dim']).to(ctx.device).eval().requires_grad_(False)
        trajectory.load_state_dict(saved['trajectory'],strict=True)
        for i in assigned:
            n=int(raw['valid'][i].sum());prefixes=[h[None].to(ctx.device).float() for h in cache['rows'][i]['prefixes'][0][:n]]
            proto=rep['prototypes'].to(ctx.device)
            result=explain_trajectory_spatial(student,trajectory,prefixes,raw['timestamps'][i,:n],proto)
            error=float((result['embedding'].cpu()-rep['embeddings'][i]).abs().max())
            local_error=float((result['local_probabilities'].cpu()-rep['local_logits'][i,0,:n].sigmoid()).abs().max())
            if max(error,local_error)>1e-4:raise RuntimeError('Prediction/attribution checkpoint mismatch')
            rows.append(dict(case=case,index=i,id=raw['ids'][i],label=int(raw['labels'][i]),margin=result['margin'],
                local_probabilities=result['local_probabilities'].cpu(),maps={k:v.cpu() for k,v in result['maps'].items()},
                embedding_replay_error=error,local_replay_error=local_error,checkpoint_sha256=digest(checkpoint)))
            if raw['ids'][i] in selected:
                # Head-only randomization must not affect the actual episode target.
                old=copy.deepcopy(student.local_head.state_dict());torch.manual_seed(991)
                for parameter in student.local_head.parameters():torch.nn.init.normal_(parameter,std=.03)
                randomized=explain_trajectory_spatial(student,trajectory,prefixes,raw['timestamps'][i,:n],proto)
                student.local_head.load_state_dict(old,strict=True)
                reversed_target=explain_trajectory_spatial(student,trajectory,prefixes,raw['timestamps'][i,:n],proto.flip(0))
                left=result['maps']['positive_gradient'].flatten();right=reversed_target['maps']['positive_gradient'].flatten()
                checks.append(dict(case=case,id=raw['ids'][i],head_only_randomization_margin_error=abs(result['margin']-randomized['margin']),
                    head_only_randomization_gradient_error=float((result['maps']['positive_gradient']-randomized['maps']['positive_gradient']).abs().max()),
                    head_randomization_local_probability_change=float((result['local_probabilities']-randomized['local_probabilities']).abs().mean()),
                    target_reversal_positive_map_cosine=float(F.cosine_similarity(left,right,dim=0))))
        del student,trajectory
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object((rows,checks),gathered,dst=0)
    else:gathered=[(rows,checks)]
    if ctx.is_main:
        rows=sorted([r for part,_ in gathered for r in part],key=lambda r:(r['case'],r['index']))
        if len({(r['case'],r['index']) for r in rows})!=2*len(raw['ids']):raise RuntimeError('Repeated or missing maps')
        checks=[r for _,part in gathered for r in part]
        torch.save(dict(rows=rows,ids=raw['ids'],labels=raw['labels'],valid=raw['valid'],timestamps=raw['timestamps'],
            content=rep['content'],train_indices=rep['train_indices'],validation_indices=rep['validation_indices']),a.output/'maps.pt')
        (a.output/'sanity_checks.json').write_text(json.dumps(dict(checks=checks,world_size=ctx.world_size,rank_source_counts=counts,
            target='Actual joint trajectory spill-minus-clean training-prototype margin',scope='Conditional final-block attention-key sensitivity; not pixel attribution or a spill mask',
            manual_annotations=0,code_sha256=digest(Path(__file__))),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
