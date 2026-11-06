#!/usr/bin/env python3
"""Fixed-weight patch-loss and projector-randomization audits on distinct ranks."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from scipy.stats import spearmanr
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.patch_vico import patch_vico
from vision_model.training.final_attention import prefix_key_map
from train_final_attention import make_student


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    maps=torch.load(a.run/'spatial_evaluation/attention_maps.pt',map_location='cpu',weights_only=False)
    indexed={(r['case'],r['index']):r for r in maps['rows']}
    val=c['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size))
    audit_distributed_indices(positions,len(val))
    probe_rows=[];fixed_rows=[]
    for name in ('FT0_frozen','FT1_attention'):
        ck=torch.load(a.run/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        rep=torch.load(a.run/name/'representations.pt',map_location='cpu',weights_only=False)
        head={k.removeprefix('projector.'):v for k,v in ck['model'].items() if k.startswith('projector.')}
        model=make_student(c,False,head).to(ctx.device).eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
        random=copy.deepcopy(model)
        torch.manual_seed(239)
        for module in random.projector.modules():
            if hasattr(module,'reset_parameters'):module.reset_parameters()
        prototypes=rep['prototypes'].to(ctx.device);direction=prototypes[1]-prototypes[0]
        for position in positions:
            i=val[position];n=int(c['valid'][i].sum());support=rep['content'][i].to(ctx.device)
            with torch.no_grad():
                q=torch.stack([random(c['rows'][i]['prefixes'][0][p][None].to(ctx.device),frames=p+1)[0] for p in range(n)])
            leaf=q.detach().requires_grad_(True)
            score=F.normalize(leaf[:,support].mean((0,1)),dim=-1)@direction
            upstream=torch.autograd.grad(score,leaf)[0];saliency=torch.zeros(n,256,device=ctx.device);raw_error=0.
            for end in range(1,n+1):
                hidden=c['rows'][i]['prefixes'][0][end-1][None].to(ctx.device).float().requires_grad_(True)
                q,attn,_=random(hidden,frames=end,capture=True)
                gradient=torch.autograd.grad(q,attn,upstream[end-1][None])[0]
                saliency[:end]+=prefix_key_map(attn,gradient,kind='positive_gradient').reshape(end,256).detach()
                raw_error=max(raw_error,float((prefix_key_map(attn).reshape(end,256).detach().cpu()-indexed[(name,i)]['prefix_maps']['raw'][end-1]).abs().max()))
            x=saliency[:,support].flatten().cpu();z=indexed[(name,i)]['maps']['positive_gradient'][:,support.cpu()].flatten()
            probe_rows.append(dict(case=name,id=c['ids'][i],label=int(c['labels'][i]),raw_attention_random_projector_max_error=raw_error,
                positive_gradient_random_projector_cosine=float(F.cosine_similarity(x,z,dim=0)),
                positive_gradient_random_projector_spearman=float(spearmanr(x.numpy(),z.numpy()).statistic)))
        # Fixed final curriculum; six deterministic one-prefix-per-source
        # schedules, covering every prefix once with identical candidate count.
        for split,indices in (('train',c['train_indices']),('validation',c['validation_indices'])):
            # Both ranks receive distinct schedules, avoiding duplicated losses.
            for offset in range(ctx.rank,6,ctx.world_size):
                steps=(indices+239+offset)%c['valid'][indices].sum(-1)
                q=rep['patch_features'][indices,:,steps].to(ctx.device)
                support=rep['content'][indices].to(ctx.device)[:,None].expand(-1,2,-1)
                source=indices.to(ctx.device)
                labels=c['labels'][indices].to(ctx.device)[:,None,None].expand_as(support)[support]
                images=(source*6+steps.to(ctx.device))[:,None,None].expand_as(support)[support]
                sources=source[:,None,None].expand_as(support)[support]
                with torch.no_grad():loss,details=patch_vico(q[support],labels,images,sources,epoch=20,distributed=False)
                fixed_rows.append(dict(case=name,split=split,schedule_offset=offset,loss=float(loss),**{k:float(v) for k,v in details.items()}))
    probes=[None]*ctx.world_size if ctx.is_main else None;fixed=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(probe_rows,probes,dst=0);dist.gather_object(fixed_rows,fixed,dst=0)
    else:probes=[probe_rows];fixed=[fixed_rows]
    if ctx.is_main:
        destination=a.run/'spatial_evaluation/additional_probes.json'
        if destination.exists():raise FileExistsError(destination)
        destination.write_text(json.dumps(dict(randomization=[r for part in probes for r in part],fixed_patch_objective=[r for part in fixed for r in part],
            randomization_protocol='Reset only projector parameters with seed 239; keep backbone and trained prototype direction fixed; all 16 selection sources',
            fixed_patch_protocol='Six balanced causal-prefix schedules, both actual views, all 192 content patches, fixed final curriculum; candidate counts differ between train and validation'),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
