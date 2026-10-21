#!/usr/bin/env python3
"""Label-blind input perturbations, recomputing every causal prefix on both GPUs.

Uniform frames preserve per-frame channel means while removing image structure.
Moment standardization is an out-of-distribution sensitivity test, not a new
normalization recommendation or a localization ground truth.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from vision_model.training.trajectory import TrajectoryModel
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',weights_only=False)
    ck=torch.load(a.diagnosis.parent/'training/trial_01_checkpoint.pt',map_location='cpu',weights_only=False)
    ref=torch.load(a.diagnosis/'audit_representations.pt',map_location='cpu',weights_only=False)
    model=TrajectoryModel(ck['width'],ck['hidden'],ck['embedding_dim']).to(ctx.device).eval()
    model.load_state_dict(ck['model'],strict=True)
    proto=ref['prototypes'].to(ctx.device)
    loaded=load_pretrained_encoder('/home/olemou/hf',local_files_only=True)
    loaded.freeze_encoder(True)
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval()
    transform=OfficialVJEPA2Transform('/home/olemou/hf',preserve_field_of_view=True)
    records={r['video_id']:r for r in (json.loads(l) for l in Path(c['manifest']).read_text().splitlines())}
    validation=ref['validation_indices'].tolist()
    positions=list(range(ctx.rank,len(validation),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(validation))
    rows=[]
    for pos in positions:
        i=validation[pos]
        r=records[c['ids'][i]]
        sampled,valid=TemporalSampler(TemporalSamplingConfig(**c['sampling'])).sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps'])
        inds=sampled[:int(valid.sum())//2*2].tolist()
        decoded=decode_selected_frames(Path(c['dataset_root'])/r['video_path'],inds,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[k]).permute(2,0,1) for k in inds])
        n=len(frames)//2
        original_pixels=transform(frames)
        uniform=frames.float().mean((-2,-1),keepdim=True).round().expand_as(frames).to(torch.uint8)
        value=frames.float()/255
        mean=value.mean((-2,-1),keepdim=True)
        std=value.std((-2,-1),keepdim=True).clamp_min(1e-5)
        standardized=((value-mean)/std*.15+.5).clamp(0,1).mul(255).round().to(torch.uint8)
        conditions={'original':original_pixels,'uniform_frames':transform(uniform),'standardized_moments':transform(standardized)}
        path=a.diagnosis/'spatial_matched_precision'/f'{r["video_id"]}_layer_maps.pt'
        if path.exists():
            maps=torch.load(path,map_location='cpu',weights_only=False)['attention_gradient'][-1].sum(0)
            scale=min(256/r['height'],256/r['width'])
            h,w=round(r['height']*scale),round(r['width']*scale)
            top,left=(256-h)//2,(256-w)//2
            content=torch.zeros(256,256)
            content[top:top+h,left:left+w]=1
            full_content=F.avg_pool2d(content[None,None],16,16)[0,0].flatten()==1
            candidates=full_content.nonzero().flatten()
            rank=candidates[maps.flatten()[candidates].argsort()]
            choices={'low_content26':rank[:26],'high_content26':rank[-26:],
                     'random_content26':candidates[torch.randperm(len(candidates),generator=torch.Generator().manual_seed(239))[:26]]}
            for name,cells in choices.items():
                mask=torch.zeros(256,dtype=torch.bool)
                mask[cells]=True
                mask=mask.reshape(16,16).repeat_interleave(16,0).repeat_interleave(16,1)
                conditions[name]=original_pixels.masked_fill(mask[None,None],0)
        scores={}
        for name,pixels in conditions.items():
            features=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for end in range(1,n+1):
                    tok=encoder(pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16))
                    features.append(tok[0,-256:].mean(0).float())
            with torch.no_grad():
                z=model(torch.stack(features)[None],torch.ones(1,n,dtype=torch.bool,device=ctx.device),c['timestamps'][i:i+1,:n].to(ctx.device))['embedding'][0]
                sims=z@proto.T
            scores[name]=dict(margin=float(sims[1]-sims[0]),prediction=int(sims.argmax()))
        rows.append(dict(id=r['video_id'],label=int(c['labels'][i]),category=r['category'],conditions=scores))
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:
        dist.gather_object(rows,gathered,dst=0)
    else:
        gathered=[rows]
    if ctx.is_main:
        rows=sorted([r for shard in gathered for r in shard],key=lambda r:r['id'])
        result=dict(source_counts_per_rank=counts,rows=rows,world_size=ctx.world_size,
                    removed_content_area_per_condition=26*16*16,selection_uses_masks=False,
                    interpretation='OOD input sensitivity controls; no spill masks or localization metric')
        for name in sorted({k for r in rows for k in r['conditions']}):
            selected=[r for r in rows if name in r['conditions']]
            result[name]=dict(n=len(selected),accuracy=sum(r['label']==r['conditions'][name]['prediction'] for r in selected)/len(selected),
                 mean_margin_change=sum(r['conditions'][name]['margin']-r['conditions']['original']['margin'] for r in selected)/len(selected))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if ctx.world_size>1:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
