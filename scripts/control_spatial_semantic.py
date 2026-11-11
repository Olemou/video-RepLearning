#!/usr/bin/env python3
"""Input removal for E1/E2, with fresh causal encoder passes on distinct ranks.

Uses learned spill-conditioned attention to select equal-area content patches.
These are OOD sensitivity controls, not spill masks or localization accuracy.
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
from vision_model.training.spatial_semantic import SpatialSemanticBranch
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from visualize_trajectory_video import decode_selected_frames
from train_spatial_semantic import DEFAULT_TEACHER,content_masks
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--encoder-dir',type=Path,default=Path('/home/olemou/hf'))
    a=p.parse_args()
    torch.set_num_threads(4)
    ctx=initialize_distributed()
    destination=a.run/'input_controls.json'
    if destination.exists():raise FileExistsError(destination)
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    if digest(a.encoder_dir/'model.safetensors')!=raw['checkpoint_sha256']:raise ValueError('Encoder hash mismatch')
    report=json.loads((a.run/'report.json').read_text())
    if report['settings']['layer']!=23:raise ValueError('Input controls currently require final-layer readout')
    ids=raw['ids']
    names=['clean_videoFrame121','clean_videoFrame124','cold_folderImage10','cold_folderImage13']
    selected=[ids.index(name) for name in names]
    positions=list(range(ctx.rank,len(selected),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(selected))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    content=content_masks(raw['manifest'],ids,(16,16))
    cases={}
    for case in report['cases']:
        name=case['case']
        case_dir=Path(case.get('artifact_directory',a.run/name))
        ck=torch.load(case_dir/'checkpoint.pt',map_location='cpu',weights_only=False)
        model=SpatialSemanticBranch(ck['class_prototypes'],**ck['metadata']['architecture']).to(ctx.device).eval()
        model.load_state_dict(ck['model'],strict=True)
        saved=torch.load(case_dir/'representations_and_maps.pt',map_location='cpu',weights_only=False)
        rows=saved['rows']
        z=torch.stack([r['embedding'][0] for r in rows])
        train=torch.tensor([ids.index(v) for v in case['train_source_ids']])
        prototypes=torch.stack([F.normalize(z[train][raw['labels'][train]==label].mean(0),dim=0) for label in (0,1)]).to(ctx.device)
        cases[name]=(model,prototypes,rows)
    loaded=load_pretrained_encoder(str(a.encoder_dir),local_files_only=True)
    loaded.freeze_encoder(True)
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval()
    transform=OfficialVJEPA2Transform(str(a.encoder_dir),preserve_field_of_view=True)
    result=[]
    for position in positions:
        i=selected[position]
        r=records[ids[i]]
        source=Path(raw['dataset_root'])/r['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source hash changed')
        sampled,valid=TemporalSampler(TemporalSamplingConfig(**raw['sampling'])).sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps'])
        inds=sampled[:int(valid.sum())//2*2].tolist()
        n=len(inds)//2
        if inds[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Frame index mismatch')
        decoded=decode_selected_frames(source,inds,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in inds])
        original=transform(frames)
        encoded_cache={}
        def encode(pixels,key):
            if key not in encoded_cache:
                features=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for end in range(1,n+1):
                        tok=encoder(pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16))
                        features.append(tok[0,-256:].reshape(16,16,1024).float())
                encoded_cache[key]=torch.stack(features)[None]
            return encoded_cache[key]
        for name,(model,prototypes,rows) in cases.items():
            score=rows[i]['attention'][0,:n,1].sum(0).flatten()
            candidates=content[i].flatten().nonzero().flatten()
            ordered=candidates[score[candidates].argsort()]
            random=candidates[torch.randperm(len(candidates),generator=torch.Generator().manual_seed(239+i))[:26]]
            masks=dict(high=ordered[-26:],low=ordered[:26],random=random)
            conditions=dict(original=(original,('original',)))
            for kind,cells in masks.items():
                mask=torch.zeros(256,dtype=torch.bool);mask[cells]=True
                mask=mask.reshape(16,16).repeat_interleave(16,0).repeat_interleave(16,1)
                conditions[kind]=(original.masked_fill(mask[None,None],0),tuple(sorted(cells.tolist())))
            uniform=frames.float().mean((-2,-1),keepdim=True).round().expand_as(frames).to(torch.uint8)
            conditions['uniform_frames']=(transform(uniform),('uniform',))
            scores={}
            with torch.no_grad():
                cached_score=float(rows[i]['embedding'][0].to(ctx.device)@(prototypes[1]-prototypes[0]))
                for condition,(pixels,key) in conditions.items():
                    feat=encode(pixels,key)
                    z=model(feat,torch.ones(1,n,device=ctx.device,dtype=torch.bool),content[i:i+1].to(ctx.device))['embedding'][0]
                    similarities=z@prototypes.T
                    scores[condition]=dict(margin=float(similarities[1]-similarities[0]),prediction=int(similarities.argmax()))
            replay_error=abs(scores['original']['margin']-cached_score)
            if replay_error>1e-4:raise RuntimeError(f'Fresh encoder replay changes margin by {replay_error:g}')
            result.append(dict(case=name,id=ids[i],label=int(raw['labels'][i]),conditions=scores,
                               cached_margin=cached_score,replay_margin_error=replay_error,
                               cells={k:v.tolist() for k,v in masks.items()},removed_pixels=26*16*16))
        print(json.dumps(dict(rank=ctx.rank,completed_source=ids[i])),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(result,gathered,dst=0)
    else:gathered=[result]
    if ctx.is_main:
        rows=sorted([r for shard in gathered for r in shard],key=lambda r:(r['case'],r['id']))
        summaries={}
        for name in cases:
            subset=[r for r in rows if r['case']==name]
            summaries[name]={}
            for condition in ('original','high','low','random','uniform_frames'):
                summaries[name][condition]=dict(n=len(subset),accuracy=sum(r['conditions'][condition]['prediction']==r['label'] for r in subset)/len(subset),
                    mean_absolute_margin_change=sum(abs(r['conditions'][condition]['margin']-r['conditions']['original']['margin']) for r in subset)/len(subset))
        destination.write_text(json.dumps(dict(source_counts_per_rank=counts,world_size=ctx.world_size,
            selected_sources=names,rows=rows,summaries=summaries,
            interpretation='Equal-area normalized-mean input replacement, every prefix re-encoded; OOD sensitivity only, no masks or localization accuracy'),indent=2)+'\n')
        print(json.dumps(summaries),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
