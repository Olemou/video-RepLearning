#!/usr/bin/env python3
"""All-validation input re-encoding controls for direct patch classifiers.

All maps target the spill-minus-clean prototype margin. High/low selections
use absolute signed contribution, never validation labels. Three seeded random
controls have equal area. These tests provide no spill-mask ground truth.
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
from vision_model.training.patch_vico import PatchProjector
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from visualize_trajectory_video import decode_selected_frames
from train_spatial_semantic import DEFAULT_TEACHER
from diagnose_spill_representation import digest


def case_name(case):
    weight=case.get('local_keep_weight',0.)
    return case['case'] if not weight else f"{case['case']}_local_keep_{weight:g}"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--additional-run',type=Path,nargs='*',default=[])
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--encoder-dir',type=Path,default=Path('/home/olemou/hf'))
    a=p.parse_args()
    torch.set_num_threads(4)
    ctx=initialize_distributed()
    destination=a.run/'input_controls.json'
    if destination.exists():raise FileExistsError(destination)
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    if digest(a.encoder_dir/'model.safetensors')!=raw['checkpoint_sha256']:raise ValueError('Encoder hash mismatch')
    cases={}
    validation_ids=None
    for run in [a.run,*a.additional_run]:
        report=json.loads((run/'report.json').read_text())
        if report['settings']['layer']!=23:raise ValueError('Fresh controls currently require final-layer models')
        for case in report['cases']:
            if validation_ids is None:validation_ids=case['validation_source_ids']
            elif validation_ids!=case['validation_source_ids']:raise ValueError('Evaluation split differs')
            name=case_name(case)
            if name in cases:raise ValueError('Duplicate case identity')
            folder=run/case['case']
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            model=PatchProjector(**ck['metadata']['architecture']).to(ctx.device).eval().requires_grad_(False)
            model.load_state_dict(ck['model'],strict=True)
            saved=torch.load(folder/'representations.pt',map_location='cpu',weights_only=False)
            if saved['ids']!=raw['ids']:raise ValueError('Saved source ordering differs')
            cases[name]=(model,saved['prototypes'].to(ctx.device),saved['signed_maps'],saved['embeddings'],saved['content'])
    selected=[raw['ids'].index(name) for name in validation_ids]
    positions=list(range(ctx.rank,len(selected),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(selected))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    loaded=load_pretrained_encoder(str(a.encoder_dir),local_files_only=True)
    loaded.freeze_encoder(True)
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval()
    transform=OfficialVJEPA2Transform(str(a.encoder_dir),preserve_field_of_view=True)
    result=[]
    for position in positions:
        i=selected[position];record=records[raw['ids'][i]]
        source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Video hash mismatch')
        sampled,valid=TemporalSampler(TemporalSamplingConfig(**raw['sampling'])).sample_with_validity(num_frames=record['num_frames'],source_fps=record['fps'])
        indices=sampled[:int(valid.sum())//2*2].tolist();n=len(indices)//2
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Frame alignment mismatch')
        decoded=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        original=transform(frames)
        encoded={}
        def encode(pixels,key):
            if key not in encoded:
                rows=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for end in range(1,n+1):
                        tokens=encoder(pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16))
                        rows.append(tokens[0,-256:].float())
                encoded[key]=torch.stack(rows)
            return encoded[key]
        for name,(model,prototypes,maps,embeddings,contents) in cases.items():
            support=contents[i]
            candidates=support.nonzero().flatten()
            importance=maps[i,0,:n].abs().mean(0)
            order=candidates[importance[candidates].argsort(stable=True)]
            chosen=dict(high=order[-26:],low=order[:26])
            for seed in range(3):
                chosen[f'random_{seed}']=candidates[torch.randperm(len(candidates),generator=torch.Generator().manual_seed(239+i+1000*seed))[:26]]
            conditions=dict(original=(original,('original',)))
            for kind,cells in chosen.items():
                mask=torch.zeros(256,dtype=torch.bool);mask[cells]=True
                mask=mask.reshape(16,16).repeat_interleave(16,0).repeat_interleave(16,1)
                conditions[kind]=(original.masked_fill(mask[None,None],0),tuple(sorted(cells.tolist())))
            uniform=frames.float().mean((-2,-1),keepdim=True).round().expand_as(frames).to(torch.uint8)
            conditions['uniform_frames']=(transform(uniform),('uniform_frames',))
            scores={}
            with torch.no_grad():
                cached_margin=float(embeddings[i,0].to(ctx.device)@(prototypes[1]-prototypes[0]))
                for condition,(pixels,key) in conditions.items():
                    q=model(encode(pixels,key))[:,support.to(ctx.device)]
                    z=F.normalize(q.mean((0,1)),dim=-1)
                    similarities=z@prototypes.T
                    scores[condition]=dict(margin=float(similarities[1]-similarities[0]),prediction=int(similarities.argmax()))
            error=abs(scores['original']['margin']-cached_margin)
            if error>1e-4:raise RuntimeError(f'Fresh score does not replay cache: {error:g}')
            result.append(dict(case=name,id=raw['ids'][i],label=int(raw['labels'][i]),conditions=scores,
                replay_margin_error=error,selected_cells={k:v.tolist() for k,v in chosen.items()},removed_pixels=26*16*16))
        print(json.dumps(dict(rank=ctx.rank,completed_source=raw['ids'][i])),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(result,gathered,dst=0)
    else:gathered=[result]
    if ctx.is_main:
        rows=sorted([r for shard in gathered for r in shard],key=lambda r:(r['case'],r['id']))
        summaries={}
        for name in cases:
            subset=[r for r in rows if r['case']==name]
            summaries[name]={}
            for condition in next(iter(subset))['conditions']:
                summaries[name][condition]=dict(n=len(subset),accuracy=sum(r['conditions'][condition]['prediction']==r['label'] for r in subset)/len(subset),
                    mean_absolute_margin_change=sum(abs(r['conditions'][condition]['margin']-r['conditions']['original']['margin']) for r in subset)/len(subset),
                    mean_true_class_margin_drop=sum((2*r['label']-1)*(r['conditions']['original']['margin']-r['conditions'][condition]['margin']) for r in subset)/len(subset))
        destination.write_text(json.dumps(dict(world_size=ctx.world_size,source_counts_per_rank=counts,selected_sources=validation_ids,
            rows=rows,summaries=summaries,selection='All 16 validation sources, no fitted intervention selection; three seeded random controls; all causal prefixes re-encoded',
            limitations='Mean-pixel replacement is OOD; no spill masks; scores test input sensitivity, not spill-region correctness'),indent=2)+'\n')
        print(json.dumps(summaries),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
