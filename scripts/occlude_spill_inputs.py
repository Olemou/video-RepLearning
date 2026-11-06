#!/usr/bin/env python3
"""Exhaustive, mask-free input-region score changes on all validation videos.

Each 32x32 checkpoint-grid cell is replaced by that frame/channel's observed
content mean throughout the sampled clip. Every causal prefix is re-encoded.
The result explains an episode decision over fixed image coordinates, not a
per-frame spill mask. No validation label selects a cell or signs the map.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from train_final_attention import make_student
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    cases={}
    for name,folder in [('FT0',a.experiment/'matched_100/FT0_frozen'),('FT1_3em6',a.experiment/'matched_100/FT1_attention'),('FT1_1em4',a.experiment/'attention_lr1em4_100/FT1_attention')]:
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        head={k.removeprefix('projector.'):v for k,v in ck['model'].items() if k.startswith('projector.')}
        model=make_student(c,False,head).to(ctx.device).eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
        saved=torch.load(folder/'representations.pt',map_location='cpu',weights_only=False)
        cases[name]=(model,saved,saved['prototypes'].to(ctx.device))
    loaded=load_pretrained_encoder('/home/olemou/hf',local_files_only=True)
    if digest(Path('/home/olemou/hf/model.safetensors'))!=c['encoder_sha256']:raise ValueError('Encoder changed')
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    transform=OfficialVJEPA2Transform('/home/olemou/hf',preserve_field_of_view=True)
    val=c['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(val));rows=[]
    cells=[(r,col) for r in range(1,7) for col in range(8)]
    for position in positions:
        i=val[position];record=records[c['ids'][i]];source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        if record['width']*3!=record['height']*4:raise ValueError('Control assumes verified 4:3 full-frame content')
        indices=c['rows'][i]['sampled_frame_indices'];n=len(indices)//2
        decoded=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        original=transform(frames).to(ctx.device);fill=original[:,:,32:224].mean((-2,-1),keepdim=True)
        output={name:dict(delta_margin=torch.zeros(8,8),original_margin=[],perturbed_margin=[],prediction=[]) for name in cases}
        for start in range(0,len(cells),3):
            # An unmodified clip accompanies each three counterfactuals with
            # the SAME batch shape. Repeated baselines are calibration only;
            # each source/counterfactual contributes once to reported metrics.
            selected=cells[start:start+3]
            batch=original[None].expand(4,-1,-1,-1,-1).clone()
            for slot,(r,col) in enumerate(selected,1):batch[slot,:,:,32*r:32*(r+1),32*col:32*(col+1)]=fill
            means={name:torch.zeros(4,256,device=ctx.device) for name in cases}
            for end in range(1,n+1):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=encoder.patch_embed(batch[:,:2*end].permute(0,2,1,3,4).to(dtype=torch.bfloat16))
                    for block in encoder.blocks[:-1]:hidden=block(hidden,frames=end,height=16,width=16)
                for name,(model,saved,prototypes) in cases.items():
                    q=model(hidden,frames=end)[:,saved['content'][i].to(ctx.device)]
                    means[name]+=q.mean(1)/n
            for name,(_,saved,prototypes) in cases.items():
                scores=F.normalize(means[name],dim=-1)@prototypes.T
                margins=scores[:,1]-scores[:,0]
                output[name]['original_margin'].append(float(margins[0]))
                for slot,(r,col) in enumerate(selected,1):
                    output[name]['delta_margin'][r,col]=margins[0]-margins[slot]
                    output[name]['perturbed_margin'].append(float(margins[slot]));output[name]['prediction'].append(int(scores[slot].argmax()))
        for name,(_,saved,prototypes) in cases.items():
            result=output[name]
            cached=float(saved['embeddings'][i,0]@(prototypes[1]-prototypes[0]).cpu())
            result['batch4_original_to_cached_margin_max_error']=max(abs(v-cached) for v in result['original_margin'])
            result['repeated_original_margin_range']=max(result['original_margin'])-min(result['original_margin'])
            if result['repeated_original_margin_range']>1e-5:raise RuntimeError('Unmodified score depends on other batch members')
            result['delta_margin']=result['delta_margin'].tolist()
        rows.append(dict(index=i,id=c['ids'][i],label=int(c['labels'][i]),cases=output))
        print(json.dumps(dict(rank=ctx.rank,completed_source=c['ids'][i])),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
    else:gathered=[rows]
    if ctx.is_main:
        allrows=sorted([r for part in gathered for r in part],key=lambda r:r['index'])
        if len({r['index'] for r in allrows})!=len(val):raise RuntimeError('Source assignment failed')
        destination=a.experiment/'dense_input_occlusion.json'
        if destination.exists():raise FileExistsError(destination)
        destination.write_text(json.dumps(dict(rows=allrows,world_size=ctx.world_size,rank_source_counts=counts,
            protocol='Exhaustive 48 disjoint 32x32 content cells; replace each with its frame/channel observed-content mean, all sampled frames; re-encode every causal prefix',
            target='Episode spill-minus-clean prototype margin; positive delta means removal decreases the spill score',
            temporal_scope='One spatial-coordinate score per whole sampled clip, NOT frame-specific attribution',
            limitation='Finite OOD interventions, contextual decisions, no masks or independent scenes; no spill-localization accuracy follows from a positive score change'),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
