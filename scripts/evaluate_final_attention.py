#!/usr/bin/env python3
"""Actual final-attention gradients and all-validation input-removal controls.

Maps preserve key time; past keys are never painted on a current-frame image.
This is a final-block diagnostic, not full multilayer LeGrad or segmentation.
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
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.final_attention import prefix_key_map
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from train_final_attention import make_student
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--encoder-dir',type=Path,default=Path('/home/olemou/hf'))
    p.add_argument('--reuse-frozen-evaluation',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    destination=a.run/'spatial_evaluation'
    if ctx.is_main:destination.mkdir(exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size))
    _,counts=audit_distributed_indices(assigned,len(c['ids']))
    cases={}
    for name in ('FT0_frozen','FT1_attention'):
        ck=torch.load(a.run/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        if ck['metadata']['cache_sha256']!=digest(a.cache):raise ValueError('Different cache')
        head={k.removeprefix('projector.'):v for k,v in ck['model'].items() if k.startswith('projector.')}
        model=make_student(c,False,head).to(ctx.device).eval().requires_grad_(False)
        model.load_state_dict(ck['model'],strict=True)
        saved=torch.load(a.run/name/'representations.pt',map_location='cpu',weights_only=False)
        if saved['ids']!=c['ids']:raise ValueError('Source order differs')
        cases[name]=(model,saved,saved['prototypes'].to(ctx.device))
    rows=[]
    reuse_maps={};reuse_controls={}
    if a.reuse_frozen_evaluation:
        if (a.run/'FT0_frozen').resolve()!=(a.reuse_frozen_evaluation.parent/'FT0_frozen').resolve():raise ValueError('Frozen checkpoints differ')
        prior=torch.load(a.reuse_frozen_evaluation/'attention_maps.pt',map_location='cpu',weights_only=False)
        reuse_maps={r['index']:r for r in prior['rows'] if r['case']=='FT0_frozen'}
        reuse_controls={r['index']:r for r in json.loads((a.reuse_frozen_evaluation/'input_controls.json').read_text())['rows'] if r['case']=='FT0_frozen'}
    for i in assigned:
        n=int(c['valid'][i].sum());support=cases['FT0_frozen'][1]['content'][i].to(ctx.device)
        for name,(model,saved,prototypes) in cases.items():
            if name=='FT0_frozen' and reuse_maps:
                rows.append(reuse_maps[i]);continue
            leaf=saved['patch_features'][i,0,:n].to(ctx.device).detach().requires_grad_(True)
            mean=leaf[:,support].mean((0,1));direction=prototypes[1]-prototypes[0]
            margin=F.normalize(mean,dim=-1)@direction
            upstream=torch.autograd.grad(margin,leaf)[0]
            maps={kind:torch.zeros(n,256,device=ctx.device) for kind in ('raw','positive_gradient','signed_attention_gradient')}
            prefix_maps={kind:[] for kind in maps};maxerror=0.;gradnorm=[]
            opposite=[]
            for end in range(1,n+1):
                hidden=c['rows'][i]['prefixes'][0][end-1][None].to(ctx.device).float().requires_grad_(True)
                q,attention,_=model(hidden,frames=end,capture=True)
                maxerror=max(maxerror,float((q[0].detach()-leaf[end-1].detach()).abs().max()))
                gradient=torch.autograd.grad(q,attention,grad_outputs=upstream[end-1][None])[0]
                gradnorm.append(float(gradient.norm()))
                opposite.append(prefix_key_map(attention,-gradient,kind='positive_gradient').reshape(end,256).detach().cpu())
                for kind in maps:
                    values=prefix_key_map(attention,gradient,kind=kind).reshape(end,256).detach()
                    maps[kind][:end]+=values
                    prefix_maps[kind].append(values.cpu())
            if maxerror>2e-5 or not max(gradnorm)>0:raise RuntimeError('Attribution replay/gradient failed')
            positive=maps['positive_gradient'][:,support]
            negative=torch.zeros_like(maps['positive_gradient'])
            for end,v in enumerate(opposite,1):negative[:end]+=v.to(ctx.device)
            target_cos=float(F.cosine_similarity(positive.flatten(),negative[:,support].flatten(),dim=0))
            rows.append(dict(index=i,case=name,maps={k:v.cpu() for k,v in maps.items()},prefix_maps=prefix_maps,
                clean_target_prefix_positive_gradients=opposite,margin=float(margin.detach()),gradient_norms=gradnorm,
                checkpoint_replay_max_error=maxerror,spill_clean_positive_gradient_cosine=target_cos))
        print(json.dumps(dict(stage='attribution',rank=ctx.rank,id=c['ids'][i])),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
    else:gathered=[rows]
    if ctx.is_main:
        allrows=sorted([r for shard in gathered for r in shard],key=lambda r:(r['case'],r['index']))
        if len({(r['case'],r['index']) for r in allrows})!=2*len(c['ids']):raise RuntimeError('Duplicate attribution source')
        torch.save(dict(rows=allrows,ids=c['ids'],world_size=ctx.world_size,source_counts_per_rank=counts,
            target='Identity-view episode spill-minus-clean training-prototype cosine margin',
            method='All heads and all final-tubelet queries; sum separate prefix derivatives at original key times; unnormalized raw values',
            attribution_scope='Final block only, conditional on frozen contextual pre23 states; not input-pixel attribution or full LeGrad'),destination/'attention_maps.pt')
    # Fresh image-level interventions. Every prefix is independently re-encoded.
    loaded=load_pretrained_encoder(str(a.encoder_dir),local_files_only=True)
    if digest(a.encoder_dir/'model.safetensors')!=c['encoder_sha256']:raise ValueError('Encoder changed')
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    transform=OfficialVJEPA2Transform(str(a.encoder_dir),preserve_field_of_view=True)
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    val=c['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size))
    _,controlcounts=audit_distributed_indices(positions,len(val))
    localmaps={(r['case'],r['index']):r['maps'] for r in rows}
    # Attribution ownership follows global index parity; interventions follow
    # validation positions. Broadcast just the small validation maps.
    needed={}
    for r in rows:
        if r['index'] in val:needed[(r['case'],r['index'])]=r['maps']
    parts=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(parts,needed)
    else:parts=[needed]
    localmaps={k:v for part in parts for k,v in part.items()}
    controls=[]
    def pre23(pixels,n):
        states=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for end in range(1,n+1):
                value=pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16)
                h=encoder.patch_embed(value)
                for block in encoder.blocks[:-1]:h=block(h,frames=end,height=16,width=16)
                states.append(h.detach())
        return states
    for position in positions:
        i=val[position];record=records[c['ids'][i]];source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        indices=c['rows'][i]['sampled_frame_indices'];n=len(indices)//2
        decoded=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices]);original=transform(frames)
        encoded={}
        for name,(model,saved,prototypes) in cases.items():
            if name=='FT0_frozen' and reuse_controls:
                controls.append(reuse_controls[i]);continue
            support=saved['content'][i];candidates=support.nonzero().flatten()
            # Predeclared positive-gradient target, summed over true key times.
            importance=localmaps[(name,i)]['positive_gradient'].sum(0)
            order=candidates[importance[candidates].argsort(stable=True)]
            choices=dict(high=order[-26:],low=order[:26])
            for seed in range(3):choices[f'random_{seed}']=candidates[torch.randperm(len(candidates),generator=torch.Generator().manual_seed(239+i+1000*seed))[:26]]
            conditions=dict(original=(original,('original',)))
            for kind,cells in choices.items():
                mask=torch.zeros(256,dtype=torch.bool);mask[cells]=True
                pixelmask=mask.reshape(16,16).repeat_interleave(16,0).repeat_interleave(16,1)
                conditions[kind]=(original.masked_fill(pixelmask[None,None],0),tuple(sorted(cells.tolist())))
            uniform=frames.float().mean((-2,-1),keepdim=True).round().expand_as(frames).to(torch.uint8)
            conditions['uniform_frames']=(transform(uniform),('uniform_frames',))
            scores={}
            with torch.no_grad():
                for condition,(pixels,key) in conditions.items():
                    if key not in encoded:encoded[key]=pre23(pixels,n)
                    q=torch.stack([model(h,frames=end+1)[0] for end,h in enumerate(encoded[key])])[:,support.to(ctx.device)]
                    z=F.normalize(q.mean((0,1)),dim=-1);s=z@prototypes.T
                    scores[condition]=dict(margin=float(s[1]-s[0]),prediction=int(s.argmax()))
            replay=abs(scores['original']['margin']-float(saved['embeddings'][i,0]@(prototypes[1]-prototypes[0]).cpu()))
            if replay>1e-4:raise RuntimeError(f'Fresh image re-encode failed: {replay:g}')
            controls.append(dict(case=name,index=i,id=c['ids'][i],label=int(c['labels'][i]),conditions=scores,
                replay_margin_error=replay,selected_cells={k:v.tolist() for k,v in choices.items()},removed_content_cells=26))
        print(json.dumps(dict(stage='input_controls',rank=ctx.rank,id=c['ids'][i])),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(controls,gathered,dst=0)
    else:gathered=[controls]
    if ctx.is_main:
        allcontrols=sorted([r for shard in gathered for r in shard],key=lambda r:(r['case'],r['index']))
        if len({(r['case'],r['index']) for r in allcontrols})!=2*len(val):raise RuntimeError('Duplicate control source')
        (destination/'input_controls.json').write_text(json.dumps(dict(rows=allcontrols,world_size=ctx.world_size,rank_source_counts=controlcounts,
            target='Episode spill-minus-clean margin; intervention cells selected by positive spill-target attention gradient, irrespective of validation label',
            intervention='26/192 content cells, all sampled frames; normalized zero mean replacement; 3 seeded equal-area random controls',
            limitation='OOD perturbations without masks; sensitivity is not spill localization accuracy'),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
