#!/usr/bin/env python3
"""Verified full pre-block-23 causal prefixes; distinct sources on every rank."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
import torch.distributed as dist
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.models.final_block_maps import replay_last_tubelet
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.training.final_attention import FinalAttentionStudent
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from visualize_trajectory_video import decode_selected_frames
from train_spatial_semantic import read_inputs,DEFAULT_SPATIAL,DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spatial-cache',type=Path,default=DEFAULT_SPATIAL)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--temporal-checkpoint',type=Path,default=EXPERIMENT/'training/trial_01_checkpoint.pt')
    p.add_argument('--encoder-dir',type=Path,default=Path('/home/olemou/hf'))
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    c,raw,ck,train,val=read_inputs(a)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size))
    _,counts=audit_distributed_indices(assigned,len(c['ids']))
    if len(set(counts))!=1:raise ValueError('Use equal distinct source counts for streaming gathers')
    if digest(a.encoder_dir/'model.safetensors')!=raw['checkpoint_sha256']:raise ValueError('Encoder hash differs')
    loaded=load_pretrained_encoder(str(a.encoder_dir),local_files_only=True)
    # Retain original pretrained FP32 last-block masters before converting prefix.
    suffix=FinalAttentionStudent(loaded.encoder.blocks[-1],loaded.encoder.norm).to(ctx.device).eval()
    initial={k:v.cpu().clone() for k,v in suffix.state_dict().items() if not k.startswith('projector.')}
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    captured={}
    hook=encoder.blocks[-1].register_forward_pre_hook(lambda m,inputs:captured.update(hidden=inputs[0].detach().clone()))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    transform=OfficialVJEPA2Transform(str(a.encoder_dir),preserve_field_of_view=True)
    rows=[]
    for i in assigned:
        record=records[c['ids'][i]];source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        sampled,valid=TemporalSampler(TemporalSamplingConfig(**raw['sampling'])).sample_with_validity(num_frames=record['num_frames'],source_fps=record['fps'])
        n=int(c['valid'][i].sum());indices=sampled[:2*n].tolist()
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist() or not valid[:2*n].all():raise ValueError('Index/padding mismatch')
        decoded=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        prefixes=[];refs=[];errors=[];teacher_errors=[];full_errors=[];precision_errors=[]
        for view in range(2):
            pixels=transform(frames if view==0 else frames.flip(-1))
            vp=[];vr=[]
            for end in range(1,n+1):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=encoder(pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16))
                hidden=captured['hidden']
                errors.append(float((output[0,-256:].float().cpu()-c['tokens'][i,view,end-1,c['layers'].index(23)].float()).abs().max()))
                teacher_errors.append(float((output[0,-256:].mean(0).float().cpu()-raw['features'][i,view,end-1]).abs().max()))
                with torch.no_grad():
                    _,_,tokens=suffix(hidden,frames=end,capture=True)
                    full=suffix.norm(suffix.blocks[-1](hidden.float(),frames=end,height=16,width=16))[:,-256:]
                    full_errors.append(float((tokens-full).abs().max()))
                    precision_errors.append(float((tokens-output[:,-256:].float()).abs().max()))
                vp.append(hidden[0].cpu());vr.append(tokens[0].cpu())
            prefixes.append(vp);refs.append(torch.stack(vr))
        row=dict(index=i,prefixes=prefixes,reference=torch.stack(refs),
                 accepted_cache_max_error=max(errors),fp32_full_replay_max_error=max(full_errors),
                 original_teacher_mean_max_error=max(teacher_errors),
                 bf16_to_fp32_suffix_max_difference=max(precision_errors),sampled_frame_indices=indices)
        # The old spatial file stores FP16 layer-normalized tokens, while the
        # original temporal teacher stores a BF16-reduced mean. Verify the
        # authoritative original mean exactly and report patch precision drift.
        if max(teacher_errors)>1e-5 or max(full_errors)>5e-4:raise RuntimeError(f'Cache replay failed: teacher={max(teacher_errors)}, FP32={max(full_errors)}')
        gathered=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(row,gathered,dst=0)
        else:gathered=[row]
        if ctx.is_main:rows.extend(gathered)
        print(json.dumps(dict(rank=ctx.rank,source=c['ids'][i],teacher_error=max(teacher_errors),stored_patch_error=max(errors),fp32_replay_error=max(full_errors))),flush=True)
    hook.remove()
    if ctx.is_main:
        rows.sort(key=lambda r:r['index'])
        if [r['index'] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Source ownership failed')
        result=dict(ids=c['ids'],labels=c['labels'],valid=c['valid'],timestamps=c['timestamps'],views=c['views'],
                    train_indices=train,validation_indices=val,rows=rows,suffix_state=initial,
                    source_sha256=raw['video_sha256'],teacher_cache_sha256=digest(a.teacher_cache),
                    accepted_spatial_sha256=digest(a.spatial_cache),encoder_sha256=raw['checkpoint_sha256'],
                    temporal_context_policy=c['temporal_context_policy'],world_size=ctx.world_size,rank_source_counts=counts,
                    precision='Frozen blocks 0-22 BF16 (original pipeline); block 23 and final norm original FP32 masters, no autocast in matched FT0/FT1')
        torch.save(result,a.output/'pre23.pt')
        summary={k:v for k,v in result.items() if k not in ('rows','suffix_state','labels','valid','timestamps','train_indices','validation_indices')}
        summary.update(max_native_error=max(r['accepted_cache_max_error'] for r in rows),
                       max_original_teacher_mean_error=max(r['original_teacher_mean_max_error'] for r in rows),
                       max_fp32_replay_error=max(r['fp32_full_replay_max_error'] for r in rows),
                       max_suffix_precision_difference=max(r['bf16_to_fp32_suffix_max_difference'] for r in rows),
                       cache_sha256=digest(a.output/'pre23.pt'))
        (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
