#!/usr/bin/env python3
"""Fresh full-encoder input attribution versus contextual patch contributions."""
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
from vision_model.training.input_attribution import input_integrated_gradients
from vision_model.training.patch_vico import PatchProjector
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from visualize_trajectory_video import decode_selected_frames
from train_spatial_semantic import DEFAULT_TEACHER
from diagnose_spill_representation import digest
from control_patch_vico import case_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--retained-run',type=Path,required=True)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--all-validation',action='store_true')
    p.add_argument('--output-name',help='Distinct subdirectory for a quadrature refinement; never overwrite prior maps')
    p.add_argument('--encoder-precision',choices=['bf16','float32'],default='bf16',help='Float32 is a separate numerical reference, not cached inference precision')
    p.add_argument('--encoder-dir',type=Path,default=Path('/home/olemou/hf'))
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    name=a.output_name or ('input_attribution_all_validation' if a.all_validation else 'input_attribution_four_sources')
    if Path(name).name!=name:raise ValueError('Output name must be one directory component')
    destination=a.run/name
    if ctx.is_main:destination.mkdir(exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    if digest(a.encoder_dir/'model.safetensors')!=raw['checkpoint_sha256']:raise ValueError('Encoder hash mismatch')
    controls=json.loads((a.run/'input_controls.json').read_text())
    control_rows={(r['case'],r['id']):r for r in controls['rows']}
    cases={}
    for run in (a.run,a.retained_run):
        report=json.loads((run/'report.json').read_text())
        case=next(c for c in report['cases'] if c['case']=='all_patch')
        folder=run/'all_patch'
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        model=PatchProjector(**ck['metadata']['architecture']).to(ctx.device).eval().requires_grad_(False)
        model.load_state_dict(ck['model'],strict=True)
        saved=torch.load(folder/'representations.pt',map_location='cpu',weights_only=False,mmap=True)
        cases[case_name(case)]=(model,saved['prototypes'].to(ctx.device),saved['content'],saved['embeddings'])
    selected=controls['selected_sources'] if a.all_validation else ['clean_videoFrame121','clean_videoFrame124','cold_folderImage10','cold_folderImage13']
    positions=list(range(ctx.rank,len(selected),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(selected))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    loaded=load_pretrained_encoder(str(a.encoder_dir),local_files_only=True);loaded.freeze_encoder(True)
    encoder_dtype=torch.bfloat16 if a.encoder_precision=='bf16' else torch.float32
    encoder=loaded.encoder.to(ctx.device,dtype=encoder_dtype).eval()
    transform=OfficialVJEPA2Transform(str(a.encoder_dir),preserve_field_of_view=True)
    result=[];outputs={}
    for position in positions:
        source_id=selected[position];i=raw['ids'].index(source_id);r=records[source_id]
        source=Path(raw['dataset_root'])/r['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Video hash mismatch')
        sampled,valid=TemporalSampler(TemporalSamplingConfig(**raw['sampling'])).sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps'])
        indices=sampled[:int(valid.sum())//2*2].tolist();n=len(indices)//2
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Frame indexing mismatch')
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        pixels=transform(frames).to(ctx.device)
        for case,(model,prototypes,contents,embeddings) in cases.items():
            support=contents[i].to(ctx.device)
            def score(observed):
                vectors=[]
                for end in range(1,n+1):
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.encoder_precision=='bf16'):
                        tokens=encoder(observed[:2*end].permute(1,0,2,3)[None].to(encoder_dtype))[0,-256:]
                    vectors.append(model(tokens.float())[support].mean(0))
                z=F.normalize(torch.stack(vectors).mean(0),dim=-1)
                return z@(prototypes[1]-prototypes[0])
            attribution,diagnostic=input_integrated_gradients(score,pixels,torch.zeros_like(pixels),steps=a.steps)
            original=float(embeddings[i,0].to(ctx.device)@(prototypes[1]-prototypes[0]))
            replay=abs(diagnostic['target']-original)
            tolerance=1e-4 if a.encoder_precision=='bf16' else .02
            if replay>tolerance:raise RuntimeError(f'Encoder replay exceeds declared precision-drift tolerance: {replay:g}')
            importance=attribution.abs().sum((0,1))
            cell=importance.reshape(16,16,16,16).sum((1,3)).flatten()
            candidates=support.nonzero().flatten()
            high=candidates[cell[candidates].argsort(stable=True)[-26:]]
            cellmask=torch.zeros(256,dtype=torch.bool,device=ctx.device);cellmask[high]=True
            mask=cellmask.reshape(16,16).repeat_interleave(16,0).repeat_interleave(16,1)
            with torch.no_grad():removed=float(score(pixels.masked_fill(mask[None,None],0)))
            control=control_rows[(case,source_id)];label=int(raw['labels'][i]);sign=2*label-1
            random_drop=sum(sign*(original-control['conditions'][f'random_{j}']['margin']) for j in range(3))/3
            high_drop=sign*(original-removed)
            result.append(dict(case=case,id=source_id,label=label,diagnostics=diagnostic,replay_error=replay,
                ig_high_margin=removed,ig_high_true_class_drop=high_drop,random_true_class_drop=random_drop,
                high_minus_random=high_drop-random_drop,selected_cells=high.tolist(),
                feature_high_true_class_drop=sign*(original-control['conditions']['high']['margin'])))
            outputs[f'{case}/{source_id}']=dict(attribution=attribution.cpu(),frame_indices=indices,diagnostics=diagnostic)
            print(json.dumps(dict(rank=ctx.rank,case=case,source=source_id,relative_completeness_error=diagnostic['relative_completeness_error'],high_minus_random=high_drop-random_drop)),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object((result,outputs),gathered,dst=0)
    else:gathered=[(result,outputs)]
    if ctx.is_main:
        rows=[r for group,_ in gathered for r in group]
        payload={key:value for _,group in gathered for key,value in group.items()}
        torch.save(payload,destination/'pixel_attributions.pt')
        (destination/'results.json').write_text(json.dumps(dict(rows=rows,source_counts_per_rank=counts,encoder_precision=a.encoder_precision,
            method='Input Integrated Gradients, Gauss-Legendre quadrature; actual normalized prototype margin; zero normalized-image baseline; full frozen encoder gradients',
            context='Episode decision uses separately encoded causal prefixes; attribution may include earlier frames through later observed prefixes',
            caveats='Finite quadrature; precision and completeness errors reported; float32 is a reference replay; OOD removal; no spatial ground truth'),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
