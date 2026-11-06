#!/usr/bin/env python3
"""Real-input faithfulness controls of novelty maps against the original decision."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.interpretability.normal_patch_controls import ranked_cells,remove_cells
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from visualize_trajectory_video import decode_selected_frames
from train_spatial_semantic import DEFAULT_TEACHER,EXPERIMENT
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--normal-student',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-reference',type=Path);p.add_argument('--cases',nargs='+')
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    if ctx.device.type!='cuda':raise RuntimeError('Fresh real encodes require the allocated GPUs')
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    reference=torch.load(a.run/'maps.pt',map_location='cpu',weights_only=False)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    ckpath=EXPERIMENT/'training/trial_01_checkpoint.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    loaded=load_pretrained_encoder('/home/olemou/hf',local_files_only=True)
    if digest(Path('/home/olemou/hf/model.safetensors'))!=raw['checkpoint_sha256']:raise ValueError('Encoder changed')
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    model=TrajectoryModel(ck['width'],ck['hidden'],ck['embedding_dim']).to(ctx.device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True)
    with torch.no_grad():
        z=model(raw['features'][:,0].to(ctx.device),raw['valid'].to(ctx.device),raw['timestamps'].to(ctx.device))['embedding']
    train=reference['train_indices'];labels=raw['labels'].to(ctx.device)
    proto=torch.stack([F.normalize(z[train][labels[train]==k].mean(0),dim=0) for k in (0,1)])
    fields={k:reference['maps'][k] for k in ('block5_raw','block17_raw','block23_raw')}
    for arm in ('normal_cosine','normal_contrast'):
        saved=torch.load(a.normal_student/arm/'maps.pt',map_location='cpu',weights_only=False)
        if saved['ids']!=raw['ids']:raise ValueError('Spatial student source mismatch')
        fields[arm]=saved['maps']
    if a.extra_reference:
        extra=torch.load(a.extra_reference,map_location='cpu',weights_only=False)
        if extra['ids']!=raw['ids'] or not torch.equal(extra['train_indices'],reference['train_indices']):raise ValueError('Extra reference source/split mismatch')
        fields.update(extra['maps'])
    if a.cases:fields={key:fields[key] for key in a.cases}
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    transform=OfficialVJEPA2Transform('/home/olemou/hf',preserve_field_of_view=True)
    sampler=TemporalSampler(TemporalSamplingConfig(**raw['sampling']))
    val=reference['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size));_,counts=audit_distributed_indices(positions,len(val))
    rows=[];begin=time.monotonic()
    for position in positions:
        i=val[position];r=records[raw['ids'][i]];source=Path(raw['dataset_root'])/r['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        n=int(raw['valid'][i].sum());sampled,_=sampler.sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps'])
        indices=sampled[:2*n].tolist()
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Invalid frame alignment')
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        pixels=transform(frames).to(ctx.device);support=reference['content'][i]
        choices={}
        for name,maps in fields.items():
            field=maps[i,0,:n].mean(0)
            choices[name+'_high']=ranked_cells(field,support)
            choices[name+'_low']=ranked_cells(field,support,lowest=True)
        gen=torch.Generator().manual_seed(239+i)
        for seed in range(3):choices[f'random{seed}']=torch.where(support)[0][torch.randperm(int(support.sum()),generator=gen)[:26]]
        result={}
        for fill in ('mean','blur'):
            names=list(choices)
            for start in range(0,len(names),3):
                selected=names[start:start+3]
                clips=[pixels]+[remove_cells(pixels,choices[name].to(ctx.device),support.to(ctx.device),fill=fill) for name in selected]
                # Keep a fixed batch of four even for the final partial group.
                while len(clips)<4:clips.append(pixels)
                batch=torch.stack(clips);features=torch.zeros(4,n,ck['width'],device=ctx.device)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for end in range(1,n+1):
                        tokens=encoder(batch[:,:2*end].permute(0,2,1,3,4).to(torch.bfloat16))
                        features[:,end-1]=tokens[:,-256:].mean(1).float()
                with torch.no_grad():
                    out=model(features,torch.ones(4,n,dtype=torch.bool,device=ctx.device),raw['timestamps'][i,:n].to(ctx.device)[None].expand(4,-1),detach_teacher=False)
                    scores=out['embedding']@proto.T;margin=scores[:,1]-scores[:,0]
                cached=float(z[i]@(proto[1]-proto[0]));error=abs(float(margin[0])-cached)
                if error>.02:raise RuntimeError(f'Original replay mismatch {error}')
                for j,name in enumerate(selected,1):result[fill+'/'+name]=dict(original_margin=float(margin[0]),perturbed_margin=float(margin[j]),margin_drop=float(margin[0]-margin[j]),prediction=int(scores[j].argmax()),cached_replay_error=error)
        rows.append(dict(id=raw['ids'][i],index=i,label=int(raw['labels'][i]),choices={k:v.tolist() for k,v in choices.items()},controls=result))
        print(json.dumps(dict(rank=ctx.rank,completed=raw['ids'][i],elapsed=time.monotonic()-begin)),flush=True)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
    else:gathered=[rows]
    if ctx.is_main:
        rows=sorted([r for shard in gathered for r in shard],key=lambda r:r['index'])
        if sorted(r['index'] for r in rows)!=sorted(val):raise RuntimeError('Repeated or absent evaluated source')
        summary={};gen=torch.Generator().manual_seed(239)
        for label in (0,1):
            selected=[r for r in rows if r['label']==label]
            for fill in ('mean','blur'):
                for name in fields:
                    advantage=torch.tensor([r['controls'][fill+'/'+name+'_high']['margin_drop']-sum(r['controls'][fill+'/random'+str(j)]['margin_drop'] for j in range(3))/3 for r in selected])
                    bootstrap=advantage[torch.randint(len(advantage),(10000,len(advantage)),generator=gen)].mean(1)
                    summary[f'{label}/{fill}/{name}']=dict(sources=len(selected),mean_high_minus_random_margin_drop=float(advantage.mean()),exploratory_paired_source_interval95=[float(torch.quantile(bootstrap,q)) for q in (.025,.975)],
                        accuracy_after_high_removal=sum(r['controls'][fill+'/'+name+'_high']['prediction']==label for r in selected)/len(selected))
        metadata=dict(world_size=ctx.world_size,rank_source_counts=counts,device=torch.cuda.get_device_name(ctx.device),
            protocol='26 distinct native content cells, same coordinates through all sampled frames; high/low novelty versus 3 seeded equal-area random masks; observed-content mean and 33x33 blur fills; every causal prefix freshly encoded',
            target='Actual original TrajectoryModel episode spill-minus-clean training-prototype margin',
            checkpoint_sha256=digest(ckpath),novelty_maps_sha256=digest(a.run/'maps.pt'),source_sha256=raw['video_sha256'],
            code_sha256=digest(Path(__file__)),manual_annotations=0,optimizer_updates=0,independent_scene_test=False,
            limitations='No liquid-region identities. A high-minus-random advantage supports model sensitivity to proposed regions, not true spill localization. Multiple exploratory comparisons, intervals unadjusted.',elapsed_seconds=time.monotonic()-begin)
        (a.output/'results.json').write_text(json.dumps(dict(metadata=metadata,summary=summary,rows=rows),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
