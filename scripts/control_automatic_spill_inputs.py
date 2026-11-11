#!/usr/bin/env python3
"""Matched image interventions against original/control/joint trajectory decisions."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.training.trajectory import TrajectoryModel
from vision_model.training.spill_localization import SpillPatchHead
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.interpretability.normal_patch_controls import ranked_cells,remove_cells
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from train_spill_localization import build_student,DEFAULT_PREFIX,DEFAULT_TEACHER
from train_spatial_semantic import EXPERIMENT
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dense',action='store_true',help='Exhaustive 32x32 input-region sensitivity instead of selected-region comparison');a=p.parse_args()
    torch.set_num_threads(4);ctx=initialize_distributed()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False);cache=torch.load(DEFAULT_PREFIX,map_location='cpu',weights_only=False,mmap=True)
    oldpath=EXPERIMENT/'training/trial_01_checkpoint.pt';original=torch.load(oldpath,map_location='cpu',weights_only=False)
    reference=torch.load(a.root/'normal_reference/maps.pt',map_location='cpu',weights_only=False)
    trained_maps=torch.load(a.root/'automatic_maps/maps.pt',map_location='cpu',weights_only=False)
    models={};fields={'block5_raw':reference['maps']['block5_raw'][:,0]}
    support=reference['content'];train=reference['train_indices'];val=reference['validation_indices'].tolist()
    baseline=TrajectoryModel(original['width'],original['hidden'],original['embedding_dim']).to(ctx.device).eval().requires_grad_(False)
    baseline.load_state_dict(original['model'],strict=True)
    with torch.no_grad():z=baseline(raw['features'][:,0].to(ctx.device),raw['valid'].to(ctx.device),raw['timestamps'].to(ctx.device))['embedding']
    labels=raw['labels'].to(ctx.device)
    proto=torch.stack([F.normalize(z[train][labels[train]==c].mean(0),dim=0) for c in (0,1)])
    models['original']=(None,baseline,proto,z)
    hashes={'original':digest(oldpath)}
    for case in ('automatic_control','automatic_joint'):
        path=a.root/case/'checkpoint.pt';saved=torch.load(path,map_location='cpu',weights_only=False);rep=torch.load(path.parent/'representations.pt',map_location='cpu',weights_only=False)
        student=build_student(cache,SpillPatchHead(**saved['metadata']['head_configuration']),train_attention=False).to(ctx.device).eval().requires_grad_(False)
        student.load_state_dict(saved['model'],strict=True)
        temporal=TrajectoryModel(original['width'],original['hidden'],original['embedding_dim']).to(ctx.device).eval().requires_grad_(False);temporal.load_state_dict(saved['trajectory'],strict=True)
        models[case]=(student,temporal,rep['prototypes'].to(ctx.device),rep['embeddings'].to(ctx.device));hashes[case]=digest(path)
    probe=torch.load(a.root/'automatic_probe/representations.pt',map_location='cpu',weights_only=False)
    fields['probe_local']=probe['local_logits'][:,0].sigmoid()
    selected=sorted([r for r in trained_maps['rows'] if r['case']=='automatic_joint'],key=lambda r:r['index'])
    fields['joint_local']=torch.stack([r['local_probabilities'] for r in selected])
    fields['joint_decision_gradient']=torch.stack([r['maps']['positive_gradient'] for r in selected])
    loaded=load_pretrained_encoder('/home/olemou/hf',local_files_only=True)
    if digest(Path('/home/olemou/hf/model.safetensors'))!=raw['checkpoint_sha256']:raise ValueError('Encoder differs')
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    capture={};hook=encoder.blocks[-1].register_forward_pre_hook(lambda _m,inputs:capture.update(hidden=inputs[0].detach()))
    transform=OfficialVJEPA2Transform('/home/olemou/hf',preserve_field_of_view=True)
    sampler=TemporalSampler(TemporalSamplingConfig(**raw['sampling']))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    positions=list(range(ctx.rank,len(val),ctx.world_size));_,counts=audit_distributed_indices(positions,len(val));rows=[];begin=time.monotonic()
    for position in positions:
        i=val[position];r=records[raw['ids'][i]];source=Path(raw['dataset_root'])/r['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        n=int(raw['valid'][i].sum());sampled,_=sampler.sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps']);indices=sampled[:2*n].tolist()
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Wrong sampled endpoints')
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        pixels=transform(torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])).to(ctx.device)
        choices={}
        for name,maps in fields.items():
            value=maps[i,:n].mean(0);choices[name+'_high']=ranked_cells(value,support[i]);choices[name+'_low']=ranked_cells(value,support[i],lowest=True)
        gen=torch.Generator().manual_seed(239+i)
        for j in range(3):choices['random'+str(j)]=torch.where(support[i])[0][torch.randperm(int(support[i].sum()),generator=gen)[:26]]
        if a.dense:
            choices={}
            for row in range(8):
                for col in range(8):
                    cells=torch.tensor([32*row+2*col,32*row+2*col+1,32*row+2*col+16,32*row+2*col+17])
                    if support[i,cells].all():choices[f'cell_{row}_{col}']=cells
        results={case:{} for case in models}
        for fill in ('mean','blur'):
            names=list(choices)
            for start in range(0,len(names),3):
                names_batch=names[start:start+3]
                clips=[pixels]+[remove_cells(pixels,choices[k].to(ctx.device),support[i].to(ctx.device),fill=fill) for k in names_batch]
                while len(clips)<4:clips.append(pixels)
                batch=torch.stack(clips);means={case:torch.zeros(4,n,original['width'],device=ctx.device) for case in models}
                with torch.no_grad():
                    for end in range(1,n+1):
                        with torch.autocast('cuda',dtype=torch.bfloat16):tokens=encoder(batch[:,:2*end].permute(0,2,1,3,4).to(torch.bfloat16))
                        means['original'][:,end-1]=tokens[:,-256:].mean(1).float()
                        for case,(student,_,_,_) in models.items():
                            if student is not None:means[case][:,end-1]=student(capture['hidden'],frames=end)['student_features']
                    for case,(_,temporal,prototypes,embeddings) in models.items():
                        output=temporal(means[case],torch.ones(4,n,dtype=torch.bool,device=ctx.device),raw['timestamps'][i,:n].to(ctx.device)[None].expand(4,-1),detach_teacher=False)
                        scores=output['embedding']@prototypes.T;margin=scores[:,1]-scores[:,0]
                        error=abs(float(margin[0])-float(embeddings[i]@(prototypes[1]-prototypes[0])))
                        if error>(.02 if case=='original' else 1e-4):raise RuntimeError(f'{case} replay error {error}')
                        for j,name in enumerate(names_batch,1):results[case][fill+'/'+name]=dict(original_margin=float(margin[0]),margin_drop=float(margin[0]-margin[j]),perturbed_margin=float(margin[j]),prediction=int(scores[j].argmax()),replay_error=error)
        rows.append(dict(id=raw['ids'][i],index=i,label=int(raw['labels'][i]),choices={k:v.tolist() for k,v in choices.items()},cases=results))
        print(json.dumps(dict(rank=ctx.rank,source=raw['ids'][i],elapsed=time.monotonic()-begin)),flush=True)
    hook.remove();parts=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,parts,dst=0)
    else:parts=[rows]
    if ctx.is_main:
        rows=sorted([r for part in parts for r in part],key=lambda r:r['index'])
        if sorted(r['index'] for r in rows)!=sorted(val):raise RuntimeError('Duplicated/missing evaluated source')
        summary={};gen=torch.Generator().manual_seed(239)
        for case in ([] if a.dense else models):
            for label in (0,1):
                subset=[r for r in rows if r['label']==label]
                for fill in ('mean','blur'):
                    for field in fields:
                        diff=torch.tensor([r['cases'][case][fill+'/'+field+'_high']['margin_drop']-sum(r['cases'][case][fill+'/random'+str(j)]['margin_drop'] for j in range(3))/3 for r in subset])
                        boot=diff[torch.randint(len(diff),(10000,len(diff)),generator=gen)].mean(1)
                        summary[f'{case}/{label}/{fill}/{field}']=dict(sources=len(diff),mean_high_minus_random_margin_drop=float(diff.mean()),exploratory_interval95=[float(torch.quantile(boot,q)) for q in (.025,.975)],accuracy_after_high_removal=sum(r['cases'][case][fill+'/'+field+'_high']['prediction']==label for r in subset)/len(subset))
        if a.dense:
            dense=[]
            for row in rows:
                fields={}
                for case,values in row['cases'].items():
                    fields[case]={}
                    for fill in ('mean','blur'):
                        grid=torch.zeros(8,8)
                        for name in row['choices']:
                            _,r,c=name.split('_');grid[int(r),int(c)]=values[fill+'/'+name]['margin_drop']
                        fields[case][fill]=grid
                dense.append(dict(id=row['id'],index=row['index'],label=row['label'],fields=fields))
            torch.save(dict(rows=dense,valid=raw['valid'],timestamps=raw['timestamps'],content=support,
                scope='Actual input-region episode sensitivity; constant spatial region is removed over all sampled frames, not per-frame attribution'),a.output/'maps.pt')
        (a.output/'results.json').write_text(json.dumps(dict(metadata=dict(world_size=ctx.world_size,rank_source_counts=counts,checkpoint_sha256=hashes,
            protocol=('Exhaustive disjoint 32x32 content regions, mean/blur fills over all sampled frames' if a.dense else 'Same 26 native cells removed over all sampled frames; independent 3-random controls, mean/blur fills')+'; original/control/joint decisions share fresh causal-prefix encodes, actual learned final suffixes and training-only prototypes',
            no_masks=True,manual_annotations=0,independent_scene_test=False,optimizer_updates=0,code_sha256=digest(Path(__file__))),summary=summary,rows=rows),indent=2)+'\n')
    if ctx.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
