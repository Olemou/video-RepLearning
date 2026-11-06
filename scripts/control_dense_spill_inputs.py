#!/usr/bin/env python3
"""Fresh-image removal against the actual dense control/joint decisions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from vision_model.models.dense_encoder import load_dense_encoder,file_sha256
from vision_model.training.dense_spill import DenseSpillReadout
from vision_model.training.thermal_candidates import cold_contrast_from_encoder_pixels
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.interpretability.normal_patch_controls import ranked_cells,remove_cells
from vision_model.data.transforms import OfficialVJEPA2Transform
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['dense_control','dense_joint'])
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.root/'vjepa21_384.pt',map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    cases=a.cases;models={};reps={};hashes={};settings={}
    for case in cases:
        path=a.root/case/'checkpoint.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        model=DenseSpillReadout(**saved['metadata']['model_configuration']).to(ctx.device).eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True);models[case]=model
        settings[case]=saved['metadata']
        reps[case]=torch.load(path.parent/'representations.pt',map_location='cpu',weights_only=False)
        hashes[case]=file_sha256(path)
    encoder=load_dense_encoder(a.checkpoint,expected_sha256=c['metadata']['checkpoint_sha256']).to(device=ctx.device,dtype=torch.bfloat16)
    size=c['metadata']['resolution'];grid=tuple(c['metadata']['grid'])
    transform=OfficialVJEPA2Transform('/home/olemou/hf',output_size=size,preserve_field_of_view=True)
    indices=c['validation_indices'].tolist();assigned=list(range(ctx.rank,len(indices),ctx.world_size))
    _,counts=audit_distributed_indices(assigned,len(indices));rows=[]
    for slot in assigned:
        i=indices[slot];r=records[c['ids'][i]];source=Path(raw['dataset_root'])/r['video_path']
        if file_sha256(source)!=raw['video_sha256'][i]:raise ValueError('Source bytes changed')
        sampled=c['rows'][i]['sampled_indices'];n=int(c['valid'][i].sum());support=c['content'][i].flatten()
        decoded=decode_selected_frames(source,sampled,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in sampled])
        pixels=transform(frames).to(ctx.device)
        valid=c['valid'][i:i+1,:n].to(ctx.device);content=support[None].to(ctx.device)
        def predict(value):
            features=[]
            for end in range(1,n+1):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    tokens=encoder(value[:2*end].permute(1,0,2,3)[None])
                features.append(tokens[:,-grid[0]*grid[1]:].float())
            patches=torch.stack(features,1)
            result={}
            for case in cases:
                x=patches
                if settings[case].get('cold_pixel_input'):
                    descriptor=cold_contrast_from_encoder_pixels(value,r['height'],r['width'])/settings[case]['cold_input_scale']
                    x=torch.cat((patches,descriptor[None,...,None]),-1)
                with torch.no_grad():result[case]=models[case](x,valid,content)
            return result
        baseline=predict(pixels)
        replay={case:float((baseline[case]['embedding'][0].cpu()-reps[case]['embeddings'][i]).abs().max()) for case in cases}
        if max(replay.values())>1e-4:raise RuntimeError(f'Fresh-image checkpoint replay failed {replay}')
        count=round(int(support.sum())*26/192)
        choices={}
        for case in cases:
            field=reps[case]['rows'][i][1]['contributions'][0,:n].clamp_min(0).mean(0)
            choices[case+'_high']=ranked_cells(field,support,count)
            choices[case+'_low']=ranked_cells(field,support,count,lowest=True)
        eligible=torch.where(support)[0]
        for seed in range(3):
            choices[f'random_{seed}']=eligible[torch.randperm(len(eligible),generator=torch.Generator().manual_seed(239+i*7+seed))[:count]]
        measurements=[]
        for fill in ['mean','blur']:
            for name,cells in choices.items():
                changed=remove_cells(pixels,cells.to(ctx.device),support.to(ctx.device),fill=fill,grid=grid)
                output=predict(changed)
                for case in cases:
                    before=float(baseline[case]['margin'][0]);after=float(output[case]['margin'][0])
                    measurements.append(dict(case=case,selection=name,fill=fill,before=before,after=after,margin_drop=before-after))
        rows.append(dict(index=i,id=c['ids'][i],label=int(c['labels'][i]),replay_errors=replay,
            cells_removed=count,content_cells=int(support.sum()),choices={k:v.tolist() for k,v in choices.items()},measurements=measurements))
        print(json.dumps(dict(rank=ctx.rank,id=c['ids'][i],replay=replay)),flush=True)
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
    else:pieces=[rows]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r['index'])
        if sorted(r['index'] for r in rows)!=sorted(indices):raise RuntimeError('Repeated/missing evaluation')
        summary={}
        for case in cases:
            summary[case]={}
            for fill in ['mean','blur']:
                values=[]
                for r in rows:
                    if r['label']!=1:continue
                    selected=[m for m in r['measurements'] if m['case']==case and m['fill']==fill]
                    high=next(m['margin_drop'] for m in selected if m['selection']==case+'_high')
                    random=[m['margin_drop'] for m in selected if m['selection'].startswith('random_')]
                    values.append(high-sum(random)/len(random))
                x=torch.tensor(values,dtype=torch.float64)
                samples=x[torch.randint(len(x),(2000,len(x)),generator=torch.Generator().manual_seed(239))].mean(1)
                summary[case][fill]=dict(cold_sources=len(x),mean_high_minus_random=float(x.mean()),
                    paired_source_bootstrap_95=torch.quantile(samples,torch.tensor([.025,.975],dtype=torch.float64)).tolist())
        result=dict(metadata=dict(world_size=ctx.world_size,rank_source_counts=counts,
            device=torch.cuda.get_device_name(ctx.device),checkpoint_sha256=hashes,
            encoder_checkpoint_sha256=c['metadata']['checkpoint_sha256'],code_sha256=file_sha256(__file__),
            target='Actual respective dense spill-minus-clean training-prototype margin',
            protocol='About 13.5% native content cells; fixed coordinates over sampled frames; high/low vs 3 seeded random masks; mean/blur fills; all prefixes freshly encoded',
            scope='Input sensitivity, not ground-truth liquid localization; bootstrap intervals exploratory and unadjusted'),
            summary=summary,rows=rows)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
