#!/usr/bin/env python3
"""Cold-only automatic pixel-contrast targets, calibrated on training clean clips."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from vision_model.training.thermal_candidates import cold_contrast_grid,cold_pixel_targets,cold_contrast_from_encoder_pixels
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.dense_spill import candidate_coverage
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.dense_encoder import file_sha256
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--encoder-aligned',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.root/'vjepa21_384.pt',map_location='cpu',weights_only=False,mmap=True)
    feature=torch.load(a.root/'vjepa21_consensus/candidates.pt',map_location='cpu',weights_only=False)
    prior=json.loads((a.root/'vjepa21_consensus/report.json').read_text())
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    assigned=list(range(ctx.rank,len(c['ids']),ctx.world_size));_,counts=audit_distributed_indices(assigned,len(c['ids']))
    rows=[]
    transform=OfficialVJEPA2Transform('/home/olemou/hf',output_size=c['metadata']['resolution']) if a.encoder_aligned else None
    for i in assigned:
        r=records[c['ids'][i]]
        if c['labels'][i]==1 and r['category']!='cold':raise ValueError('This teacher is explicitly cold-only')
        path=Path(raw['dataset_root'])/r['video_path']
        if file_sha256(path)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        indices=c['rows'][i]['sampled_indices']
        decoded=decode_selected_frames(path,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        score=(cold_contrast_from_encoder_pixels(transform(frames).to(ctx.device),r['height'],r['width'])
               if transform is not None else cold_contrast_grid(frames.to(ctx.device),size=c['metadata']['resolution'])).cpu()
        rows.append((i,score))
        print(json.dumps(dict(rank=ctx.rank,id=c['ids'][i])),flush=True)
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
    else:pieces=[rows]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r[0])
        if [r[0] for r in rows]!=list(range(len(c['ids']))):raise RuntimeError('Source ownership mismatch')
        scores=torch.zeros(len(rows),c['valid'].shape[1],c['tokens'].shape[-2])
        for i,score in rows:scores[i,:len(score)]=score
        train=c['train_indices'];clean=train[c['labels'][train]==0];support=c['content'].flatten(1)
        values=torch.cat([scores[i,c['valid'][i]][:,support[i]].flatten() for i in clean])
        low,high=[float(torch.quantile(values,q)) for q in (.95,.999)]
        target=cold_pixel_targets(scores,feature['targets'],c['labels'],c['valid'],support,low=low,high=high)
        report=dict(metadata={**prior['metadata'],'code_sha256':file_sha256(__file__),
            'policy':'Cold-only pixel local contrast; 31/63 windows, source-clean q99.9 FG; feature-normal AND cold-score below q95 BG; no ROI or fixed area',
            'teacher_assumption':'Cold liquid is apparently darker than its local surroundings; dark objects/shoes remain confounders',
            'input_alignment':'encoder_input_uint8_reconstruction_v1' if a.encoder_aligned else 'source_pixels',
            'world_size':ctx.world_size,'rank_source_counts':counts},
            calibration=dict(low_q95=low,high_q999=high),coverage=candidate_coverage(target,c['labels'],train))
        feature['targets']=target;feature['cold_contrast']=scores
        torch.save(feature,a.output/'candidates.pt')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['coverage']),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
