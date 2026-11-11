#!/usr/bin/env python3
"""Cache local relative measurements and unverified within-frame region pairs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
import torch.distributed as dist
from vision_model.training.local_thermal import LocalThermalSettings,local_relative_features,region_pairs,compact_pair
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.dense_encoder import file_sha256
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dense-cache',type=Path,default=ROOT/'reports/experiments/spill_dense_fix_80_20261022/vjepa21_384.pt')
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed();settings=LocalThermalSettings()
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    dense=torch.load(a.dense_cache,map_location='cpu',mmap=True,weights_only=False)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    if dense['ids']!=raw['ids'] or len(set(raw['video_sha256']))!=len(raw['ids']):raise ValueError('Sources differ or duplicate')
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    owned=list(range(ctx.rank,len(raw['ids']),ctx.world_size));_,counts=audit_distributed_indices(owned,len(raw['ids']))
    rows=[]
    for i in owned:
        r=records[raw['ids'][i]];path=Path(raw['dataset_root'])/r['video_path']
        if file_sha256(path)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        if raw['labels'][i]==1 and r['category']!='cold':raise ValueError('This experiment is explicitly cold-only')
        indices=dense['rows'][i]['sampled_indices'];decoded=decode_selected_frames(path,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices]).float()
        # Separate full-content local path, not a changed V-JEPA input transform.
        h,w=288,384
        if r['height']*w!=r['width']*h:raise ValueError('Explicit 4:3 diagnostic requires a new protocol for other aspect ratios')
        gray=F.interpolate(frames.mean(1,keepdim=True),size=(h,w),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).to(torch.uint8)
        with torch.no_grad():
            features=torch.stack([local_relative_features((gray if view==0 else gray.flip(-1)).to(ctx.device).float()/255,settings).cpu().half() for view in range(2)])
        pairs=[[compact_pair(p) for p in region_pairs(image[0].numpy()/255,settings)] for image in gray]
        rows.append(dict(index=i,gray=gray,features=features,pairs=pairs,sampled_indices=indices))
        print(json.dumps(dict(rank=ctx.rank,id=raw['ids'][i],pairs=sum(map(len,pairs)))),flush=True)
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
    else:pieces=[rows]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r['index'])
        if [r['index'] for r in rows]!=list(range(len(raw['ids']))):raise RuntimeError('Repeated/missing source')
        features=torch.stack([r.pop('features') for r in rows]);gray=torch.stack([r.pop('gray') for r in rows])
        pairs=[r.pop('pairs') for r in rows]
        valid=dense['valid'].repeat_interleave(2,1)
        if not valid.all():raise ValueError('Variable-tail source requires per-source frame-cache padding')
        metadata=dict(settings=settings.state_dict(),world_size=ctx.world_size,rank_source_counts=counts,
            source_sha256=raw['video_sha256'],code_sha256=file_sha256(__file__),module_sha256=file_sha256(ROOT/'src/vision_model/training/local_thermal.py'),
            source_split='Original64/16 source-selection split',manual_annotations=0,scene_independent=False,
            teacher='Unverified within-image rank-component/local-ring comparisons; no external clean reference',
            feature_receptive_radius_max=39,preprocessing='Full-content384x288 grayscale uint8; separate from unchanged V-JEPA transform',
            causal=True,temporal_consistency_enabled=False,temporal_reason='No verified registration/occlusion correspondences; frame-local proposal only')
        cache=dict(ids=raw['ids'],labels=raw['labels'],train_indices=dense['train_indices'],validation_indices=dense['validation_indices'],
            valid=valid,features=features,gray=gray,pairs=pairs,rows=rows,metadata=metadata)
        torch.save(cache,a.output/'cache.pt')
        coverage=[dict(id=cache['ids'][i],label=int(cache['labels'][i]),split='train' if i in cache['train_indices'] else 'selection',
            pairs=sum(map(len,pairs[i])),frames_with_pairs=sum(bool(x) for x in pairs[i]),
            region_pixel_instances=sum(p['pixels'] for time in pairs[i] for p in time)) for i in range(len(rows))]
        (a.output/'report.json').write_text(json.dumps(dict(metadata=metadata,cache_sha256=file_sha256(a.output/'cache.pt'),coverage=coverage),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
