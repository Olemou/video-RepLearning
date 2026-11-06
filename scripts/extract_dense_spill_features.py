#!/usr/bin/env python3
"""Frozen V-JEPA 2/2.1 patch cache, matched 384 grid, extra training-clean windows."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import torch
import torch.distributed as dist
from vision_model.models.dense_encoder import load_dense_encoder, file_sha256
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.data.sampling import TemporalSampler, TemporalSamplingConfig
from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from train_spatial_semantic import DEFAULT_TEACHER, EXPERIMENT
from visualize_trajectory_video import decode_selected_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['vjepa2', 'vjepa21'], required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt')
    p.add_argument('--encoder-dir', type=Path, default=Path('/home/olemou/hf'))
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    ctx = initialize_distributed()
    if ctx.device.type != 'cuda': raise RuntimeError('Real CUDA extraction required')
    if a.output.exists(): raise FileExistsError(a.output)
    raw = torch.load(DEFAULT_TEACHER, map_location='cpu', weights_only=False)
    trained = torch.load(EXPERIMENT/'training/trial_01_checkpoint.pt', map_location='cpu', weights_only=False)
    train = [raw['ids'].index(name) for name in trained['train_source_ids']]
    val = [raw['ids'].index(name) for name in trained['validation_source_ids']]
    if set(train) & set(val) or sorted(train+val) != list(range(len(raw['ids']))): raise ValueError('Bad source split')
    if len(set(raw['video_sha256'])) != len(raw['ids']): raise ValueError('Duplicate videos')
    if a.model == 'vjepa21':
        encoder = load_dense_encoder(a.checkpoint)
        provenance = encoder.provenance
    else:
        encoder = load_pretrained_encoder(str(a.encoder_dir), local_files_only=True).encoder.eval().requires_grad_(False)
        provenance = dict(model_id=raw['model_id'], checkpoint_sha256=file_sha256(a.encoder_dir/'model.safetensors'))
    encoder.to(device=ctx.device, dtype=torch.bfloat16)
    transform = OfficialVJEPA2Transform(str(a.encoder_dir), preserve_field_of_view=True, output_size=a.size)
    records = {r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    assigned = list(range(ctx.rank,len(raw['ids']),ctx.world_size))
    _, counts = audit_distributed_indices(assigned,len(raw['ids']))
    side = a.size // 16; rows = []
    sampler = TemporalSampler(TemporalSamplingConfig(**raw['sampling']))
    for i in assigned:
        r = records[raw['ids'][i]]; path = Path(raw['dataset_root'])/r['video_path']
        if file_sha256(path) != raw['video_sha256'][i]: raise ValueError('Source changed')
        n = int(raw['valid'][i].sum())
        sampled, valid = sampler.sample_with_validity(num_frames=r['num_frames'], source_fps=r['fps'])
        indices = sampled[:2*n].tolist()
        if indices[1::2] != raw['frame_indices'][i,:n].tolist() or not valid[:2*n].all(): raise ValueError('Sampling mismatch')
        extra_indices = []
        if i in train and raw['labels'][i] == 0:
            offsets = torch.arange(2*n, dtype=torch.float64) * r['fps']/raw['sampling']['target_fps']
            max_start = max(0., r['num_frames']-1-float(offsets[-1]))
            extra_indices = [(offsets+start).round().long().clamp_max(r['num_frames']-1).tolist() for start in (0.,max_start)]
            extra_indices = [idx for j,idx in enumerate(extra_indices) if idx != indices and idx not in extra_indices[:j]]
        all_indices = sorted(set(indices + [x for seq in extra_indices for x in seq]))
        decoded = decode_selected_frames(path, all_indices, frame_count=r['num_frames'], width=r['width'], height=r['height'])
        def pixels(idx):
            return torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in idx])
        central = pixels(indices); views = []
        for view in range(2):
            x = transform(central if view==0 else central.flip(-1))
            features = []
            for end in range(1,n+1):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    tokens = encoder(x[:2*end].permute(1,0,2,3)[None].to(ctx.device))
                features.append(tokens[0,-side*side:].to(dtype=torch.float16,device='cpu'))
            views.append(torch.stack(features))
        extras = []
        for idx in extra_indices:
            clip = pixels(idx); paired = []
            for view in range(2):
                x = transform(clip if view==0 else clip.flip(-1))
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    tokens = encoder(x.permute(1,0,2,3)[None].to(ctx.device))
                paired.append(tokens[0,-side*side:].to(dtype=torch.float16,device='cpu'))
            extras.append(torch.stack(paired))
        rh = round(r['height']*min(a.size/r['height'],a.size/r['width']))
        rw = round(r['width']*min(a.size/r['height'],a.size/r['width']))
        mask = torch.zeros(a.size,a.size);top,left=(a.size-rh)//2,(a.size-rw)//2
        mask[top:top+rh,left:left+rw]=1
        support = torch.nn.functional.avg_pool2d(mask[None,None],16)[0,0]>.5
        rows.append(dict(index=i,tokens=torch.stack(views),normal_extra=extras,
                         extra_indices=extra_indices,sampled_indices=indices,content=support))
        print(json.dumps(dict(rank=ctx.rank,id=raw['ids'][i],model=a.model,extra_normal_windows=len(extras))),flush=True)
    pieces = [None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1: dist.gather_object(rows,pieces,dst=0)
    else: pieces=[rows]
    if ctx.is_main:
        rows=sorted([x for part in pieces for x in part],key=lambda r:r['index'])
        if [x['index'] for x in rows]!=list(range(len(raw['ids']))):raise RuntimeError('Repeated/missing sources')
        maximum=raw['valid'].shape[1]
        bank=torch.zeros(len(rows),2,maximum,side*side,1024,dtype=torch.float16)
        for i,r in enumerate(rows):
            value=r.pop('tokens')
            bank[i,:,:value.shape[1]]=value
        data=dict(tokens=bank,rows=rows,ids=raw['ids'],labels=raw['labels'],valid=raw['valid'],
            timestamps=raw['timestamps'],content=torch.stack([r['content'] for r in rows]),
            train_indices=torch.tensor(train),validation_indices=torch.tensor(val),
            temporal_context_policy=raw['temporal_context_policy'],views=['identity','mirror'],
            metadata=dict(**provenance,resolution=a.size,grid=[side,side],normalization=transform.normalization,
                precision='BF16 frozen encoder, FP16 patch cache',world_size=ctx.world_size,rank_source_counts=counts,
                source_sha256=raw['video_sha256'],teacher_cache_sha256=file_sha256(DEFAULT_TEACHER),
                code_sha256=file_sha256(__file__),encoder_optimizer_updates=0,
                extra_reference_policy='Only training-clean sources, complete start/end windows; final observed tubelet',
                manual_annotations=0,source_validation_is_selection=True,independent_scene_test=False))
        a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(data,a.output)
        (a.output.with_suffix('.json')).write_text(json.dumps(dict(**data['metadata'],shape=list(bank.shape),
            cache_sha256=file_sha256(a.output)),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
