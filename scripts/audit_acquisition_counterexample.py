#!/usr/bin/env python3
"""Fixed-model source-excluded counterexample; no new training or region labels."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from attribute_experiment81 import load_case
from control_experiment81 import transform_bank
from prepare_experiment81 import normalize_bank,digest
from vision_model.data.video_dataset import VideoManifestDataset,_ffmpeg_frames
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.online_trajectory import OnlinePrefixEncoder,collect_prefix_features


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['frozen','full'])
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    device=torch.device('cuda',0)
    out=a.experiment/'acquisition_counterexample'
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    out.mkdir(exist_ok=True)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    ds=VideoManifestDataset('/home/olemou/videos/manifest.jsonl',sampling=TemporalSamplingConfig(clip_length=12,target_fps=4,training=False))
    record=next(r for r in ds.records if r.video_id=='clean_folderImage81')
    source=ds.dataset_root/record.video_path
    source_sha=digest(source)
    if record.video_id in c['ids'] or source_sha in c['source_sha256']:
        raise ValueError('Counterexample overlaps the 80 training/selection sources')
    offsets=np.arange(12)*record.fps/4
    max_start=record.num_frames-1-offsets[-1]
    windows={name:np.rint(offsets+start).astype(np.int64) for name,start in
        (('beginning',0.),('center',max_start/2),('end',max_start))}
    decoded={}
    for name,indices in windows.items():
        rgb,fps,count=_ffmpeg_frames(source,record,indices,threads=1)
        if fps!=record.fps or count!=record.num_frames:raise RuntimeError('Source metadata changed')
        decoded[name]=rgb
        canvas=Image.new('RGB',(960,530),'white');draw=ImageDraw.Draw(canvas)
        draw.text((3,3),f'{record.video_id} | manifest label CLEAN | {name} window; not a spatial annotation',fill='black',font=ImageFont.load_default(size=15))
        for slot,t in enumerate(range(1,12,2)):
            x,y=slot%3*320,30+slot//3*250
            canvas.paste(Image.fromarray(rgb[t].permute(1,2,0).numpy()).resize((320,240)),(x,y))
            draw.text((x+3,y+240),f'frame {indices[t]}',fill='black')
        canvas.save(out/f'{name}_input.png')
    rows=[]
    for case in a.cases:
        encoder,model,reps,proto,settings=load_case(a.baseline,case,device)
        transform=OfficialVJEPA2Transform(settings['teacher_metadata']['encoder_dir'],preserve_field_of_view=True)
        wrapper=OnlinePrefixEncoder(encoder)
        for name,rgb in decoded.items():
            px=transform_bank(rgb,[],transform,include_augmentations=False)
            peer=normalize_bank(c['pixels'][0,:2],c['normalization'])
            features=collect_prefix_features(wrapper,torch.stack((peer,px)),torch.ones(2,6,dtype=torch.bool),device,chunk_sources=2)
            peer_error=float((features[0].cpu()-reps['features'][0]).abs().max())
            if peer_error>1e-6:raise RuntimeError(f'Known-source production replay changed: {peer_error}')
            # Preserve the original selection-wide trajectory batch size. Sources do not interact.
            selected=reps['selection_indices']
            batch=reps['features'][selected].to(device).clone();batch[0]=features[1]
            valid=reps['valid'][selected].to(device).clone();valid[0]=True
            clocks=reps['timestamps'][selected].to(device).clone()
            clocks[0]=torch.tensor(windows[name][1::2]/record.fps,device=device)
            clock_scores={}
            for mode in ('actual','relative_zero'):
                modified=clocks.clone()
                if mode=='relative_zero':modified[0]-=modified[0,0].clone()
                n,v,t,d=batch.shape
                pred=model(batch.flatten(0,1),valid[:,None].expand(n,v,t).flatten(0,1),
                    modified[:,None].expand(n,v,t).flatten(0,1),detach_teacher=False)
                similarities=pred['embedding'][0]@proto.T
                margin=float(similarities[1]-similarities[0])
                clock_scores[mode]=dict(spill_minus_clean_margin=margin,prediction=int(margin>=0),
                    correct_for_manifest_clean_label=margin<0,class_cosines=similarities.cpu().tolist())
            row=dict(case=case,window=name,frame_indices=windows[name].tolist(),known_peer_replay_error=peer_error,
                clock_controls=clock_scores,checkpoint_sha256=digest(a.baseline/case/'checkpoint.pt'))
            rows.append(row);print(json.dumps(row),flush=True)
        del encoder,model,wrapper
        torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(dict(source=record.video_id,label_from_manifest=record.label,
        source_sha256=source_sha,excluded_from_current80=True,native_resolution=[record.height,record.width],
        windows_are_one_source_not_independent_samples=True,physical_scene_match_verified=False,
        label_independently_verified=False,selection_policy='Only clean manifest source sharing cold-cohort resolution',
        previous_experiment_use_unknown=True,training_updates=0,manual_spatial_annotations=0,results=rows),indent=2)+'\n')


if __name__=='__main__':main()
