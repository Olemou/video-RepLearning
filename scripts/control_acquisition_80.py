#!/usr/bin/env python3
"""Class/acquisition confounds and label-blind resolution equalization control."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from vision_model.models.encoder import load_pretrained_encoder
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from train_final_attention import make_student
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);ctx=initialize_distributed()
    c=torch.load(a.cache,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    cases={}
    for name,folder in [('FT0',a.experiment/'matched_100/FT0_frozen'),('FT1_3em6',a.experiment/'matched_100/FT1_attention'),('FT1_1em4',a.experiment/'attention_lr1em4_100/FT1_attention')]:
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        head={k.removeprefix('projector.'):v for k,v in ck['model'].items() if k.startswith('projector.')}
        model=make_student(c,False,head).to(ctx.device).eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
        saved=torch.load(folder/'representations.pt',map_location='cpu',weights_only=False)
        cases[name]=(model,saved,saved['prototypes'].to(ctx.device))
    loaded=load_pretrained_encoder('/home/olemou/hf',local_files_only=True)
    encoder=loaded.encoder.to(device=ctx.device,dtype=torch.bfloat16).eval().requires_grad_(False)
    transform=OfficialVJEPA2Transform('/home/olemou/hf',preserve_field_of_view=True)
    val=c['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size))
    _,counts=audit_distributed_indices(positions,len(val));rows=[]
    for position in positions:
        i=val[position];r=records[c['ids'][i]];source=Path(raw['dataset_root'])/r['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        indices=c['rows'][i]['sampled_frame_indices'];n=len(indices)//2
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=torch.stack([torch.from_numpy(decoded[j]).permute(2,0,1) for j in indices])
        # Identical pre-resize dimensions for BOTH classes; no label-based policy.
        resized=F.interpolate(frames.float(),size=(288,384),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).to(torch.uint8)
        pixels=transform(resized);prefixes=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for end in range(1,n+1):
                value=encoder.patch_embed(pixels[:2*end].permute(1,0,2,3)[None].to(device=ctx.device,dtype=torch.bfloat16))
                for block in encoder.blocks[:-1]:value=block(value,frames=end,height=16,width=16)
                prefixes.append(value)
        scores={}
        for name,(model,saved,prototypes) in cases.items():
            with torch.no_grad():
                support=saved['content'][i].to(ctx.device)
                q=torch.stack([model(h,frames=end+1)[0] for end,h in enumerate(prefixes)])[:,support]
                z=F.normalize(q.mean((0,1)),dim=-1);s=z@prototypes.T
                baseline=saved['embeddings'][i,0].to(ctx.device)@prototypes.T
                scores[name]=dict(original_margin=float(baseline[1]-baseline[0]),equalized_margin=float(s[1]-s[0]),
                    original_prediction=int(baseline.argmax()),equalized_prediction=int(s.argmax()))
        rows.append(dict(id=c['ids'][i],label=int(c['labels'][i]),native_resolution=[r['width'],r['height']],scores=scores,
            unique_decoded_sampled_frames=len({frames[j].numpy().tobytes() for j in range(len(frames))}),
            successive_sampled_frame_mean_absolute_change=float((frames[1:].float()-frames[:-1].float()).abs().mean())))
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
    else:gathered=[rows]
    if ctx.is_main:
        rows=sorted([r for part in gathered for r in part],key=lambda r:r['id'])
        if len({r['id'] for r in rows})!=len(val):raise RuntimeError('Repeated validation source')
        descriptors=np.array([[records[name]['width'],records[name]['height']] for name in c['ids']])
        train=c['train_indices'];labels=c['labels'].numpy()
        proxy=make_pipeline(StandardScaler(),LogisticRegression(random_state=239)).fit(descriptors[train],labels[train])
        summary={name:dict(equalized_accuracy=float(np.mean([r['scores'][name]['equalized_prediction']==r['label'] for r in rows])),
            mean_absolute_margin_change=float(np.mean([abs(r['scores'][name]['equalized_margin']-r['scores'][name]['original_margin']) for r in rows]))) for name in cases}
        native=Counter((records[name]['category'],records[name]['width'],records[name]['height']) for name in c['ids'])
        result=dict(rows=rows,world_size=ctx.world_size,rank_source_counts=counts,summary=summary,
            resolution_by_class=[dict(category=k[0],width=k[1],height=k[2],n=v) for k,v in native.items()],
            source_resolution_only_validation_accuracy=float((proxy.predict(descriptors[val])==labels[val]).mean()),
            protocol='All 16 source-validation videos; resize complete frames to 384x288 with antialiasing, then unchanged official 256 letterbox transform; independently encode every causal prefix',
            limitation='Native resolution metadata is not given explicitly to the encoder. Equalization tests one resize intervention; retained accuracy does not remove acquisition, scene, thermal-calibration or compression confounds.')
        destination=a.experiment/'acquisition_controls.json'
        if destination.exists():raise FileExistsError(destination)
        destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
