#!/usr/bin/env python3
"""Render matched readout sensitivities and frozen feature-similarity diagnostics.

No mask scores or favorable-head selection. All maps retain their signed
values; positive sensitivities share one scale across readouts and frames.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from ablate_spill_readouts import PatchReadout
from diagnose_spill_spatial import heat_panel,LAYERS
from visualize_trajectory_video import decode_selected_frames


def render(frames,maps,labels,path,title,*,probability=False):
    maximum=1. if probability else max(float(maps.abs().max()),1e-30)
    canvas=Image.new('RGB',(256*len(frames),225*len(labels)+45),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((4,4),title,fill='black')
    draw.text((4,23),f'Shared scale: {maximum:.5g}; no localization ground truth',fill='black')
    for row,label in enumerate(labels):
        for t,frame in enumerate(frames):
            x=maps[row,t].numpy()/maximum
            if probability:
                rgb=np.stack((x,np.zeros_like(x),1-x),-1)
            else:
                rgb=np.stack((np.maximum(x,0),1-np.abs(x),np.maximum(-x,0)),-1)
            color=Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8)).resize((256,256))
            h,w=frame.shape[:2]
            scale=min(256/h,256/w)
            rh,rw=round(h*scale),round(w*scale)
            top,left=(256-rh)//2,(256-rw)//2
            color=color.crop((left,top,left+rw,top+rh)).resize((256,192))
            source=Image.fromarray(frame).resize((256,192))
            canvas.paste(Image.blend(source,color,.5),(t*256,row*225+45))
            draw.text((t*256+2,row*225+240),f'{label} | t={t}',fill='black')
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(239)
    folder=a.diagnosis/'readout_ablations'
    c=torch.load(a.diagnosis/'spatial_matched_precision/spatial_tokens.pt',weights_only=False,map_location='cpu',mmap=True)
    raw=torch.load(a.diagnosis.parent.parent/'trajectory_80_sources_12f_lrsearch_20261021/causal_features.pt',map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    selected=['clean_videoFrame121','clean_videoFrame124','cold_folderImage10','cold_folderImage13']
    report={}
    # Whole-video class prototypes fitted on training sources; this is a
    # feature-similarity diagnostic, not positive patch labeling.
    train=c['train_indices']
    proto=[]
    for slot,layer in enumerate(LAYERS):
        means=c['tokens'][train,0,:,slot].float().mean((1,2))
        proto.append(torch.stack([F.normalize(means[c['labels'][train]==label].mean(0),dim=0) for label in (0,1)]))
    results=json.loads((folder/'results.json').read_text())['results']
    for video_id in selected:
        i=c['ids'].index(video_id)
        r=records[video_id]
        ends=raw['frame_indices'][i][raw['valid'][i]].tolist()
        decoded=decode_selected_frames(Path(raw['dataset_root'])/r['video_path'],ends,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        frames=[decoded[j] for j in ends]
        similarity=[]
        for slot,layer in enumerate(LAYERS):
            patch=F.normalize(c['tokens'][i,0,:,slot].float(),dim=-1)
            similarity.append((patch@(proto[slot][1]-proto[slot][0])).reshape(6,16,16))
        render(frames,torch.stack(similarity),[f'block {l}' for l in LAYERS],folder/f'{video_id}_feature_similarity.png','Frozen patches versus training whole-video class prototypes; signed cosine difference')
        gradients,probabilities,names,mil_names=[],[],[],[]
        statistics=[]
        for case in results:
            ck=torch.load(folder/(case['case']+'.pt'),map_location='cpu',weights_only=False)
            model=PatchReadout(c['norm_state'],mil=case['mil'],adapter=case['adapter']).eval()
            model.load_state_dict(ck['model'],strict=True)
            bank=(c['final_raw'][i,0] if case['adapter'] else c['tokens'][i,0,:,c['layers'].index(case['layer'])]).float()[None].clone().requires_grad_(True)
            # These four sources have full 4:3 content patches in rows 2..13.
            valid=torch.zeros(1,6,16,16,dtype=torch.bool)
            valid[:,:,2:14]=True
            valid=valid.flatten(2)
            embedded,logits,bag,weights=model(bank,valid)
            z=torch.stack([row['embedding'][0] for row in ck['rows']])
            prototypes=torch.stack([F.normalize(z[train][c['labels'][train]==label].mean(0),dim=0) for label in (0,1)])
            margin=embedded[0]@(prototypes[1]-prototypes[0])
            grad=torch.autograd.grad(margin,bank)[0]
            saliency=(grad*bank).sum(-1).detach()[0].reshape(6,16,16)
            gradients.append(saliency)
            names.append(case['case'])
            stats=dict(case=case['case'],margin=float(margin.detach()),positive_mass=float(saliency.clamp_min(0).sum()),negative_mass=float((-saliency).clamp_min(0).sum()))
            if case['mil']:
                probability=logits.detach()[0].sigmoid().reshape(6,16,16)
                probabilities.append(probability)
                mil_names.append(case['case'])
                content=probability[:,2:14]
                stats.update(mean_content_patch_score=float(content.mean()),maximum_patch_score=float(content.max()),fraction_content_above_half=float((content>.5).float().mean()),bag_logit=float(bag.detach()[0]))
            statistics.append(stats)
        render(frames,torch.stack(gradients),names,folder/f'{video_id}_readout_gradient_times_token.png','Readout margin: gradient times contextual patch token; not input attribution')
        render(frames,torch.stack(probabilities),mil_names,folder/f'{video_id}_MIL_scores.png','MIL patch sigmoid scores; contextual, uncalibrated; not segmentation',probability=True)
        torch.save(dict(gradient_times_token=torch.stack(gradients),MIL_scores=torch.stack(probabilities),feature_similarity=torch.stack(similarity)),folder/f'{video_id}_probe_maps.pt')
        report[video_id]=statistics
    (folder/'map_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
