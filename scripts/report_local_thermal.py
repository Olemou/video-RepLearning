#!/usr/bin/env python3
"""Native-cell local-detector gallery; no per-image renormalization or blur."""
import argparse,html,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/tmp/spill_plot_portable']
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from vision_model.models.dense_encoder import file_sha256
from vision_model.interpretability.map_geometry import nonnegative_evidence_colors,blend_rgb
from vision_model.training.local_thermal import pair_masks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--targets-only',action='store_true');p.add_argument('--relative-pairs',action='store_true')
    p.add_argument('--joint-case',default='joint');p.add_argument('--control-case',default='control');p.add_argument('--output-name')
    a=p.parse_args();torch.set_num_threads(4)
    c=torch.load(a.root/'prepared/cache.pt',map_location='cpu',mmap=True,weights_only=False)
    if a.relative_pairs:c['pairs']=torch.load(a.root/'relative_pairs.pt',map_location='cpu',weights_only=False)['pairs']
    out=a.root/(a.output_name or (('relative_target_review' if a.relative_pairs else 'target_review') if a.targets_only else 'figures'));out.mkdir(exist_ok=False)
    learned=None;control=None;threshold=None;scale=2.
    if not a.targets_only:
        learned=torch.load(a.root/a.joint_case/'maps.pt',map_location='cpu',weights_only=False)
        control=torch.load(a.root/a.control_case/'maps.pt',map_location='cpu',weights_only=False)
        settings=json.loads((a.root/a.joint_case/'report.json').read_text())
        control_settings=json.loads((a.root/a.control_case/'report.json').read_text())
        threshold=[control_settings['calibration']['clean_q995'],settings['calibration']['clean_q995']]
        # Same normal-FPR policy for both outputs; pooled training-only magnitude
        # scale. A learned bias shift must not make the control artificially blue.
        train_ids=c['train_indices']
        calibration=torch.cat([(maps[train_ids,0]-cut).clamp_min(0).flatten()
            for maps,cut in zip([control,learned],threshold)])
        scale=max(float(torch.quantile(calibration,.999)),1e-6)
    train=set(c['train_indices'].tolist());records=[]
    order=[c['ids'].index(s) for s in ['cold_folderImage56','cold_folderImage10','cold_folderImage13','cold_folderImage14','clean_videoFrame121']]
    for i in order+[j for j in range(len(c['ids'])) if j not in order]:
        name=c['ids'][i];folder=out/name;folder.mkdir()
        gray=c['gray'][i,:,0].numpy();h,w=gray.shape[-2:]
        physical=c['features'][i,0,:,::2].float().clamp_min(0).mean(1)
        panels=3 if a.targets_only else 4
        board=Image.new('RGB',(panels*w,6*(h+24)+30),'white');d=ImageDraw.Draw(board)
        d.text((3,3),f'{name} | {"TRAIN" if i in train else "SOURCE SELECTION"} | local scores, not V-JEPA attention; targets unverified',fill='black')
        videos=[]
        for t in range(len(gray)):
            rgb=torch.from_numpy(np.repeat(gray[t,...,None],3,axis=-1).copy())
            union=torch.zeros(physical.shape[-2:]);ring=torch.zeros_like(union)
            for pair in c['pairs'][i][t]:
                rr,aa=pair_masks(pair,union.shape);union=torch.maximum(union,rr);ring=torch.maximum(ring,aa)
            target=torch.full_like(rgb,128)
            up=lambda x:F.interpolate(x[None,None].float(),size=(h,w),mode='nearest')[0,0]
            rr=up(union);aa=up(ring)
            target[aa>.5]=torch.tensor([0,0,255],dtype=torch.uint8);target[rr>.25]=torch.tensor([255,0,0],dtype=torch.uint8)
            target_panel=blend_rgb(rgb,target,.4)
            fields=([physical[t]] if a.targets_only else [(control[i,0,t]-threshold[0]).clamp_min(0),(learned[i,0,t]-threshold[1]).clamp_min(0)])
            images=[rgb.numpy()]
            for field in fields:images.append(blend_rgb(rgb,nonnegative_evidence_colors(up(field),scale=scale),.4).numpy())
            images.append(target_panel.numpy());videos.append(np.concatenate(images,1))
            if t%2:
                y=30+(t//2)*(h+24)
                for j,img in enumerate(images):
                    board.paste(Image.fromarray(img),(j*w,y+24))
                    title=(['Grayscale input','Untrained relative contrast','Within-frame pairs (unverified)'] if a.targets_only else
                           ['Grayscale input','Absence-only control','Pair-trained local evidence','Within-frame pairs (unverified)'])[j]
                    d.text((j*w+3,y+3),title,fill='black')
        board.save(folder/'comparison.png');Image.fromarray(videos[-1]).save(folder/'endpoint.png')
        if not a.targets_only:
            command=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{panels*w}x{h}',
                '-framerate','4','-i','-','-an','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(folder/'comparison.mp4')]
            subprocess.run(command,input=np.stack(videos).tobytes(),check=True)
        records.append(dict(id=name,split='train' if i in train else 'selection',image=f'{name}/comparison.png',
            endpoint=f'{name}/endpoint.png',video=None if a.targets_only else f'{name}/comparison.mp4',sampled_indices=c['rows'][i]['sampled_indices']))
    cards=''.join(f'<article><h3>{html.escape(r["id"])} — {r["split"]}</h3><a href="{r["image"]}"><img width="1200" src="{r["endpoint"]}"></a>'+
        (f'<p><a href="{r["video"]}">12-frame video</a></p>' if r['video'] else '')+'</article>' for r in records)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Local thermal audit</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}</style><h1>Finite-field thermal evidence</h1><p>Shared scale; native cells; unverified automatic region/ring comparisons. This is a separate local output, not frozen backbone attention or a liquid mask.</p>'+cards)
    manifest=dict(rows=records,targets_only=a.targets_only,shared_scale=scale,threshold=threshold,
        threshold_policy='Each respective model training-clean q99.5; no per-frame adjustment; common pooled training magnitude scale',
        cache_sha256=file_sha256(a.root/'prepared/cache.pt'),code_sha256=file_sha256(__file__),localization_validated=False)
    if not a.targets_only:
        manifest['checkpoints']={case:file_sha256(a.root/case/'checkpoint.pt') for case in [a.control_case,a.joint_case]}
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        h=json.loads((a.root/a.joint_case/'report.json').read_text())['history'];fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,key in zip(axes,['total','absence','pair']):
            ax.plot([r['update'] for r in h],[r[key] for r in h]);ax.set(xlabel='Optimizer update',ylabel=key);ax.grid(alpha=.2)
        fig.savefig(out/'loss_curves.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);ax.plot([r['update'] for r in h],[r['total'] for r in h]);ax.set(xlabel='Optimizer update',ylabel='Local absence + calibrated ranking loss');ax.grid(alpha=.2);fig.savefig(out/'single_total_loss.png',dpi=180);plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
