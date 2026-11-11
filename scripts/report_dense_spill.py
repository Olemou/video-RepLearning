#!/usr/bin/env python3
"""Real-frame galleries and curves for the decision-connected dense pilots."""
import argparse,html,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),'/tmp/spill_plot_portable']
import numpy as np
import torch
from PIL import Image,ImageDraw
from vision_model.interpretability.map_geometry import inverse_letterbox_map,nonnegative_evidence_colors,blend_rgb
from vision_model.models.dense_encoder import file_sha256
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--joint-case',default='dense_joint');p.add_argument('--candidate-dir',default='vjepa21_consensus')
    p.add_argument('--control-case',default='dense_control')
    p.add_argument('--output-name',default='figures');a=p.parse_args()
    torch.set_num_threads(4);out=a.root/a.output_name;out.mkdir(exist_ok=False)
    c=torch.load(a.root/'vjepa21_384.pt',map_location='cpu',mmap=True,weights_only=False)
    pseudo=torch.load(a.root/a.candidate_dir/'candidates.pt',map_location='cpu',weights_only=False)
    jointcase=a.joint_case
    controlcase=a.control_case
    cases=[controlcase,jointcase];reps={};checkpoints={}
    for case in cases:
        reps[case]=torch.load(a.root/case/'representations.pt',map_location='cpu',weights_only=False)
        checkpoints[case]=torch.load(a.root/case/'checkpoint.pt',map_location='cpu',weights_only=False)
    train=set(c['train_indices'].tolist());support=c['content'].flatten(1);size=c['metadata']['resolution'];side=size//16
    # Same training-only magnitude scale for positive and negative panels.
    calibration=torch.cat([reps[case]['rows'][i][1]['contributions'][0,:,support[i]].abs().flatten()
                           for case in cases for i in sorted(train)])
    scale=float(torch.quantile(calibration,.999));assert scale>0
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    rows=[];stats={case:[] for case in cases}
    first=[c['ids'].index(name) for name in ['cold_folderImage10','cold_folderImage13','clean_videoFrame121','clean_videoFrame124'] if name in c['ids']]
    for i in first+[i for i in range(len(c['ids'])) if i not in first]:
        name=c['ids'][i]
        r=records[name];path=Path(raw['dataset_root'])/r['video_path']
        if file_sha256(path)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        indices=c['rows'][i]['sampled_indices'];n=int(c['valid'][i].sum())
        decoded=decode_selected_frames(path,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        folder=out/name;folder.mkdir()
        canvas=Image.new('RGB',(5*384,52+n*312),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,3),f'{name} | {"TRAINING" if i in train else "SOURCE SELECTION"} | signed decision contributions, not masks',fill='black')
        titles=['Original','Control: spill evidence','Joint: spill evidence','Joint: clean evidence','Automatic candidates (unverified)']
        for j,title in enumerate(titles):draw.text((384*j+3,27),title,fill='black')
        frames=[]
        for slot,index in enumerate(indices):
            t=slot//2;rgb=torch.from_numpy(np.asarray(Image.fromarray(decoded[index]).resize((384,288))).copy())
            control=reps[controlcase]['rows'][i][1]['contributions'][0,t]
            joint=reps[jointcase]['rows'][i][1]['contributions'][0,t]
            panels=[rgb.numpy()]
            for field in (control.clamp_min(0),joint.clamp_min(0),(-joint).clamp_min(0)):
                value=inverse_letterbox_map(field.reshape(side,side),288,384,size=size,interpolation='nearest')
                panels.append(blend_rgb(rgb,nonnegative_evidence_colors(value,scale=scale),.4).numpy())
            target=pseudo['targets'][i,0,t].reshape(side,side)
            # Map categorical values by nearest sampling, preserving unknowns.
            category=inverse_letterbox_map(target.float(),288,384,size=size,interpolation='nearest')
            color=torch.full_like(rgb,128);color[category==1]=torch.tensor([255,0,0],dtype=torch.uint8)
            color[category==0]=torch.tensor([0,0,255],dtype=torch.uint8)
            panels.append(blend_rgb(rgb,color,.4).numpy())
            frames.append(np.concatenate(panels,1))
            if slot%2:
                for j,panel in enumerate(panels):canvas.paste(Image.fromarray(panel),(j*384,52+t*312))
        canvas.save(folder/'comparison.png')
        Image.fromarray(frames[-1]).save(folder/'endpoint.png')
        command=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x288','-framerate','4','-i','-',
                 '-an','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(folder/'comparison.mp4')]
        subprocess.run(command,input=np.stack(frames).tobytes(),check=True)
        for case in cases:
            result=reps[case]['rows'][i][1];field=result['contributions'][0,:n,support[i]]
            positive=field.clamp_min(0).sum(0);prob=positive/positive.sum().clamp_min(1e-20)
            stats[case].append(dict(id=name,label=int(c['labels'][i]),split='train' if i in train else 'selection',
                positive_fraction=float((field>0).float().mean()),
                positive_spatial_entropy=float(-(prob*prob.clamp_min(1e-20).log()).sum()/np.log(len(prob))) if positive.sum()>0 else None,
                margin=float(result['margin'][0])))
        rows.append(dict(id=name,split='train' if i in train else 'selection',label=int(c['labels'][i]),sampled_indices=indices,
            image=f'{name}/comparison.png',endpoint=f'{name}/endpoint.png',video=f'{name}/comparison.mp4',frames=len(frames)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for case in cases:
        history=checkpoints[case]['history']
        for ax,key in zip(axes,['total_loss','semantic_loss','weighted_local_loss']):
            ax.plot([r['update'] for r in history],[r[key] for r in history],label=case)
            ax.set(xlabel='Optimizer update',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Dense readout losses; original four-loss trajectory checkpoint preserved')
    fig.savefig(out/'loss_curves.png',dpi=180);fig.savefig(out/'loss_curves.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);history=checkpoints[jointcase]['history']
    ax.plot([r['update'] for r in history],[r['total_loss'] for r in history],color='royalblue')
    ax.set(title='Dense joint run: total training loss',xlabel='Optimizer update',ylabel='Total loss');ax.grid(alpha=.2)
    fig.savefig(out/'single_total_loss.png',dpi=180);plt.close(fig)
    from sklearn.manifold import TSNE
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,case in zip(axes,cases):
        xy=TSNE(n_components=2,perplexity=15,random_state=239,init='pca',learning_rate='auto').fit_transform(reps[case]['embeddings'].numpy())
        for ids,marker,label in [(c['train_indices'],'o','train'),(c['validation_indices'],'^','selection')]:
            for y,color in [(0,'royalblue'),(1,'tomato')]:
                select=ids[c['labels'][ids]==y];ax.scatter(xy[select,0],xy[select,1],c=color,marker=marker,label=f'{label}: {"clean" if y==0 else "cold"}')
        ax.set_title(case);ax.legend(fontsize=8)
    fig.suptitle('t-SNE does not validate liquid localization');fig.savefig(out/'tsne.png',dpi=180);plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(rows=rows,shared_scale=scale,scale_fit='Training-only absolute contribution q99.9, both arms',
        cases={case:file_sha256(a.root/case/'checkpoint.pt') for case in cases},statistics=stats,
        code_sha256=file_sha256(__file__),localization_validated=False),indent=2)+'\n')
    cards=''.join(f'<article><h3>{html.escape(r["id"])} — {r["split"]}</h3><video controls preload="none" width="960" src="{r["video"]}"></video><p><a href="{r["image"]}">All six tubelets</a> · <a href="{r["endpoint"]}">Last sampled frame</a></p></article>' for r in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense spill readout audit</title><style>body{font:16px sans-serif;margin:24px}article{margin:32px 0}img,video{max-width:100%}</style><h1>80-source dense readout comparison</h1><p>Positive/negative contributions to actual dense episode decisions. Blue means low numerical evidence, not verified background. Automatic candidates are unverified. The original trajectory output is preserved separately.</p><p><a href="loss_curves.png">Loss curves</a> · <a href="single_total_loss.png">Single total loss</a> · <a href="tsne.png">t-SNE</a></p>'+cards)


if __name__=='__main__':main()
