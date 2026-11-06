#!/usr/bin/env python3
"""Measured joint losses, t-SNE and actual-frame auxiliary/decision comparisons."""
import argparse,html,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),'/tmp/spill_plot_portable']
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from vision_model.interpretability.map_geometry import inverse_letterbox_map,nonnegative_evidence_colors,blend_rgb
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from train_spill_localization import DEFAULT_TEACHER
from diagnose_spill_representation import digest
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);out=a.root/'automatic_figures';out.mkdir(exist_ok=False)
    maps=torch.load(a.root/'automatic_maps/maps.pt',map_location='cpu',weights_only=False)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    probe=torch.load(a.root/'automatic_probe/representations.pt',map_location='cpu',weights_only=False)
    cases=('automatic_control','automatic_joint');reports={};reps={};histories={}
    for case in cases:
        reports[case]=json.loads((a.root/case/'report.json').read_text());histories[case]=reports[case]['evaluations']
        reps[case]=torch.load(a.root/case/'representations.pt',map_location='cpu',weights_only=False)
    train=maps['train_indices'];train_set=set(train.tolist());support=maps['content'].flatten(1)
    gradient_scale=max(float(r['maps']['positive_gradient'][:,support[r['index']]].max()) for r in maps['rows'] if r['index'] in train_set)
    if not gradient_scale>0:raise ValueError('No differentiable trajectory gradient to display')
    lookup={(r['case'],r['index']):r for r in maps['rows']};diagnostics={}
    for case in cases:
        values=[]
        for i in range(len(raw['ids'])):
            r=lookup[case,i];local=r['local_probabilities'][:,support[i]];gradient=r['maps']['positive_gradient'][:,support[i]]
            p=gradient.sum(0);p=p/p.sum().clamp_min(1e-20)
            values.append(dict(id=r['id'],label=r['label'],split='train' if i in train_set else 'validation_selection',
                local_score_above_half_fraction=float((local>.5).float().mean()),local_score_mean=float(local.mean()),
                decision_gradient_spatial_entropy=float(-(p*p.clamp_min(1e-20).log()).sum()/np.log(len(p))),
                embedding_replay_error=r['embedding_replay_error'],local_replay_error=r['local_replay_error']))
        diagnostics[case]=values
    comparison=[]
    for i in range(len(raw['ids'])):
        left=lookup[cases[0],i]['maps']['positive_gradient'].flatten();right=lookup[cases[1],i]['maps']['positive_gradient'].flatten()
        comparison.append(dict(id=raw['ids'][i],label=int(raw['labels'][i]),decision_map_cosine=float(F.cosine_similarity(left,right,dim=0)),
            local_probability_difference=float((lookup[cases[0],i]['local_probabilities']-lookup[cases[1],i]['local_probabilities']).abs().mean())))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for case in cases:
        ck=torch.load(a.root/case/'checkpoint.pt',map_location='cpu',weights_only=False);h=ck['history'];weight=ck['metadata']['fg_weight']
        for ax,key in zip(axes[0],('loss','foreground_background_loss','trajectory_loss')):
            y=[v['loss']-weight*v['foreground_background_loss'] if key=='trajectory_loss' else v[key] for v in h]
            ax.plot([v['update'] for v in h],y,label=case);ax.set_title('pre-update '+key)
        for ax,key in zip(axes[1],('semantic_loss','prediction_loss','retention_loss')):
            ax.plot([v['update'] for v in histories[case]],[v['validation']['trajectory_losses'][key] for v in histories[case]],label=case);ax.set_title('fixed-curriculum validation '+key)
    axes[1,1].axhline(histories[cases[0]][0]['validation']['trajectory_losses']['persistence_loss'],ls='--',c='gray',label='frozen persistence')
    for ax in axes.flat:ax.set_xlabel('optimizer updates');ax.legend(fontsize=8)
    fig.suptitle('Matched 100-update shared-suffix pilot; local loss uses unverified automatic candidates')
    fig.tight_layout();fig.savefig(out/'joint_loss_curves.png',dpi=160);fig.savefig(out/'joint_loss_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(20,7))
    for row,(split,label) in enumerate((('train','training fit'),('validation','source validation selection'))):
        for col,(key,w) in enumerate((('total_loss',1),('semantic_loss',1),('alignment_loss',.1),('prediction_loss',1),('retention_loss',.1))):
            ax=axes[row,col]
            for case in cases:ax.plot([v['update'] for v in histories[case]],[w*v[split]['trajectory_losses'][key] for v in histories[case]],label=case)
            ax.set_title(f'{label}: {w:g} x {key}');ax.set_xlabel('updates');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'all_trajectory_components.png',dpi=160);fig.savefig(out/'all_trajectory_components.pdf');plt.close(fig)
    from sklearn.manifold import TSNE
    fig,axes=plt.subplots(1,2,figsize=(13,5));coordinates={}
    for ax,case in zip(axes,cases):
        z=reps[case]['embeddings'].numpy();xy=TSNE(n_components=2,perplexity=15,random_state=239,init='pca',learning_rate='auto').fit_transform(z);coordinates[case]=xy.tolist()
        for ids,marker,label in ((maps['train_indices'],'o','train'),(maps['validation_indices'],'^','validation selection')):
            for y,color in ((0,'royalblue'),(1,'tomato')):
                idx=ids[raw['labels'][ids]==y];ax.scatter(xy[idx,0],xy[idx,1],marker=marker,color=color,label=f'{label}: {"clean" if y==0 else "cold"}')
        ax.set_title(case);ax.legend(fontsize=8)
    fig.suptitle('t-SNE is a representation diagnostic, not spill localization');fig.tight_layout();fig.savefig(out/'joint_tsne.png',dpi=160);plt.close(fig)
    (out/'joint_tsne.json').write_text(json.dumps(dict(ids=raw['ids'],coordinates=coordinates,seed=239,perplexity=15),indent=2)+'\n')
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    sampler=TemporalSampler(TemporalSamplingConfig(**raw['sampling']));gallery=[]
    for i,name in enumerate(raw['ids']):
        r=records[name];source=Path(raw['dataset_root'])/r['video_path'];n=int(raw['valid'][i].sum())
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        sampled,_=sampler.sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps']);indices=sampled[:2*n].tolist()
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Frame indexing mismatch')
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        folder=out/name;folder.mkdir();canvas=Image.new('RGB',(1280,45+n*215),'white');draw=ImageDraw.Draw(canvas);draw.text((4,3),f'{name} | {"TRAIN" if i in train_set else "VALIDATION SELECTION"} | no verified spill masks',fill='black')
        titles=('Original','Frozen-probe local score','Control decision gradient','Joint local score','Joint decision gradient')
        for col,title in enumerate(titles):draw.text((col*256+3,24),title,fill='black')
        movie=[]
        for slot,index in enumerate(indices):
            frame=torch.from_numpy(np.asarray(Image.fromarray(decoded[index]).resize((256,192))).copy());t=slot//2
            fields=[(probe['local_logits'][i,0,t].sigmoid(),1.),(lookup[cases[0],i]['maps']['positive_gradient'][t],gradient_scale),
                (lookup[cases[1],i]['local_probabilities'][t],1.),(lookup[cases[1],i]['maps']['positive_gradient'][t],gradient_scale)]
            panels=[frame.numpy()]
            for value,scale in fields:
                field=inverse_letterbox_map(value.reshape(16,16),192,256,interpolation='nearest')
                panels.append(blend_rgb(frame,nonnegative_evidence_colors(field,scale=scale),.4).numpy())
            movie.append(np.concatenate(panels,1))
            if slot%2:
                for col,panel in enumerate(panels):canvas.paste(Image.fromarray(panel),(col*256,45+t*215))
                draw.text((4,45+t*215+195),f'frame {index}; {index/r["fps"]:.3f}s; local [0,1] uncalibrated, gradient shared scale {gradient_scale:.5g}',fill='black')
        arr=np.stack(movie);video=folder/'comparison.mp4'
        command=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x192','-r','4','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)]
        proc=subprocess.run(command,input=arr.tobytes(),capture_output=True)
        if proc.returncode:raise RuntimeError(proc.stderr.decode())
        canvas.save(folder/'comparison.png');Image.fromarray(arr[-1]).save(folder/'last_frame.png')
        gallery.append(dict(id=name,label=int(raw['labels'][i]),split='train' if i in train_set else 'validation_selection',sampled_indices=indices,
            source_sha256=raw['video_sha256'][i],video=str(video.relative_to(out)),figure=str((folder/'comparison.png').relative_to(out))))
    page=['<!doctype html><meta charset="utf-8"><title>Automatic spatial pilot</title>',
        '<h1>Automatic candidates: auxiliary scores versus actual trajectory gradients</h1>',
        '<p>80 real videos, no manual spatial annotations, no decoder or MIL. Local scores are trained on unverified candidates; gradients explain the actual trained trajectory margin conditional on cached pre23. Neither is a verified spill mask.</p>',
        '<p>MP4 order: original | frozen-probe local | trajectory-only control gradient | joint local | joint gradient. Local scale [0,1]; one training-derived maximum gradient scale shared by all sources and both models. No per-frame renormalization or smoothing.</p>',
        '<img width="1000" src="joint_loss_curves.png"><img width="1000" src="all_trajectory_components.png"><img width="1000" src="joint_tsne.png">']
    for r in gallery:page.append(f'<h2>{html.escape(r["id"])} — {r["split"]}</h2><video width="1200" controls preload="none" src="{r["video"]}"></video><p><a href="{r["figure"]}">All six endpoint comparisons</a></p>')
    (out/'index.html').write_text('\n'.join(page));(out/'diagnostics.json').write_text(json.dumps(dict(cases=diagnostics,comparison=comparison),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(rows=gallery,gradient_scale=gradient_scale,gradient_scale_fit='Maximum across both arms, training sources only',
        local_scale=[0,1],map_sha256=digest(a.root/'automatic_maps/maps.pt'),checkpoint_sha256={case:digest(a.root/case/'checkpoint.pt') for case in cases},
        code_sha256=digest(Path(__file__)),localization_validated=False),indent=2)+'\n')
    print(json.dumps(dict(real_sources=len(gallery),videos=len(gallery),output=str(out))))


if __name__=='__main__':main()
