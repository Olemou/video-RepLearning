#!/usr/bin/env python3
"""Export every normal-reference case on real frames, without map smoothing."""
import argparse,csv,html,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),'/tmp/spill_plot_portable']
import numpy as np
import torch
from PIL import Image,ImageDraw
from vision_model.interpretability.map_geometry import inverse_letterbox_map,nonnegative_evidence_colors,blend_rgb
from train_spatial_semantic import DEFAULT_TEACHER,EXPERIMENT
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest,line_figure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--student',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    torch.set_num_threads(4)
    result=json.loads((a.run/'results.json').read_text());maps=torch.load(a.run/'maps.pt',map_location='cpu',weights_only=False)
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    out=a.output or a.run/'figures';out.mkdir(exist_ok=False)
    student_histories={}
    if a.student:
        for arm in ('normal_cosine','normal_contrast'):
            extra=torch.load(a.student/arm/'maps.pt',map_location='cpu',weights_only=False)
            if extra['ids']!=maps['ids']:raise ValueError('Student source mismatch')
            maps['maps'][arm]=extra['maps']
            evaluation=json.loads((a.student/arm/'evaluation_history.json').read_text())[-1]
            result['cases'][arm]=dict(calibration={'q'+key:value for key,value in evaluation['calibration'].items()})
            student_histories[arm]=json.loads((a.student/arm/'history.json').read_text())
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    main_cases=['block5_raw','block17_raw','normal_cosine','normal_contrast'] if a.student else ['block5_raw','block17_raw','block23_raw'];gallery=[]
    main_cases=[case for case in main_cases if case in result['cases']]
    if not main_cases:main_cases=list(result['cases'])
    from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
    sampler=TemporalSampler(TemporalSamplingConfig(**raw['sampling']))
    for i,name in enumerate(raw['ids']):
        r=records[name];source=Path(raw['dataset_root'])/r['video_path'];n=int(raw['valid'][i].sum())
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source hash differs')
        sampled,_=sampler.sample_with_validity(num_frames=r['num_frames'],source_fps=r['fps']);indices=sampled[:2*n].tolist()
        if indices[1::2]!=raw['frame_indices'][i,:n].tolist():raise ValueError('Wrong endpoint frame')
        decoded=decode_selected_frames(source,indices,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        width,height=256,192
        canvas=Image.new('RGB',(n*width,45+len(result['cases'])*225),'white');d=ImageDraw.Draw(canvas)
        d.text((4,4),f'{name} | {"TRAIN" if i in maps["train_indices"] else "SOURCE VALIDATION SELECTION"} | NOVELTY, NOT SPILL PROBABILITY',fill='black')
        movie=[]
        for slot,frame_index in enumerate(indices):
            frame=torch.from_numpy(np.asarray(Image.fromarray(decoded[frame_index]).resize((width,height))).copy())
            panels=[frame.numpy()]
            for row,(case,c) in enumerate(result['cases'].items()):
                q=c['calibration'];lo,hi=q['q0.95'],q['q0.999']
                field=maps['maps'][case][i,0,slot//2].reshape(16,16)
                score=((field-lo)/max(hi-lo,1e-12)).clamp(0,1)
                pixel=inverse_letterbox_map(score,height,width,interpolation='nearest')
                colored=blend_rgb(frame,nonnegative_evidence_colors(pixel,scale=1.),.4)
                if slot%2==1:
                    canvas.paste(Image.fromarray(colored.numpy()),((slot//2)*width,45+row*225))
                    d.text(((slot//2)*width+3,45+row*225+195),f'{case} | src {frame_index}',fill='black')
                    d.text(((slot//2)*width+3,45+row*225+209),f'blue <= {lo:.4g}; red >= {hi:.4g}',fill='black')
                if case in main_cases:panels.append(colored.numpy())
            movie.append(np.concatenate(panels,1))
        folder=out/name;folder.mkdir();canvas.save(folder/'all_cases.png')
        arr=np.stack(movie);video=folder/'comparison.mp4'
        cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{arr.shape[2]}x{arr.shape[1]}','-r','4','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)]
        proc=subprocess.run(cmd,input=arr.tobytes(),capture_output=True)
        if proc.returncode:raise RuntimeError(proc.stderr.decode())
        Image.fromarray(arr[-1]).save(folder/'last_frame.png')
        gallery.append(dict(id=name,label=int(raw['labels'][i]),split='train' if i in maps['train_indices'] else 'validation_selection',sampled_indices=indices,
            source_sha256=raw['video_sha256'][i],video=str(video.relative_to(out)),figure=str((folder/'all_cases.png').relative_to(out))))
    # Existing temporal history is reused as a labeled baseline, not new fitting.
    history=EXPERIMENT/'training/loss_history.csv';rows=[]
    for r in csv.DictReader(history.open()):rows.append({k:float(v) for k,v in r.items() if v})
    line_figure(out/'original_trajectory_loss_curves.png',rows)
    shutil.copy2(history,out/'original_trajectory_loss_history.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if student_histories:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,key in zip(axes.flat,('loss','cosine_loss','spatial_contrast_loss','margin_loss')):
            for arm,h in student_histories.items():ax.plot([v['update'] for v in h],[v[key] for v in h],label=arm)
            ax.axvline(21,color='gray',linestyle=':',label='AdamW reset before 21');ax.set_title(key);ax.set_xlabel('full normal-source updates');ax.legend()
        fig.suptitle('Measured normal-only auxiliary losses; original trajectory unchanged. Different totals use different objectives.')
        fig.tight_layout();fig.savefig(out/'normal_training_loss_curves.png',dpi=160);plt.close(fig)
        (out/'normal_training_loss_history.json').write_text(json.dumps(student_histories,indent=2)+'\n')
    from sklearn.manifold import TSNE,trustworthiness
    z=maps['trajectory_embeddings'].numpy();y=maps['labels'].numpy();coordinates=TSNE(n_components=2,perplexity=15,random_state=239,init='pca',learning_rate='auto').fit_transform(z)
    fig,ax=plt.subplots(figsize=(8,6))
    for split,marker in [('train','o'),('validation_selection','^')]:
        idx=np.array([r['split']==split for r in gallery])
        for label,color in [(0,'royalblue'),(1,'tomato')]:
            take=idx&(y==label);ax.scatter(*coordinates[take].T,c=color,marker=marker,label=f'{split} {"clean" if label==0 else "cold"}')
    ax.legend();ax.set_title('Original trajectory checkpoint: t-SNE is not localization evidence');fig.tight_layout();fig.savefig(out/'trajectory_tsne.png',dpi=160);plt.close(fig)
    (out/'trajectory_tsne.json').write_text(json.dumps(dict(coordinates=coordinates.tolist(),ids=maps['ids'],seed=239,
        source_checkpoint_sha256=result['metadata']['trajectory_checkpoint_sha256'],original_embedding_trustworthiness=float(trustworthiness(z,coordinates,n_neighbors=5))),indent=2)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Annotation-free spill diagnostics</title>',
        '<h1>80 real-source normal-reference diagnostics</h1><p>No manual annotations, decoder or MIL. Maps show patch novelty, not class attention or liquid probability. Both H100s evaluated distinct source assignments.</p>',
        '<p>MP4 panels: original | '+ ' | '.join(main_cases)+'. PNGs show every fixed comparison. Shared per-case scales use training-clean q95/q99.9; native cells, no blur or per-frame rescaling.</p>',
        '<p>Curves and t-SNE below reuse the original trajectory checkpoint/history. The normal-reference fit has no optimizer/loss curve.</p>',
        '<img width="800" src="original_trajectory_loss_curves.png"><img width="600" src="trajectory_tsne.png">']
    if student_histories:page.append('<h2>New normal-only training</h2><img width="1000" src="normal_training_loss_curves.png">')
    for r in gallery:page.append(f'<h2>{html.escape(r["id"])} — {r["split"]}</h2><video controls preload="none" width="1000" src="{r["video"]}"></video><p><a href="{r["figure"]}">All 12 cases</a></p>')
    (out/'index.html').write_text('\n'.join(page))
    (out/'manifest.json').write_text(json.dumps(dict(rows=gallery,map_sha256=digest(a.run/'maps.pt'),scales='Fixed training-clean q95/q99.9 separately for each case',
        baseline_history_sha256=digest(history),plot_scope='Reused original trajectory fit, not new trajectory optimization'),indent=2)+'\n')
    print(json.dumps(dict(real_sources=len(gallery),videos=len(gallery),directory=str(out))))


if __name__=='__main__':main()
