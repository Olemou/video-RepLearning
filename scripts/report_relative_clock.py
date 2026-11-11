#!/usr/bin/env python3
"""Measured elapsed-clock comparison; plots first, spatial conclusions separate."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--plots-only',action='store_true');group.add_argument('--report-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    roots={'absolute clock':a.baseline/'frozen','elapsed clock':a.experiment/'frozen'}
    histories={k:json.loads((r/'history.json').read_text()) for k,r in roots.items()}
    settings={k:json.loads((r/'settings.json').read_text()) for k,r in roots.items()}
    assert len(set(v['initial_trajectory_sha256'] for v in settings.values()))==1
    assert len(set(v['prepared_sha256'] for v in settings.values()))==1
    figures=a.experiment/'figures'
    if not a.report_only:
        figures.mkdir(exist_ok=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        def export(fig,name):
            for suffix in ('png','pdf','svg'):fig.savefig(figures/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for color,(name,history) in zip(('tab:blue','tab:orange'),histories.items()):
            for split,style in (('train','-'),('selection','--')):
                ax.plot([r['update'] for r in history],[r[split]['total_loss'] for r in history],style,color=color,label=f'{name}: {split}')
        ax.axvline(20.5,color='gray',ls=':',lw=1,label='Optimizer reset before21')
        ax.set(xlabel='Full-batch optimizer update',ylabel='Four-loss total',title='80 sources: matched frozen V-JEPA, absolute vs elapsed clock')
        ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();export(fig,'total_loss')
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,(key,weight,title) in zip(axes.flat,[('semantic_loss',1,'Episode ViCo'),('alignment_loss',.1,'0.1 alignment'),('prediction_loss',1,'Prediction'),('retention_loss',.1,'0.1 retention')]):
            for color,(name,h) in zip(('tab:blue','tab:orange'),histories.items()):
                for split,style in (('train','-'),('selection','--')):
                    ax.plot([r['update'] for r in h],[weight*r[split][key] for r in h],style,color=color,label=f'{name}: {split}')
            if key=='semantic_loss':ax.set_yscale('log')
            ax.set(title=title,xlabel='Optimizer update');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle('Weighted components; fixed final-curriculum evaluation');fig.tight_layout();export(fig,'loss_components')
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        coords={}
        for ax,(name,root) in zip(axes,roots.items()):
            reps=torch.load(root/'representations.pt',map_location='cpu',weights_only=False)
            xy=TSNE(n_components=2,perplexity=15,random_state=239,init='pca',learning_rate='auto',max_iter=2000).fit_transform(reps['embeddings'][:,0].numpy())
            train=set(reps['train_indices'].tolist())
            for label,color in ((0,'tab:blue'),(1,'tab:red')):
                for split,marker in (('train','o'),('selection','s')):
                    ids=[i for i in range(80) if int(reps['labels'][i])==label and (i in train)==(split=='train')]
                    ax.scatter(xy[ids,0],xy[ids,1],color=color,marker=marker,edgecolors='black',linewidths=.4,label=f'{"clean" if label==0 else "cold"}: {split}')
            ax.set_title(name);ax.set_xticks([]);ax.set_yticks([]);ax.legend(fontsize=7)
            coords[name]=dict(ids=reps['ids'],coordinates=xy.tolist(),train_indices=reps['train_indices'].tolist(),selection_indices=reps['selection_indices'].tolist())
        fig.suptitle('Separate t-SNE fits; 64 train / 16 reused selection; not localization evidence');fig.tight_layout();export(fig,'tsne_comparison')
        (figures/'tsne_coordinates.json').write_text(json.dumps(coords,indent=2)+'\n')
        rows=[dict(arm=name,update=r['update'],split=split,**{key:r[split][key] for key in ('total_loss','semantic_loss','alignment_loss','prediction_loss','retention_loss')})
            for name,h in histories.items() for r in h for split in ('train','selection')]
        with (figures/'loss_history.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (a.experiment/'measured_results.json').write_text(json.dumps({name:h[-1] for name,h in histories.items()},indent=2)+'\n')
        print('Saved actual total loss, weighted components and t-SNE (PNG/PDF/SVG).',flush=True)
    if a.plots_only:return
    from PIL import Image,ImageDraw,ImageFont
    from control_experiment81 import content_cells
    from vision_model.interpretability.map_geometry import inverse_letterbox_map,nonnegative_evidence_colors,blend_rgb,letterbox_bounds
    import torch.nn.functional as F
    prepared=torch.load(a.experiment/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    spatial=a.experiment/'spatial_comparison';spatial.mkdir(exist_ok=True)
    spatial_rows=[]
    cases=json.loads((a.experiment/'all_blocks_gpu/completed.json').read_text())['results']
    for r in cases:
        name=r['id'];i=prepared['ids'].index(name);rgb=prepared['rgb'][i][-1].permute(1,2,0)
        h,w=rgb.shape[:2];canvas=Image.new('RGB',(3*w,h+75),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=15)
        draw.text((3,3),name+' | production input-gradient magnitude; NOT attention or spill probability',fill='black',font=font)
        draw.text((3,27),'Shared fixed scale0.001; yellow cross = automatic peak; no manual region labels',fill='black',font=font)
        canvas.paste(Image.fromarray(rgb.numpy()),(0,75));draw.text((3,51),'Original',fill='black',font=font)
        for column,(label,root) in enumerate((('absolute clock',a.baseline),('elapsed clock',a.experiment)),1):
            d=torch.load(root/'all_blocks_gpu'/f'{name}_frozen.pt',map_location='cpu',weights_only=False)
            field=F.avg_pool2d(d['input_gradient'][-1].abs().mean(0)[None,None],16,16)[0,0]
            support=content_cells(h,w);allowed=support.flatten().nonzero().flatten();values=field.flatten()[allowed]
            pos=int(allowed[values.argmax()]);top,left,rh,rw=letterbox_bounds(h,w)
            overlay=blend_rgb(rgb,nonnegative_evidence_colors(inverse_letterbox_map(field,h,w,interpolation='nearest'),scale=.001),alpha=.4)
            canvas.paste(Image.fromarray(overlay.numpy()),(column*w,75))
            x=((pos%16+.5)*16-left)*w/rw+column*w;y=((pos//16+.5)*16-top)*h/rh+75
            draw.line((x-8,y,x+8,y),fill='yellow',width=3);draw.line((x,y-8,x,y+8),fill='yellow',width=3)
            draw.text((column*w+3,51),label,fill='black',font=font)
            spatial_rows.append(dict(id=name,arm=label,native_peak=[pos//16,pos%16],maximum=float(values.max()),median=float(values.median())))
        canvas.save(spatial/f'{name}.png')
    (spatial/'native_peaks.json').write_text(json.dumps(spatial_rows,indent=2)+'\n')
    report=['# Elapsed-clock correction: matched80-source experiment','',
        '**The opt-in correction removes arbitrary file-origin dependence. It does not establish spill localization.** '
        'The authoritative original pipeline and checkpoints remain intact.','',
        '[Total-loss curves](figures/total_loss.png) · [Weighted components](figures/loss_components.png) · [t-SNE](figures/tsne_comparison.png) · [CSV](figures/loss_history.csv)','',
        '## Controlled protocol','',
        'Both arms freeze the same pretrained V-JEPA and train the same trajectory initialization for100 full-batch updates, '
        'using identical64 training sources,16 reused selection sources, seed239, two deterministically selected augmentation-bank views, '
        'AdamW settings and reset before21. The new run took14.12 seconds after initialization on the allocated H100. '
        'Only the adapter clock policy changes: `elapsed = timestamp - first_observed_timestamp`. '
        'The four losses and real future timestamp differences remain unchanged. The absolute control is reused from Experiment81, not rerun.','',
        '## Final selection results','',
        '| Arm | Total | ViCo | Alignment (raw) | Prediction | Retention (raw) | Correct |','|---|---:|---:|---:|---:|---:|---:|']
    for name,h in histories.items():
        r=h[-1]['selection'];report.append(f'| {name} | {r["total_loss"]:.8f} | {r["semantic_loss"]:.3g} | {r["alignment_loss"]:.6g} | {r["prediction_loss"]:.6g} | {r["retention_loss"]:.6g} | {round(r["accuracy"]*16)}/16 |')
    report+=['','The total is `ViCo + 0.1 align + predict + 0.1 keep`. t-SNE includes the same80 identity embeddings; '
        'circle/square denotes training/selection. The selected final checkpoint is update100, not a localization-selected epoch. '
        'Neither near-zero semantic loss nor t-SNE establishes complete convergence, material identity, or unseen-scene generalization.','',
        '## Source-excluded acquisition counterexample','',
        '`clean_folderImage81` is the only640×480 clean source in the173-record manifest; all78 cold sources have that resolution. '
        'It is byte-hash/source-ID excluded from the current80. It is not a verified same-scene pair and its label is taken from the manifest. '
        'Three predefined windows are repeated measurements of one source, not three independent test videos. '
        'Prior use in other research/model-selection operations is unknown.','',
        '| Window | Reused absolute frozen margin | Absolute full margin | Retrained elapsed frozen margin | Elapsed origin-shift difference |','|---|---:|---:|---:|---:|']
    original=json.loads((a.experiment.parent/'spill_attention_competition_80_20261106/acquisition_counterexample/results.json').read_text())
    relative=json.loads((a.experiment/'acquisition_counterexample/results.json').read_text())
    for window in ('beginning','center','end'):
        first=next(r for r in original['results'] if r['case']=='frozen' and r['window']==window)
        full=next(r for r in original['results'] if r['case']=='full' and r['window']==window)
        r=next(r for r in relative['results'] if r['window']==window)
        score=r['clock_controls']['actual']['spill_minus_clean_margin'];delta=score-r['clock_controls']['relative_zero']['spill_minus_clean_margin']
        report.append(f'| {window} | {first["clock_controls"]["actual"]["spill_minus_clean_margin"]:.5f} | {full["clock_controls"]["actual"]["spill_minus_clean_margin"]:.5f} | {score:.5f} | {delta:.3g} |')
    report+=['','Positive margin predicts spill; the manifest label is clean. **The retrained elapsed model still predicts spill for all three windows.** '
        'Its clock-origin dependence is removed in this check, but recognition of this acquisition counterexample is not improved. '
        'The original full model also predicts spill in all three actual-time windows, despite16/16 on its reused source-selection set. '
        'This is evidence of a real generalization gap; it does not establish exclusive resolution/scene dependence or verify a clean pixel mask.','',
        '## Spatial acceptance','',
        '[Native four-source input-gradient audit](all_blocks_gpu/completed.json) retains all24 blocks and actual production input derivatives. '
        'All saved-feature and gradient replays pass. Compare the new raw maxima with the original input, not color strength. '
        'No real masks are available; no IoU/pointing accuracy is claimed. A clock invariance correction is not a liquid-supervision objective.','',
        '[Cold56 comparison](spatial_comparison/cold_folderImage56.png) · [Cold10](spatial_comparison/cold_folderImage10.png) · '
        '[Cold13](spatial_comparison/cold_folderImage13.png) · [Raw peaks](spatial_comparison/native_peaks.json).','',
        '## Code changes and recommendation','',
        '`RelativeClockTrajectoryModel` is an opt-in subclass with identical parameter names/shapes. '
        '`train_experiment81.py --clock-policy relative_prefix` selects it; the default remains absolute. '
        'Checkpoint loaders honor the recorded policy, and resume rejects policy mismatch. '
        'Original four-loss/cached training, pretrained teacher and original checkpoints are preserved.','',
        'Keep this as a robustness ablation, not a deployed localization fix. The continued clean-counterexample failures and spatial misses '
        'support prioritizing cross-class acquisition overlap and reliable liquid-specific evidence before another global training run. '
        'Do not solve the appearance of a map by thresholding it into the requested target shape.']
    (a.experiment/'report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
