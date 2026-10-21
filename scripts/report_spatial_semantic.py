#!/usr/bin/env python3
"""Report E0/E1/E2 with shared-scale mirror figures for every validation source."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from train_spatial_semantic import DEFAULT_TEACHER,DEFAULT_SPATIAL
from visualize_trajectory_video import decode_selected_frames
from vision_model.training.spatial_semantic import mirror_total_variation,attention_diagnostics


def heat(map,maximum,shape):
    x=np.clip(map/max(maximum,1e-30),0,1)
    color=np.stack((x,np.zeros_like(x),1-x),-1)
    square=Image.fromarray((color*255).astype(np.uint8)).resize((256,256),Image.Resampling.BILINEAR)
    height,width=shape
    scale=min(256/height,256/width)
    rh,rw=round(height*scale),round(width*scale)
    top,left=(256-rh)//2,(256-rw)//2
    return square.crop((left,top,left+rw,top+rh)).resize((192,144),Image.Resampling.BILINEAR)


def compare_figure(frames,maps,source,path,maximum,difference_max,times):
    canvas=Image.new('RGB',(1152,68+174*len(frames)),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((4,3),f'{source} | spill-prototype-conditioned query; all heads averaged; not a spill probability',fill='black')
    draw.text((4,20),f'Fixed attention scale 0..{maximum:.5g}; difference scale 0..{difference_max:.5g}',fill='black')
    for col,label in enumerate(('Original frame','Original attention','Mirrored frame','Mirrored attention','Aligned mirror attention','Absolute difference')):
        draw.text((192*col+3,45),label,fill='black')
    for t,raw in enumerate(frames):
        frame=Image.fromarray(raw).resize((192,144),Image.Resampling.BILINEAR)
        a=maps[0,t,1].numpy()
        b=maps[1,t,1].numpy()
        aligned=b[:,::-1]
        shape=raw.shape[:2]
        pictures=[frame,heat(a,maximum,shape),frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                  heat(b,maximum,shape),heat(aligned,maximum,shape),heat(np.abs(a-aligned),difference_max,shape)]
        for col,picture in enumerate(pictures):canvas.paste(picture,(192*col,68+174*t))
        draw.text((3,68+174*t+146),f'tubelet {t} | source {times[t]:.3f}s',fill='black')
    canvas.save(path)


def curves(run,cases,path):
    canvas=Image.new('RGB',(1200,650),'white');draw=ImageDraw.Draw(canvas)
    for pane,(title,key,logarithm) in enumerate((('Validation ViCo; fixed final curriculum','fixed_semantic_loss',True),
        ('Validation mirror total variation','mirror_TV',False),('Validation normalized entropy','normalized_entropy',False),
        ('Validation clean/spill-query map cosine','class_map_cosine',False))):
        x0,y0=40+(pane%2)*590,30+(pane//2)*320
        draw.text((x0,y0),title,fill='black')
        histories=[json.loads((Path(c.get('artifact_directory',run/c['case']))/'evaluation_history.json').read_text()) for c in cases]
        values=[np.log10(max(row['validation'][key],1e-30)) if logarithm else row['validation'][key] for hist in histories for row in hist]
        low,high=min(values),max(values)
        span=max(high-low,1e-8)
        draw.rectangle((x0,y0+25,x0+510,y0+235),outline='black')
        for case,hist,color in zip(cases,histories,('blue','red')):
            points=[]
            for row in hist:
                value=np.log10(max(row['validation'][key],1e-30)) if logarithm else row['validation'][key]
                points.append((x0+510*row['update']/hist[-1]['update'],y0+235-210*(value-low)/span))
            draw.line(points,fill=color,width=2)
            draw.text((x0,y0+255+(0 if color=='blue' else 15)),case['case'],fill=color)
        draw.text((x0,y0+238),f'y {low:.4g}..{high:.4g}'+(' log10' if logarithm else ''),fill='black')
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pilot',type=Path)
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    p.add_argument('--spatial-cache',type=Path,default=DEFAULT_SPATIAL)
    a=p.parse_args()
    torch.set_num_threads(4)
    report=json.loads((a.run/'report.json').read_text())
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    spatial=torch.load(a.spatial_cache,map_location='cpu',weights_only=False,mmap=True)
    ids=raw['ids'];labels=raw['labels']
    train=torch.tensor([ids.index(i) for i in report['cases'][0]['train_source_ids']])
    validation=torch.tensor([ids.index(i) for i in report['cases'][0]['validation_source_ids']])
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    destination=a.run/'figures'
    destination.mkdir(exist_ok=False)
    maps_by_case={}
    saved_by_case={}
    for case in report['cases']:
        folder=Path(case.get('artifact_directory',a.run/case['case']))
        saved=torch.load(folder/'representations_and_maps.pt',map_location='cpu',weights_only=False)
        maps_by_case[case['case']]=torch.stack([row['attention'] for row in saved['rows']])
        saved_by_case[case['case']]=saved
    maximum=max(float(maps[validation].max()) for maps in maps_by_case.values())
    difference_max=max(float((maps[validation,0]-maps[validation,1].flip(-1)).abs().max()) for maps in maps_by_case.values())
    diagnostics={}
    for case in report['cases']:
        name=case['case'];maps=maps_by_case[name];saved=saved_by_case[name]
        content=saved['content'];valid=saved['valid']
        diag=attention_diagnostics(maps,valid,content)
        per_video=[]
        z=torch.stack([r['embedding'] for r in saved['rows']])
        prototypes=torch.stack([F.normalize(z[train,0][labels[train]==label].mean(0),dim=0) for label in (0,1)])
        predictions=(z@prototypes.T).argmax(-1)
        for i in validation.tolist():
            norm=spatial['tokens'][i,0,:,spatial['layers'].index(report['settings']['layer'])].float().norm(dim=-1).reshape(6,16,16)
            mask=content[i][None].expand_as(norm)
            correlation=float(spearmanr(norm[mask].numpy(),maps[i,0,:,1][mask].numpy()).statistic)
            tv=mirror_total_variation(maps[i:i+1],valid[i:i+1],content[i:i+1])
            row=dict(id=ids[i],label=int(labels[i]),mirror_TV=float(tv),token_norm_spearman=correlation,
                     identity_prediction=int(predictions[i,0]),mirror_prediction=int(predictions[i,1]),
                     **{k:float(v[i]) for k,v in diag.items()})
            per_video.append(row)
            record=records[ids[i]]
            ends=raw['frame_indices'][i][valid[i]].tolist()
            frames=decode_selected_frames(Path(raw['dataset_root'])/record['video_path'],ends,
                frame_count=record['num_frames'],width=record['width'],height=record['height'])
            compare_figure([frames[j] for j in ends],maps[i],
                           ids[i],destination/f'{name}_{ids[i]}.png',maximum,difference_max,raw['timestamps'][i].tolist())
        strata={}
        for label,category in ((0,'clean'),(1,'cold')):
            subset=[r for r in per_video if r['label']==label]
            strata[category]={k:float(np.mean([r[k] for r in subset])) for k in ('mirror_TV','normalized_entropy','class_map_cosine','maximum_probability','token_norm_spearman')}
        diagnostics[name]=dict(per_video=per_video,strata=strata,
            mirrored_validation_accuracy=float((predictions[validation,1]==labels[validation]).float().mean()),
            classification_flip_disagreement=float((predictions[validation,0]!=predictions[validation,1]).float().mean()),
            mean_token_norm_spearman=float(np.mean([r['token_norm_spearman'] for r in per_video])))
        # Supplementary joint t-SNE: validation participates in display fitting.
        xy=TSNE(n_components=2,perplexity=15,random_state=239,init='pca',max_iter=1500).fit_transform(z[:,0].numpy())
        canvas=Image.new('RGB',(800,600),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,10),f'{name}: all 80 embeddings jointly projected; filled=train, outline=validation',fill='black')
        span=np.maximum(xy.max(0)-xy.min(0),1e-6)
        scaled=(xy-xy.min(0))/span
        for i,(x,y) in enumerate(scaled):
            color='blue' if labels[i]==0 else 'red'
            px,py=50+700*x,60+480*y
            draw.ellipse((px-5,py-5,px+5,py+5),outline=color,fill=color if i in train else 'white',width=2)
        draw.text((15,570),'Class separation is not spill localization or an independent test.',fill='black')
        canvas.save(destination/f'{name}_tsne.png')
        with (destination/f'{name}_tsne.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['id','label','split','x','y'])
            writer.writerows((ids[i],int(labels[i]),'train' if i in train else 'validation',float(xy[i,0]),float(xy[i,1])) for i in range(len(ids)))
    first,second=[c['case'] for c in report['cases']]
    differences=np.array([r['mirror_TV'] for r in diagnostics[second]['per_video']])-np.array([r['mirror_TV'] for r in diagnostics[first]['per_video']])
    rng=np.random.default_rng(239)
    means=differences[rng.integers(0,len(differences),size=(2000,len(differences)))].mean(1)
    diagnostics['paired_source_bootstrap']=dict(mean_difference=float(differences.mean()),interval95=np.quantile(means,[.025,.975]).tolist(),
        caveat='Exploratory source bootstrap; shared scenes violate an independent-scene interpretation; one training seed')
    (a.run/'map_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    curves(a.run,report['cases'],destination/'validation_curves.png')
    pilot=json.loads((a.pilot/'report.json').read_text()) if a.pilot else None
    lines=['# Decoder-free spatial semantic attention: E0/E1/E2', '',
      'This experiment implements the plan in `info.md` with a parallel spatial semantic branch and an unchanged temporal checkpoint. It is an adaptation of localization/equivariance ideas, not a published combined objective or a validated spill localizer.', '',
      '## Measured comparison', '',
      '| Variant | Source-validation accuracy | Fixed ViCo loss | Mirror TV | Normalized entropy | Class-query map cosine |',
      '|---|---:|---:|---:|---:|---:|',
      '| E0 original temporal checkpoint | 16/16 | 6.6863e-6 | N/A | N/A | N/A |']
    for case in report['cases']:
        m=case['metrics']['validation']
        lines.append(f"| {case['case']} | {int(round(m['accuracy']*16))}/16 | {m['fixed_semantic_loss']:.6g} | {m['mirror_TV']:.6g} | {m['normalized_entropy']:.6g} | {m['class_map_cosine']:.6g} |")
    lines += ['', 'All variants are evaluated on the original 16-source validation **selection** set. The new branch fits only the original 64 training sources. Scenes recur across splits; no independent scene test or pixel masks exist. There are 40 clean and 40 cold sources, with no warm videos.', '',
      'Mirror TV is half the L1 distance between spatially aligned, normalized content-patch attention distributions, averaged over two queries, valid tubelets and sources. Raw probabilities are not pixel spill probabilities. Lower TV does not establish localization.', '',
      '## Architecture and objective', '',
      'The accepted cache retains `[80,2,6,4,256,1024]` frozen causal-prefix patch features. This run selects final block 23. A 128-wide, four-head, one-layer spatial Transformer processes each tubelet independently. Two queries are conditioned on frozen training-only class prototypes from the original trajectory model. A second all-head query-to-patch attention actually weights original patch values; fixed-order concatenation of both summaries enters a 128-dimensional episode projector. No ground-truth label selects a query, embedding route or validation map. There is no decoder or per-positive-patch label.', '',
      'Class conditioning identifies the source of the queries; it does not guarantee that the learned maps remain semantically distinct. Query similarity and target sensitivity are reported rather than assumed.', '',
      'The optimized spatial objective is `ViCo(z_spatial) + lambda(t) * mirror_TV`. The unchanged alignment, future-prediction and retention losses are monitored constants, not spatial regularizers. Original ViCo Eq.10, detached mining weights and float64 semantic arithmetic are reused without modification. No auxiliary CE or segmentation objective was introduced.', '',
      'Each condition uses seed 239, 100 full-batch updates of 64 sources (6,400 source exposures), both real identity/mirror views, AdamW LR 1e-3, zero decay, epsilon 1e-12, clipping 5 and a state reset before update 21. This differs from the original mini-batch run; the E1/E2 comparison has identical source exposure and initialization. Both H100s receive distinct source shards; global ViCo uses differentiable gathering and metrics are assembled from 80 unique sources. Only rank zero writes consolidated results.', '',
      '## Coefficient and scheduling evidence', '',
      f"Training-only initialization calibration chose target lambda={report['calibration']['selected_mirror_weight']:.9g}, making its shared-parameter gradient initially at most 10% of the semantic gradient. The semantic-only projector is excluded from that gradient-ratio comparison. This is an experimental scaling rule, not a paper-backed coefficient.", '',
      'The completed warmup comparison ramps this target linearly over 20 updates, then holds it fixed. It reuses the verified matched no-mirror baseline; no validation source chooses the coefficient. The warmup schedule is a hypothesis tested after the guarded pilot, so this is exploratory selection rather than a preregistered final test.']
    if pilot:
        pm=pilot['cases'][1]['metrics']['validation']
        final_weight=pilot['cases'][1]['final_mirror_weight']
        lines += ['',f"The preserved guarded pilot finished at validation TV {pm['mirror_TV']:.6g}. Its relative-gradient guard reduced lambda to {final_weight:.6g} as the already small semantic term converged, effectively removing equivariance pressure. This failed schedule motivated the single warmup comparison; it is not silently replaced or reported as a successful mirror experiment."]
    lines += ['', '## Temporal regression and convergence', '',
      'On both ranks, temporal parameter, state/dynamics/embedding and horizon-1/2/4 future-prediction maximum drift are exactly zero. The original checkpoint is never migrated or rewritten. Validation alignment remains 0.00118970, prediction 0.02693530, retention 0.00772590 and persistence 0.03907881. Frozen temporal losses do not contribute gradients to the spatial branch. Optional joint temporal fine-tuning (E3) was not justified and was not run.', '',
      'Fixed-final-curriculum validation history is saved independently of changing-curriculum training loss. A tiny semantic term is class fit, not convergence of every objective or evidence of pixel localization.', '',
      '## Spatial diagnostics and limitations', '',
      'Every validation source has a six-column mirror comparison in `figures/`: original frame, original attention, mirrored frame, mirrored attention, aligned mirrored attention and absolute difference. The spill-conditioned query is displayed for every video, including clean videos, without selecting by ground truth. All four heads are averaged. Attention and difference scales are shared across both conditions, sources and times; no per-frame display renormalization is used. All head tensors are retained in the saved map files.', '',
      'The displayed frames are the second source frame in each of the six observed tubelets, with exact source timestamps. No favorable video subset is used for mirror figures. Class-query similarity, effective attended patch count, entropy, maximum probability, per-video TV, mirrored classification and frozen-token norm correlation are in `map_diagnostics.json`. Clean and cold strata are reported separately.', '',
      f"The paired source bootstrap mean TV difference (E2 minus E1) is {diagnostics['paired_source_bootstrap']['mean_difference']:.6g}, with exploratory 95% interval {diagnostics['paired_source_bootstrap']['interval95']}. Shared scenes and a single seed prevent interpreting this as an independent-scene or repeated-training confidence interval.", '',
      'No IoU, pixel AP, pointing accuracy or inside-spill concentration is claimed because verified annotations are unavailable. t-SNE jointly fits all 80 points for display and is supplementary. Neither improved consistency nor perfect recognition establishes spill localization.', '',
      '## Input removal controls', '']
    control_file=a.run/'input_controls.json'
    if control_file.exists():
        control=json.loads(control_file.read_text())
        lines += ['For two predetermined cold and two clean validation sources, 26 equal-area high-, low- or seeded-random content patches are replaced by checkpoint-mean pixels. Every causal encoder prefix is recomputed from images. Original encoder and source hashes and cached-score replay are checked. Uniform-frame controls preserve per-frame channel means. These are OOD sensitivity tests with four sources, not localization ground truth.', '', '| Variant | Original accuracy | High removal absolute margin change | Low removal | Random removal |', '|---|---:|---:|---:|---:|']
        for name,summary in control['summaries'].items():
            lines.append(f"| {name} | {summary['original']['accuracy']:.3f} | {summary['high']['mean_absolute_margin_change']:.6g} | {summary['low']['mean_absolute_margin_change']:.6g} | {summary['random']['mean_absolute_margin_change']:.6g} |")
    else:lines.append('Input-level controls have not been generated; no input-faithfulness claim is supported.')
    m1,m2=[case['metrics']['validation'] for case in report['cases']]
    improved=m2['mirror_TV']<m1['mirror_TV']
    lines += ['', '## Research decision', '',
      ('The warmup condition improves measured mirror consistency on this selection set.' if improved else 'The warmup condition does not improve measured mirror consistency on this selection set.'), '',
      f"The clean/spill-conditioned query maps have mean cosine {m1['class_map_cosine']:.6g} without mirror and {m2['class_map_cosine']:.6g} with mirror. Both conditions therefore fail to establish distinct class-specific spatial selection. E2 has normalized entropy {m2['normalized_entropy']:.9g}, effective patch count {m2['effective_patches']:.6g}, and maximum probability {m2['maximum_probability']:.9g}. For these 192-content-patch videos, uniform attention has entropy 1, effective count 192 and maximum probability 1/192 = 0.00520833. E2 has collapsed to essentially uniform attention. Its lower mirror TV is a trivial consistency solution, so it fails the plan's no-collapse success criterion.", '',
      'Retain E0 for the existing recognition/temporal claim. E1/E2 are research checkpoints; no checkpoint is recommended as a validated spill localizer. The exact candidate checkpoints and their source directories are recorded in `report.json`, including the reused baseline path. This experiment does not justify unfreezing V-JEPA or promoting a new localization model. Verified spatial evaluation and scene/acquisition controls remain necessary to resolve the localization claim.', '',
      '## Sources and reproducibility', '',
      '- [MCTformer paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Class_Token_Transformer_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf): class-specific localization ideas. The prototype-conditioned adapter here is a distinct implementation.',
      '- [SEAM paper, Sections 3.2 and 4.2](https://arxiv.org/html/2004.04581v1): equivariance regularization; its main training transformation was rescaling, and adding horizontal flip gave only a small extra benefit in its own experiment. Our TV loss on normalized video attention is an adaptation, not its full objective.',
      '- `report.json`, per-condition checkpoints/optimizer states, update histories, gradient probes, view exposures and full attention maps record the executed settings. The available environment is Torch 2.11.0+cu128, differing from the package-declared Torch 2.7 range; `.venv/bin/python` is absent. No original experiment checkpoint or figure was changed.', '']
    (a.run/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(a.run/'report.md'),mirror_improved=improved,figures=len(list(destination.glob('*.png'))))),flush=True)


if __name__=='__main__':main()
