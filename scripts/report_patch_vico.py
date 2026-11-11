#!/usr/bin/env python3
"""Consolidate direct-patch ViCo experiments without inventing localization metrics."""
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
from PIL import Image,ImageDraw
from control_patch_vico import case_name
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(239)
    means=values[rng.integers(0,len(values),(3000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),interval95=np.quantile(means,[.025,.975]).tolist(),
                interpretation='Exploratory paired-source interval conditional on one fit; not independent scenes or repeated training')


def draw_map(values,scale,shape):
    values=np.clip(values/scale,-1,1)
    color=np.ones((*values.shape,3))
    color[:,:,0]-=np.maximum(-values,0)
    color[:,:,1]-=np.abs(values)
    color[:,:,2]-=np.maximum(values,0)
    square=Image.fromarray((color*255).astype(np.uint8)).resize((256,256),Image.Resampling.BILINEAR)
    h,w=shape;factor=min(256/h,256/w)
    rh,rw=round(h*factor),round(w*factor)
    top,left=(256-rh)//2,(256-rw)//2
    return square.crop((left,top,left+rw,top+rh)).resize((192,144),Image.Resampling.BILINEAR)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--additional-run',type=Path,nargs='*',default=[])
    p.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    a=p.parse_args();torch.set_num_threads(4)
    destination=a.run/'figures';destination.mkdir(exist_ok=False)
    reports=[(run,json.loads((run/'report.json').read_text())) for run in [a.run,*a.additional_run]]
    cases=[];maps={}
    for run,report in reports:
        for case in report['cases']:
            name=case_name(case)
            if name in maps:raise ValueError('Duplicate experiment identity')
            folder=run/case['case']
            saved=torch.load(folder/'representations.pt',map_location='cpu',weights_only=False,mmap=True)
            maps[name]=saved['signed_maps']
            cases.append((name,folder,case))
    base=reports[0][1]
    raw=torch.load(a.teacher_cache,map_location='cpu',weights_only=False)
    ids=raw['ids'];validation=[ids.index(v) for v in base['cases'][0]['validation_source_ids']]
    maximum=max(float(value[validation].abs().max()) for value in maps.values())
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    for i in validation:
        r=records[ids[i]];ends=raw['frame_indices'][i][raw['valid'][i]].tolist()
        frames=decode_selected_frames(Path(raw['dataset_root'])/r['video_path'],ends,frame_count=r['num_frames'],width=r['width'],height=r['height'])
        canvas=Image.new('RGB',(192*(len(cases)+1),65+174*len(ends)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,2),f'{ids[i]} | signed contribution to spill-minus-clean margin | fixed scale +/-{maximum:.5g}',fill='black')
        draw.text((4,20),'Red: positive spill evidence; blue: negative; these are not spill probabilities or verified regions.',fill='black')
        for col,title in enumerate(['Source frame',*[name for name,_,_ in cases]]):draw.text((192*col+4,43),title,fill='black')
        for t,frame_index in enumerate(ends):
            frame=frames[frame_index]
            canvas.paste(Image.fromarray(frame).resize((192,144)),(0,65+174*t))
            for col,(name,_,_) in enumerate(cases,1):
                canvas.paste(draw_map(maps[name][i,0,t].reshape(16,16).numpy(),maximum,frame.shape[:2]),(192*col,65+174*t))
            draw.text((4,65+174*t+147),f'tubelet {t}; source timestamp {float(raw["timestamps"][i,t]):.3f}s',fill='black')
        canvas.save(destination/f'{ids[i]}.png')
    curves=Image.new('RGB',(1250,700),'white');draw=ImageDraw.Draw(curves)
    colors=['black','red','blue','green','purple']
    for panel,(title,key,logarithm) in enumerate([
        ('Validation within-image cosine','within_image_patch_cosine',False),
        ('Validation mirror correspondence top-1','mirror_correspondence_top1',False),
        ('Validation feature variance','within_image_feature_variance',False),
        ('Validation episode ViCo; fixed final curriculum','vico_loss',True)]):
        x0,y0=40+(panel%2)*620,25+(panel//2)*345
        histories=[json.loads((folder/'evaluation_history.json').read_text()) for _,folder,_ in cases]
        values=[np.log10(max(row['validation'][key],1e-30)) if logarithm else row['validation'][key] for history in histories for row in history]
        low,high=min(values),max(values);span=max(high-low,1e-10)
        draw.text((x0,y0),title,fill='black');draw.rectangle((x0,y0+25,x0+535,y0+220),outline='black')
        for (name,_,case),history,color in zip(cases,histories,colors):
            points=[]
            for row in history:
                value=np.log10(max(row['validation'][key],1e-30)) if logarithm else row['validation'][key]
                points.append((x0+535*row['update']/case['updates'],y0+220-195*(value-low)/span))
            if len(points)>1:draw.line(points,fill=color,width=2)
        draw.text((x0,y0+225),f'y {low:.6g} .. {high:.6g}'+(' log10' if logarithm else ''),fill='black')
        for j,((name,_,_),color) in enumerate(zip(cases,colors)):draw.text((x0,y0+243+14*j),name,fill=color)
    curves.save(destination/'validation_curves.png')
    records=[]
    for name,folder,case in cases:
        for split in ('train','validation'):
            records.append(dict(case=name,split=split,checkpoint=str(folder/'checkpoint.pt'),**{k:v for k,v in case['metrics'][split].items() if isinstance(v,(int,float))}))
    with (a.run/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    paired={}
    baseline=dict((r['id'],r) for r in base['cases'][1]['metrics']['per_video'] if r['split']=='validation')
    for name,_,case in cases:
        if case.get('local_keep_weight',0):
            rows=[r for r in case['metrics']['per_video'] if r['split']=='validation']
            paired[name]={key:bootstrap([r[key]-baseline[r['id']][key] for r in rows]) for key in ('within_image_feature_variance','mirror_correspondence_top1','class_map_spatial_std')}
    (a.run/'paired_diagnostics.json').write_text(json.dumps(paired,indent=2)+'\n')
    control_path=a.run/'input_controls.json'
    controls=json.loads(control_path.read_text()) if control_path.exists() else None
    lines=['# All-patch ViCo: pipeline breakpoints, fixes and evaluation','',
      'The requested direct all-patch pathway is implemented without MIL, learned instance selection or a segmentation decoder. Engineering failures were corrected and the requested weighting was tested. The remaining question is physical spill localization; no verified spill masks or independent scene test exist. All new results are selection-set findings.','',
      '## Measured comparison','',
      '| Variant | Validation accuracy | Patch cosine within image | Feature variance | Participation rank | Mirror correspondence top-1 | Class-map spatial std |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('frozen_raw','initial'):
        m=base[name]['validation']
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['within_image_patch_cosine']:.6f} | {m['within_image_feature_variance']:.6f} | {m['participation_rank']:.4f} | {m['mirror_correspondence_top1']:.6f} | {m['class_map_spatial_std']:.6g} |")
    for name,_,case in cases:
        m=case['metrics']['validation']
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['within_image_patch_cosine']:.6f} | {m['within_image_feature_variance']:.6f} | {m['participation_rank']:.4f} | {m['mirror_correspondence_top1']:.6f} | {m['class_map_spatial_std']:.6g} |")
    lines += ['', 'Accuracy uses training-only identity-view prototypes and the original 16-source validation selection set (8 clean, 8 cold). All variants have the same original 64 training sources. The untrained projector is not a learned baseline. Frozen raw V-JEPA already classifies all 16 sources correctly. Mirror correspondence is exact patch-position retrieval after inverse reflection; chance is 1/192. Neither correspondence nor feature variance measures spill-region correctness.','',
      '## The actual paper and the implemented change','',
      'The local ViCo-LWIR PDF, Section IV-A.2 on printed page 7, explicitly describes patch-level ViT-B/16 representations, a 256-dimensional MLP projector and backbone fine-tuning before freezing for downstream classification. Eq.6 on page 5 gives same-original-image positives weight one and different-image, same-class positives uncertainty/curriculum weights. This adaptation follows the user\'s clarified all-patch interpretation. The paper does not explicitly write the video patch-index tensor, so it is not claimed to be an exact reproduction of an unavailable author implementation.','',
      'The accepted causal cache is `[80,2,6,4,256,1024]`. We select block 23 and retain all 192 content-valid cells on each native 16x16 grid; no spatial patch sampling occurs. Each update deterministically selects one valid tubelet per source, preserves both genuine cached identity/mirror views, and uses `image_id = video_index * 6 + tubelet_index`. Different tubelets are different original-image units. Video IDs remain separate split/group identifiers.','',
      'The patch projector is LayerNorm(1024), Linear(1024,256), GELU, Linear(256,256), then L2 normalization. It processes patches independently with shared weights, no added coordinates and no token-mixing attention. Contrast occurs directly before spatial averaging. Whole-episode evaluation uses only a fixed mean of valid patch embeddings.','',
      'Every same-class distinct patch is a positive under its source-image weak label. Same-image positive weights are exactly one, and cross-image same-class weights use the established detached uncertainty curriculum. Negatives are patches associated with the opposite source class. This is weak image-label propagation, not verified spill-patch supervision; background patches in positive sources also receive this association.','',
      'The optimized patch competition is `softplus(log(sum_neg alpha*exp(cos/tau)+epsilon) - (cos_positive/tau + log(w_positive)))`, averaged over positive partners, patches within an image, images within a source, and sources. Alpha retains Eq.7\'s square-root negative-count factor. Temperature is applied once. The original episode loss implementation is unchanged.','',
      '## Confirmed breakpoints and corrections','',
      '1. **Spatial bottleneck:** original trajectory input has shape `[B,T,1024]` after patch averaging. The new branch keeps `[B,V,T,256,256]` projected patches up to the direct loss. The original temporal model is preserved.','',
      '2. **Cubic uncertainty ranks:** the original small-batch helper creates a `[M,M,M]` comparison for average ranks. Calling it on 24,576 patch instances is infeasible. The new helper uses rowwise sorting and exact left/right bounds for ties. It retains the same rank definition and detached statistics.','',
      '3. **Quadratic autograd storage:** a dense all-pair matrix over all six times would need 162 GiB for float64 similarities alone. One time per source gives 24,576 instances; row chunks of 128 and a tested analytic stop-mining derivative retain all pair comparisons without storing their entire backward graph. Both anchor and candidate gradients are included. Only the normalized-feature gradient is saved. Higher-order derivatives are unsupported.','',
      '4. **Image/video provenance:** same-video identity previously served episode pairing. Patch weighting now uses original-image/tubelet IDs, while source IDs control splitting and source-balanced reduction. A different time is never silently given the same-image rule.','',
      '5. **Numeric precision:** the first new rank implementation unintentionally performed integer division in float32. Dense-reference tests exposed approximately 2e-9 loss disagreement. Casting ranks explicitly to cosine dtype fixed it. The corrected implementation passes dense float64 loss and first-order gradient checks, including ties and unequal-shard DDP.','',
      '6. **Objective-induced spatial homogenization:** the direct all-patch run and uniform-positive control both retain perfect recognition but suppress within-image feature differences. This is not a missing gradient. A class-constant construction (`q=+a` for one class, `q=-a` for the other) simultaneously gives high positive similarity and low negative similarity without representing spill boundaries. The all-patch image-label objective admits that solution. Same-image weight one does not prevent it.','',
      '## Tested correction: preserve local teacher information','',
      'After observing homogenization, two new matched replicas retain the requested all-patch loss and add a separate local feature term: `L = L_patch_ViCo + beta * L_local_keep`, with beta 0.1 or 1. The local term is half squared distance between normalized student patches and the immutable initial projector applied to the same frozen causal V-JEPA patch. Valid patches are averaged within each source and then sources are balanced. Targets are detached; original/mirror targets use their own actual encodes. No masks or patch selection are introduced.','',
      'This is a proposed correction, not a term from the ViCo-LWIR paper and not the original temporal `L_keep`. Its teacher preserves a seeded random nonlinear projection of pretrained patch information; it is not a spill detector or spatial ground truth. The comparison is exploratory after the base failure. Test its measured tradeoff rather than assuming preservation implies localization.','',
      '## Protocol, gradients and temporal protection','',
      'Every replica starts from the same projector weights, seed 239, learning rate 1e-3, AdamW epsilon 1e-12, zero decay, clipping 5 and 100 full-batch updates, with optimizer state reset before update 21. Each replica sees 6,400 source exposures, 12,800 source-view exposures and 2,457,600 valid patch-instance exposures. Prefix counts are recorded individually; temporal scheduling differs from the earlier all-times pooled experiment and is matched across these new controls.','',
      'Both H100 GPUs own distinct source shards (32 training and 40 evaluation sources each). Every training patch is an anchor exactly once globally. Candidates are gathered differentiably, sampler padding is absent, and rank loss scaling compensates for DDP gradient averaging. Metrics are assembled from all 80 unique sources. Only global rank zero writes consolidated artifacts.','',
      'Frozen cached patches have no training gradient into V-JEPA. The trainable projector receives direct patch gradients. The unchanged episode ViCo, alignment, future prediction and retention values are monitoring constants for the frozen temporal branch; they do not regularize the independent patch projector. Horizon-1/2/4 predictions, states, embeddings and parameters are checked before/after each replica, with exactly zero observed drift. No original checkpoint or figure is rewritten.','',
      '## Decision-specific spatial maps','',
      'For the evaluated fixed-mean classifier, let `mu` be the mean valid patch embedding, `z=mu/||mu||`, and `d=prototype_spill-prototype_clean`, fitted from training sources only. The signed map is `r(t,n)=q(t,n)^T d / ||mu||`. Its valid-patch mean equals the exact prediction margin `z^T d`. This reconstruction is tested with padding and reported per run. It is a classifier-contribution decomposition in contextual patch-feature space, not a pixel causal effect or a calibrated spill probability. V-JEPA tokens already contain spatial context.','',
      'All 16 validation videos have shared-scale signed contribution figures at every valid cached tubelet. Red increases the spill-versus-clean margin and blue decreases it. No validation label selects a class query, favorable head or region. The optional normalized class-map entropy uses fixed temperature 0.1; it is not the old attention-probability entropy and should not be compared numerically to that measure without this distinction.','',
      '## Input-level controls','']
    if controls:
        lines += ['All 16 validation videos are evaluated on both GPUs. For every classifier, 26 content cells are replaced by normalized channel-mean pixels across the clip: highest absolute signed contributions, lowest absolute contributions, or three matched seeded random selections. Every causal prefix is freshly encoded. Original video/encoder hashes and cached-score replay are checked. Uniform-frame controls preserve per-frame channel means. These interventions are OOD sensitivity tests, not masks.','',
          '| Variant | High removal true-class margin drop | Low removal | Mean random removal | Uniform-frame accuracy |',
          '|---|---:|---:|---:|---:|']
        for name,summary in controls['summaries'].items():
            random=np.mean([summary[f'random_{j}']['mean_true_class_margin_drop'] for j in range(3)])
            lines.append(f"| {name} | {summary['high']['mean_true_class_margin_drop']:.6g} | {summary['low']['mean_true_class_margin_drop']:.6g} | {random:.6g} | {summary['uniform_frames']['accuracy']:.4f} |")
        paired_controls={}
        for name in controls['summaries']:
            rows=[r for r in controls['rows'] if r['case']==name]
            differences=[]
            for r in rows:
                base_margin=r['conditions']['original']['margin'];sign=2*r['label']-1
                high=sign*(base_margin-r['conditions']['high']['margin'])
                random=np.mean([sign*(base_margin-r['conditions'][f'random_{j}']['margin']) for j in range(3)])
                differences.append(high-random)
            paired_controls[name]=bootstrap(differences)
        (a.run/'paired_input_controls.json').write_text(json.dumps(paired_controls,indent=2)+'\n')
        lines += ['', 'Per-source intervention scores, selected cells and clean/cold labels are retained in `input_controls.json`; exploratory high-minus-random paired intervals are in `paired_input_controls.json`. These intervals treat videos as the resampling unit; recurring scenes and one primary training seed limit their interpretation.']
    else:lines += ['Input controls are pending; no input-faithfulness improvement is claimed.']
    lines += ['', '## Limitations and recommendation','',
      'The direct patch implementation corrects the engineering bottlenecks and provides a mathematically connected contribution map. It does not turn image labels into verified patch labels. Within-image variation, correspondence and input sensitivity are distinct from spill localization. Inspect the retention comparisons and paired diagnostics before calling a spatial proxy improved. No variant is established as a validated spill localizer, and no classifier can improve the already perfect accuracy of this small selection set.','',
      'Keep the original temporal checkpoint as the supported recognition/temporal baseline. Treat the corrected patch path as an experimental branch. No backbone unfreezing is justified merely by a low patch loss or cleaner heatmap; the weak-label objective itself admits spatially homogeneous features. Independent scene evaluation and verified region annotations remain necessary to resolve physical localization.','',
      '## Reproducibility and artifacts','',
      '- `comparison.csv` includes train/validation classification, retrieval, pair separation, worst-pair loss and patch diagnostics.',
      '- `paired_diagnostics.json` compares local-retention variants with the direct all-patch baseline on the same sources.',
      '- Every case has its checkpoint/optimizer state, view-time exposures, training history, fixed-curriculum evaluation history, prototypes, full patch embeddings and signed maps.',
      '- Run `report.json` files record configuration, source hashes and git head. The primary baseline has an exact executed source snapshot from before adding local retention.',
      '- `figures/validation_curves.png` plots fixed-protocol validation diagnostics. Training patch losses with changing curriculum/sampled prefix are distinct from fixed whole-episode evaluation losses.',
      '- Actual localization IoU, pixel AP, pointing accuracy and inside-spill mass are unavailable, not zero. There are no warm sources, no verified scene groups and no untouched final test.', '',
      '## Exact checkpoints','']
    for name,folder,case in cases:lines.append(f'- {name}: `{folder / "checkpoint.pt"}`; temporal/future drift {case["temporal_output_and_future_max_drift"]}; peak rank-0 allocated CUDA memory {case["cuda_peak_allocated_bytes"]/2**30:.3f} GiB.')
    (a.run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(a.run/'report.md'),figures=len(list(destination.glob('*.png'))))),flush=True)


if __name__=='__main__':main()
