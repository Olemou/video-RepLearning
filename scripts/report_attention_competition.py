#!/usr/bin/env python3
"""Publish all-source competition maps, raw peaks and matched-removal evidence."""
import argparse
import html
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from control_experiment81 import content_cells
from vision_model.interpretability.map_geometry import spatial_evidence_overlay,letterbox_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    out=a.experiment/'figures';out.mkdir(exist_ok=False)
    keys=('probability_positive','weighted_positive','logit_positive')
    maps,stats,replays={},{},{}
    for case in ('frozen','full'):
        maps[case]={};stats[case]=[]
        for path in sorted((a.experiment/case).glob('group_*.pt')):
            d=torch.load(path,map_location='cpu',weights_only=False)
            for j,i in enumerate(d['ids'].tolist()):
                maps[case][i]={key:d[key][j] for key in keys}
            for r in d['statistics']:
                stats[case].append(dict(r,index=int(d['ids'][r['source_in_group']])))
        if sorted(maps[case])!=list(range(80)):raise ValueError('Need all80 distinct sources')
        replays[case]=json.loads((a.experiment/case/'progress.json').read_text())
    scales={}
    for key in keys:
        values=[]
        for case in maps:
            for i in c['train_indices'].tolist():
                content=content_cells(*c['rgb'][i].shape[-2:])
                values.append(maps[case][i][key].mean(0)[:,content].flatten())
        scales[key]=max(float(torch.quantile(torch.cat(values),.995)),1e-20)
    layer_values=[]
    for case in maps:
        for i in c['train_indices'].tolist():
            content=content_cells(*c['rgb'][i].shape[-2:])
            layer_values.append(maps[case][i]['logit_positive'][:,:,content].flatten())
    layer_scale=max(float(torch.quantile(torch.cat(layer_values),.995)),1e-20)
    font=ImageFont.load_default(size=14)
    observations,inventory=[],[]
    for case in maps:
        for i,name in enumerate(c['ids']):
            folder=out/f'{name}_{case}';folder.mkdir()
            rgb=c['rgb'][i].permute(0,2,3,1);h,w=rgb.shape[1:3]
            content=content_cells(h,w);allowed=content.flatten().nonzero().flatten()
            frames=[]
            fields={key:maps[case][i][key].mean(0) for key in keys}
            fields['logit_last_block']=maps[case][i]['logit_positive'][-1]
            for t in range(6):
                canvas=Image.new('RGB',(5*w,h+65),'white');draw=ImageDraw.Draw(canvas)
                draw.text((4,3), f'{name} | {case} | tubelet {t} | normalized-attention sensitivity, NOT spill masks',fill='black',font=font)
                canvas.paste(Image.fromarray(rgb[2*t+1].numpy()),(0,65))
                titles=['Original','Old: positive dS/dA','All blocks: positive A*dS/dA','All blocks: positive dS/dlogit','Last block: positive dS/dlogit']
                for col,title in enumerate(titles):draw.text((col*w+4,33),title,fill='black',font=font)
                for col,key in enumerate(fields,1):
                    field=fields[key][t]
                    scale=layer_scale if key=='logit_last_block' else scales[key]
                    overlay=spatial_evidence_overlay(rgb[2*t+1],field,scale=scale,
                        scope='conditional_attention_key_gradient',signed=False,style='blue_red',alpha=.4)
                    canvas.paste(Image.fromarray(overlay.numpy()),(col*w,65))
                    v=field.flatten()[allowed];position=int(allowed[v.argmax()]);prob=v/v.sum().clamp_min(1e-30)
                    observations.append(dict(case=case,index=i,id=name,label=int(c['labels'][i]),
                        split='train' if i in c['train_indices'] else 'selection',tubelet=t,method=key,
                        native_peak=[position//16,position%16],max_value=float(v.max()),
                        peak_to_median=float(v.max()/v.median().clamp_min(1e-30)),
                        top5percent_mass=float(v.topk(round(len(v)*.05)).values.sum()/v.sum().clamp_min(1e-30)),
                        normalized_entropy=float(-(prob*prob.clamp_min(1e-30).log()).sum()/np.log(len(v)))))
                frames.append(canvas)
            frames[-1].save(folder/'endpoint.png')
            storyboard=Image.new('RGB',(5*w,6*(h+65)),'white')
            for t,frame in enumerate(frames):storyboard.paste(frame,(0,t*(h+65)))
            storyboard.save(folder/'all_times.png')
            if name in ('cold_folderImage56','cold_folderImage10','cold_folderImage13'):
                montage=Image.new('RGB',(1280,6*280+70),'white');draw=ImageDraw.Draw(montage)
                draw.text((3,3),f'{name} | {case} | all24 blocks: positive normalized-attention logit sensitivity',fill='black',font=font)
                draw.text((3,25),f'One training-only all-block scale {layer_scale:.4g}; yellow crosses = automatic maxima, not labels',fill='black',font=font)
                for layer in range(24):
                    field=maps[case][i]['logit_positive'][layer,-1]
                    overlay=spatial_evidence_overlay(rgb[-1],field,scale=layer_scale,
                        scope='conditional_attention_key_gradient',signed=False,style='blue_red',alpha=.4)
                    tile=Image.fromarray(overlay.numpy()).resize((320,240))
                    v=field.flatten()[allowed];position=int(allowed[v.argmax()])
                    top,left,rh,rw=letterbox_bounds(h,w)
                    x=(((position%16)+.5)*16-left)*320/rw;y=(((position//16)+.5)*16-top)*240/rh
                    marker=ImageDraw.Draw(tile);marker.line((x-6,y,x+6,y),fill='yellow',width=2);marker.line((x,y-6,x,y+6),fill='yellow',width=2)
                    cx,cy=(layer%4)*320,70+(layer//4)*280
                    draw.text((cx+3,cy),f'Block {layer:02d}; peak/median {float(v.max()/v.median().clamp_min(1e-30)):.2f}',fill='black',font=font)
                    montage.paste(tile,(cx,cy+35))
                montage.save(folder/'all24.png')
            inventory.append(dict(id=name,case=case,path=f'{name}_{case}/endpoint.png',all_times=f'{name}_{case}/all_times.png'))
    calibration=dict(scales=scales,all24_logit_layer_scale=layer_scale,per_frame_normalization=False,
        policy='Training-only q99.5, pooled across arms, every observed time, valid content; separate units for each derivative',
        no_manual_roi=True,no_head_selection=True,localization_validated=False)
    (a.experiment/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (a.experiment/'native_measurements.json').write_text(json.dumps(observations,indent=2)+'\n')
    (a.experiment/'layer_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Normalized-attention audit: all80 sources, both checkpoints</h1>'
        '<p>Same final decision; all24 blocks and all heads. Different derivatives have different fixed training-only scales. '
        'These are conditional token sensitivities, not spill probabilities. Each tubelet spans two sampled frames.</p>'+''.join(
            f'<p>{html.escape(e["id"])} / {e["case"]}: <a href="{e["path"]}">Last tubelet</a> · <a href="{e["all_times"]}">All times</a></p>' for e in inventory))
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    report=['# Spill-attention focus: normalized-competition follow-up','',
        '**This experiment changes attribution only. No model or teacher weights, losses or training defaults were changed.** '
        'The four-loss Experiment81 classification metrics and t-SNE remain unchanged. Localization acceptance is assessed separately after inspecting these new maps.','',
        '[All80-source comparison](figures/index.html) · [Raw native measurements](native_measurements.json) · [Calibration](calibration.json)','',
        '## A limitation of unconstrained attention gradients','',
        'For A=softmax(U), g=dS/dA, the exact chain rule gives `dS/dU = A * (g - sum_key(A*g))`. '
        'The original positive gradient g measures sensitivity to treating attention probabilities as independent. '
        'The new derivative measures normalized attention redistribution. A row-constant term in g disappears under the latter. '
        'Adding a term c*(sum_key A - 1) to a score extension changes its unconstrained dS/dA but leaves every valid normalized score unchanged. '
        'Therefore positive dS/dA alone is not invariant to that choice of extension. It is still a well-defined conditional derivative for the captured graph; '
        'this is a difference in the question answered, not evidence that the old autograd tensor was disconnected.','',
        'This calculation is an ordinary softmax derivative, **not a reproduced LeGrad, AttnLRP, or new localization loss**. '
        'The original all-layer positive gradients, the weighted gradients and the new logit gradients use the same final trained prototype margin. '
        'All probabilities remain normalized; all heads/queries and all24 blocks contribute. ReLU is applied only to the explicitly positive maps; signed native results remain saved.','',
        '## Measured derivative components','',
        'The table reports the median attention-weighted row-constant energy fraction across sources, prefixes, and blocks in each band. '
        'It is a derivative decomposition, not the fraction of false-positive pixels.','',
        '| Arm | Blocks0–5 | Blocks6–17 | Blocks18–23 |','|---|---:|---:|---:|']
    for case in maps:
        values=[statistics.median([r['row_constant_weighted_energy_fraction'] for r in stats[case] if lo<=r['block']<=hi]) for lo,hi in ((0,5),(6,17),(18,23))]
        report.append(f'| {case} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    report+=['','The decomposition can remove a broad component from an explanation. It cannot demonstrate that the remaining peaks are liquid. '
        'Existing production input gradients already place their strongest cold56 response on background; a better attention derivative alone cannot change that learned decision.','',
        '## Reproduction checks','']
    for case,rows in replays.items():
        report.append(f'- {case}: {sum(len(r["ids"]) for r in rows)} distinct sources; maximum old-gradient reproduction error '
            f'{max(r["old_gradient_max_error"] for r in rows):.3g}; maximum score error {max(r["old_score_max_error"] for r in rows):.3g}; '
            f'explicit feature replay error {max(r["explicit_replay_max_error"] for r in rows):.3g}.')
    report+=['','Production-versus-explicit BF16 discrepancies are inherited from the baseline and remain disclosed in '
        '[Experiment81 verification](../experiment_81_execution_20261106/verification.json). '
        'Exact derivative replay does not establish bit-identical production attribution. The fixed checkpoints are not retrained.','',
        '## Matched real-input removal','',
        'Select the top5% of content cells by all-layer positive logit sensitivity summed over key times; remove the same cells across the sampled clip. '
        'Recompute raw-video preprocessing and all causal-prefix features under mean/blur33 fills. '
        'Use the original checkpoint-specific training prototypes and reuse the three original matched random controls, after exact unperturbed replay. '
        'The 16 source-selection videos and cold56 training diagnostic remain separate; random reuse avoids unnecessary GPU work.','',
        '| Arm | Selection category | Fill | Old high-minus-random | Logit high-minus-random |','|---|---|---|---:|---:|']
    summary=[]
    if (a.experiment/'input_controls/completed.json').exists():
        for case in maps:
            rows=json.loads((a.experiment/'input_controls'/f'{case}.json').read_text())['sources']
            for label,name in ((1,'cold'),(0,'clean')):
                for fill in ('mean','blur33'):
                    records=[p for r in rows if r['split']=='selection' and r['label']==label for p in r['records'] if p['fill']==fill]
                    old=float(np.mean([p['old_high_minus_random'] for p in records]));new=float(np.mean([p['high_minus_random'] for p in records]))
                    summary.append(dict(case=case,label=label,fill=fill,n=len(records),old=old,logit=new))
                    report.append(f'| {case} | {name} (n={len(records)}) | {fill} | {old:.6g} | {new:.6g} |')
        (a.experiment/'removal_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report+=['','These are decision-faithfulness controls, **not localization metrics**. Region removal can cause distribution shifts; '
        'stronger score changes do not certify liquid identity. No IoU or pointing accuracy is manufactured from automatic peaks.','',
        '## Research grounding','',
        '[PyTorch Softmax definition](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Softmax.html) defines the normalized row operation; '
        'the derivative above follows directly by the chain rule and is tested against autograd and central finite differences. '
        '[LeGrad](https://arxiv.org/html/2404.03214v2) motivates inspecting attention derivatives but uses a different layerwise target formulation. '
        '[AttnLRP, ICML2024](https://proceedings.mlr.press/v235/achtibat24a.html) explicitly handles attention in relevance propagation; '
        'its full relevance algorithm has not been implemented or silently substituted here.','',
        '## Limits and acceptance','',
        'No spatial masks, trusted scene/session groups, warm examples, or untouched independent scene test are available. '
        'The source-selection set is repeatedly used. Inspect native known failures and original/map pairs; no favorable layer or threshold may be selected to match a reference image. '
        'A successful mathematical derivative check is not successful spill localization. The original global four losses still supply no liquid-extent constraint.']
    (a.experiment/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(sources_per_arm=80,figures=len(inventory),extra_training_updates=0,removal_rows=len(summary))),flush=True)


if __name__=='__main__':main()
