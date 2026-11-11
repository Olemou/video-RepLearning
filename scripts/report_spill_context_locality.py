#!/usr/bin/env python3
"""Publish model-level locality, padding, static-prefix and source-excluded tests."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from control_experiment81 import content_cells
from vision_model.interpretability.map_geometry import inverse_letterbox_map,nonnegative_evidence_colors,blend_rgb,letterbox_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);out=a.experiment/'figures';out.mkdir(exist_ok=False)
    def load(name):return json.loads((a.experiment/name).read_text())
    locality=load('locality/results.json');dataset=load('dataset/results.json');extra=load('source_excluded/results.json')
    constant=load('constant_prefix/results.json');padding=load('padding_readout/results.json');attribution=load('content_only_attribution/results.json')
    norm=load('cached_token_norm_audit.json')['rows']
    stages=['patch_embed']+[f'block_{i:02d}' for i in range(24)]+['final_norm']
    def export(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,case in zip(axes,('frozen','full')):
        for fill,color in [('gain09','tab:blue'),('blur33','tab:orange')]:
            rows=[r for r in locality['rows'] if r['case']==case and r['fill']==fill]
            med=[statistics.median(r['relative_l2_change'] for r in rows if r['stage']==s) for s in stages]
            ax.plot(range(26),med,color=color,label=fill,lw=2)
        ax.set(title=case+' encoder',xlabel='Patch embedding → Transformer blocks0–23 → final norm')
        ax.set_xticks([0,1,6,12,18,24,25],['patch','0','5','11','17','23','norm']);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Relative L2 feature change at unchanged core patches')
    fig.suptitle('Distant-context interventions: protected input pixels and patch embeddings unchanged')
    fig.tight_layout();export(fig,'locality_by_block')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    names=['cold_folderImage56','cold_folderImage10','cold_folderImage13','clean_videoFrame121']
    for ax,case in zip(axes,('frozen','full')):
        for offset,kind,color in [(-.18,'real','tab:blue'),(.18,'constant','tab:orange')]:
            values=[next(r for r in constant['rows'] if r['case']==case and r['id']==n)['observations'][kind]['normalized_prefix_variation'] for n in names]
            ax.bar(np.arange(4)+offset,values,width=.36,color=color,label=kind+' input')
        ax.set_xticks(range(4),['cold56','cold10','cold13','clean121']);ax.set_title(case);ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Normalized contextual feature variation (not motion accuracy)')
    fig.suptitle('Independently encoded prefixes change even with identical input frames');fig.tight_layout();export(fig,'constant_prefix_variation')
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    font=ImageFont.load_default(size=15)
    for name in names:
        i=c['ids'].index(name);rgb=c['rgb'][i][-1].permute(1,2,0);h,w=rgb.shape[:2]
        canvas=Image.new('RGB',(3*w,h+75),'white');draw=ImageDraw.Draw(canvas)
        draw.text((3,3),name+' | Full100 checkpoint: actual input-gradient magnitude, NOT spill probabilities',fill='black',font=font)
        draw.text((3,27),'Original256-token pooling vs content192-only intervention; fixed scale0.001; automatic peak crosses',fill='black',font=font)
        canvas.paste(Image.fromarray(rgb.numpy()),(0,75));draw.text((3,51),'Original image',fill='black',font=font)
        for col,(label,path) in enumerate([('Original pooling',a.baseline/'all_blocks_gpu'/f'{name}_full.pt'),
            ('Content-only pooling',a.experiment/'content_only_attribution'/f'{name}_full.pt')],1):
            d=torch.load(path,map_location='cpu',weights_only=False)
            grid=torch.nn.functional.avg_pool2d(d['input_gradient'][-1].abs().mean(0)[None,None],16,16)[0,0]
            support=content_cells(h,w);allowed=support.flatten().nonzero().flatten();pos=int(allowed[grid.flatten()[allowed].argmax()])
            field=inverse_letterbox_map(grid,h,w,interpolation='nearest')
            overlay=blend_rgb(rgb,nonnegative_evidence_colors(field,scale=.001),alpha=.4)
            canvas.paste(Image.fromarray(overlay.numpy()),(col*w,75));draw.text((col*w+3,51),label,fill='black',font=font)
            top,left,rh,rw=letterbox_bounds(h,w);x=((pos%16+.5)*16-left)*w/rw+col*w;y=((pos//16+.5)*16-top)*h/rh+75
            draw.line((x-8,y,x+8,y),fill='yellow',width=3);draw.line((x,y-8,x,y+8),fill='yellow',width=3)
        canvas.save(out/f'{name}_content_only.png')
    text=['# Why spill recognition still fails to identify the liquid region','',
        '6 November2026. **New model-level tests confirm that class information is spatially distributed and that content-only pooling does not solve the known liquid-focus failures.** '
        'This round adds zero optimizer updates. Original four losses, teacher, checkpoints and previous loss/t-SNE figures are preserved.','',
        '## 1. Strong recognition persists on76 additional unique sources','',
        'The complete173-record manifest contains156 byte-unique videos:78 clean and78 cold. Thirteen duplicate groups account for17 redundant clean entries; '
        'none has contradictory labels. The selected80 remain byte-unique with no cross-split byte overlap. After excluding all80 IDs and hashes, '
        '76 unique sources remain:38 clean and38 cold.','',
        'All three fixed models use their original64-source training prototypes and one predefined centered12-frame window per added source. '
        'There is no prototype refitting or threshold selection on these76. These are source-excluded from the current models, '
        'not certified scene-held-out or historically untouched final-test data.','',
        '| Model | Correct /76 | AUROC | Retrieval mAP |','|---|---:|---:|---:|']
    for arm,d in extra['models'].items():
        m=d['metrics']['actual'];text.append(f'| {arm} | {round(m["accuracy"]*76)}/76 | {m["roc_auc"]:.4f} | {m["retrieval_mean_average_precision"]:.4f} |')
    text+=['','The full and elapsed models misclassify `clean_folderImage81`; the frozen absolute model recognizes its centered window correctly. '
        'The earlier first/last-window experiment remains a separate within-source diagnostic. '
        'AUROC1.0 with a classification error reflects preserved ranking under the unchanged decision boundary; it is not localization evidence. '
        'Most source-excluded data retains the acquisition/class coupling: after deduplication,77 clean sources are384×288, '
        'one clean source is640×480, and all78 cold sources are640×480. Resolution does not establish camera/session identity or exclusive neural dependence.','',
        '[Full source hashes and duplicates](dataset/results.json) · [76-source metrics and provenance](source_excluded/results.json)','',
        '## 2. Final patch tokens respond strongly to distant pixels','',
        'Twelve fixed4×4 tiles cover the12×16 real-image content grid. In every tile, the inner2×2 patches are the probe; '
        'a one-patch guard preserves their resize neighborhood. Pixels outside each tile are modified by gain0.9 or blur33. '
        'Every source/tile/fill uses the same complete causal-prefix path. These are systematically enumerated inference probes, '
        'not manual spill ROIs, training augmentations, or masks. The intervention modifies11/12 of the original frame, '
        'so results demonstrate nonlocal sensitivity under that scope, not sensitivity to a single distant pixel.','',
        '**Protected normalized pixels and pre-attention patch embeddings remain exactly unchanged in all192 interventions.** '
        'Nevertheless, their contextual tokens change. The table reports median relative L2 change over four sources and twelve tiles.','',
        '| Arm / intervention | Patch embed | Block0 | Block5 | Block11 | Block17 | Block23 | Final norm |','|---|---:|---:|---:|---:|---:|---:|---:|']
    chosen=['patch_embed','block_00','block_05','block_11','block_17','block_23','final_norm']
    for case in ('frozen','full'):
        for fill in ('gain09','blur33'):
            values=[statistics.median(r['relative_l2_change'] for r in locality['rows'] if r['case']==case and r['fill']==fill and r['stage']==s) for s in chosen]
            text.append(f'| {case} / {fill} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    text+=['','[All24-block locality curves](figures/locality_by_block.png) · [4,992 native measurements](locality/results.json)','',
        'Spatial indexing is preserved through the encoder, but a final token is not a measurement restricted to the pixels at its grid cell. '
        'Global attention mixes information before spatial mean pooling. Early blocks are more local under these probes; '
        'this does not establish that an early block identifies liquid, and it is not permission to select a favorable layer using the known failures. '
        'Nonlocal context can be useful; its presence alone is not an implementation defect or proof of background-only classification.','',
        '## 3. Empty letterbox positions carry enough class information','',
        'All80 originals have4:3 aspect ratio. Their256×256 input has192 image-content cells and64 artificial padding cells. '
        'At the patch embedding, padding vectors are exactly identical across sources/times in this test. '
        'The final padding tokens are contextualized, and can encode information gathered from real image pixels. '
        'I substituted full-grid, content-only and padding-only mean features for every causal prefix, retaining the original trained trajectory and training prototypes.','',
        '| Encoder / readout | Training correct /64 | Selection correct /16 | Min correct selection margin |','|---|---:|---:|---:|']
    for case,d in padding['models'].items():
        for name,m in d['readouts'].items():
            text.append(f'| {case} / {name} | {round(m["train"]["accuracy"]*64)}/64 | {round(m["selection"]["accuracy"]*16)}/16 | {m["selection"]["minimum_correct_class_margin"]:.4f} |')
    text+=['','**Both models classify all16 selection sources correctly using only final tokens positioned over padding.** '
        'This is not evidence that the model never uses liquid: padding tokens have read the image through attention. '
        'It demonstrates why class information at a token location cannot be equated with liquid at the corresponding pixel location. '
        'This is a conditional internal-readout substitution, not input-level removal or a trained padding classifier. '
        '[All-source readout results](padding_readout/results.json).','',
        '## 4. Content-only pooling was tested and rejected as a stand-alone focus fix','',
        'The candidate removes the64 artificial cells from the mean and changes no weights. Its classification remains16/16, '
        'but independent production input attribution still misses the spill. Exact replay is checked against the saved content-only features. '
        'The original full checkpoint is retained; the candidate is explicitly a different inference readout.','',
        '| Full model example | Original input-gradient peak | Content-only input-gradient peak |','|---|---|---|']
    old=json.loads((a.baseline/'all_blocks_gpu/completed.json').read_text())['results']
    for name in names[:3]:
        first=next(r for r in old if r['id']==name and r['case']=='full')
        second=next(r for r in attribution['rows'] if r['id']==name and r['case']=='full')
        text.append(f'| {name} | {first["input_gradient_peak"]} | {second["native_input_gradient_peak"]} |')
    text+=['','Coordinates are automatic native16×16 argmax values, not hand-labeled points. '
        'Cold56 and cold10 keep the same background maxima. Cold13 still peaks away from the main visible spill. '
        'Removing padding from the mean does not undo earlier global token mixing or create a liquid-region training signal.','',
        '[Cold56 comparison](figures/cold_folderImage56_content_only.png) · [Cold10](figures/cold_folderImage10_content_only.png) · '
        '[Cold13](figures/cold_folderImage13_content_only.png) · [Exact-gradient replay and raw results](content_only_attribution/results.json)','',
        '## 5. High-norm tokens are not a sufficient explanation','',
        'The cached all-block norm audit finds modest final-block maximum/median ratios in these examples. '
        'Full cold56 has max/median1.59, while its wrong strongest input-gradient cell is at only the19th norm percentile. '
        'Other wrong maxima also need not be the maximum-norm token. A generic register-token fix is therefore not justified by these measurements. '
        'This limited four-source audit cannot rule out register-like effects everywhere. [Native norm audit](cached_token_norm_audit.json).','',
        '## 6. Temporal teacher changes do not require visual motion','',
        'A separate encoder function test repeats the first observed image at every input position and encodes2,4,…12-frame prefixes independently. '
        'This is a synthetic diagnostic, not dataset tail padding, a training example, or a new prediction target. '
        'The measured quantity is normalized feature variation at horizons1/2/4, not physical-motion accuracy or a revised training loss.','',
        '| Encoder / source | Real-input prefix variation | Constant-input prefix variation | Ratio |','|---|---:|---:|---:|']
    for r in constant['rows']:
        text.append(f'| {r["case"]} / {r["id"]} | {r["observations"]["real"]["normalized_prefix_variation"]:.5f} | {r["observations"]["constant"]["normalized_prefix_variation"]:.5f} | {r["constant_to_real_prefix_variation"]:.2f} |')
    text+=['','[Prefix-variation figure](figures/constant_prefix_variation.png) · [Raw constant-input controls](constant_prefix/results.json)','',
        'The frozen teacher can change substantially because context length and temporal positions change even when the pixels do not. '
        'Thus prediction below persistence demonstrates prediction of the evaluated representation trajectory; it does not by itself demonstrate learning liquid motion. '
        'Ratios are not an additive decomposition of motion versus context, and nonlinear content/context interactions prevent subtracting one from the other to recover true dynamics. '
        'Causal-prefix encoding remains correct: no future frames were leaked. Do not replace it silently with full-video features, '
        'nor assume the first image is clean or remove static liquid by default.','',
        '## 7. Relation to papers and the current four losses','',
        'The local [ViCo-LWIR paper](../../../loss_doc/VicoLWIR.pdf), printed page7, describes ViT-B/16 fine-tuning using patch-level representations '
        'and a256-dimensional projector. Its evaluation is primarily image-level; the qualitative attention figure is not a guarantee of mask-level localization. '
        'Patch features and weak image labels alone do not establish that every positive-image patch is liquid. '
        'The present episode objective operates after spatial averaging and temporal adaptation, a different supervision interface.','',
        '[Vision Transformers Need More Than Registers, CVPR2026](https://arxiv.org/html/2602.22394v2), Sections4–5, '
        'studies background/global-semantic aggregation and proposes selective feature pooling during pretraining. '
        'It is relevant to the nonlocal-token hypothesis, but does not establish liquid identity for this JEPA pipeline. '
        'Its pooling method and benchmarks have not been reproduced here, and no untrained gate/register tokens or new loss were silently added.','',
        '[Token Contrast, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.html) '
        'uses intermediate token relations to address late-token over-smoothing. It motivates preserving spatial structure, '
        'but preserving distinct patches does not tell the model which dark structure is liquid. Its full segmentation system is outside this no-decoder experiment.','',
        'The four global terms remain `ViCo + 0.1 align + predict + 0.1 keep`. '
        'Their targets constrain episode class, matched-view states and contextual teacher trajectories. '
        'They do not require a spill-specific spatial readout. Full fine-tuning, elapsed-clock correction, attention-logit derivatives '
        'and content-only pooling have each failed to establish reliable liquid focus under the tested protocols.','',
        '## 8. Updated root-cause assessment and next correction','',
        '1. **Spatially distributed class information is directly demonstrated.** Distant-context tests and padding-only recognition show why contextual token scores do not identify local material. The causal adapter then discards the spatial axis entirely.','',
        '2. **The global objective does not resolve liquid identity/extent.** Strong recognition on76 additional sources coexists with known spatial failures. Improving another global curve is not the next localization criterion.','',
        '3. **Acquisition/class overlap remains sparse.** One of78 unique clean sources shares the cold cohort’s native dimensions, and that additional clean source is a full/elapsed-model false positive. This supports further cross-acquisition validation, not a claim of exclusive resolution causality.','',
        '4. **Temporal/context and attribution effects are secondary explanations.** Static-input variation and imperfect spatial-token locality limit interpretation of prediction/attention. Neither high-norm outliers nor padding inclusion alone explains all misses.','',
        '**Keep the original four-loss branch for video recognition and trajectory learning.** A regional correction needs both a genuinely local feature path and an independently supported liquid cue. '
        'The previously tested relative-thermal candidate teacher has already failed liquid identity/coverage; repeating it with a different projector or colormap is unsupported. '
        'The practical annotation-free route is verified same-camera/exposure dry-versus-spill acquisition, with dark non-liquid/person-action controls, '
        'or an independent calibrated thermal/material signal. Capture-time clean status, registration and occlusion checks must be established; '
        'the existing first frame or unrelated clean clips cannot serve as automatic truth. This requires no hand-drawn spill masks, MIL or decoder.','',
        'After that evidence exists, test a separate patch-preserving local head with explicit matched-region targets and retain the four global terms. '
        'Until then, no available code-only loss adjustment is validated as the spill-focus fix. '
        'The current report therefore records **localization acceptance failed**, rather than claiming a solution from better recognition or sharper heatmaps.','',
        '## Execution and limitations','',
        f'Current Slurm allocation{locality["settings"]["allocation"]}, explicit srun steps, one{locality["settings"]["gpu"]} on{locality["settings"]["host"]}. '
        'All probes preserve saved checkpoints and publish exact unperturbed replays. '
        'No masks, trusted scene/session groups, warm sources or independently untouched final scene test are available. '
        'Feature substitutions, distant interventions, source-level classification and qualitative localization are separate measurements. '
        'No IoU, pixel AP, verified pointing accuracy, or claim that the model never uses any liquid information is made.']
    (a.experiment/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(locality_observations=len(locality['rows']),source_excluded=76,padding_sources=80,training_updates=0,localization_acceptance='failed')),flush=True)


if __name__=='__main__':main()
