#!/usr/bin/env python3
"""Matched real-input removal of normalized-attention keys; reuse random controls."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from attribute_experiment81 import load_case, evaluation_groups, margins_for_group
from control_experiment81 import content_cells, transform_bank
from prepare_experiment81 import normalize_bank,digest
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.online_trajectory import OnlinePrefixEncoder,collect_prefix_features
from vision_model.interpretability.map_geometry import inverse_letterbox_map


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    if torch.cuda.device_count()!=1:raise RuntimeError('This bounded control is for the current one-GPU allocation')
    device=torch.device('cuda',0)
    output=a.experiment/'input_controls';output.mkdir(exist_ok=False)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    for case in ('frozen','full'):
        old=json.loads((a.baseline/'input_controls'/case/'results.json').read_text())
        sha=digest(a.baseline/case/'checkpoint.pt')
        if old['model_checkpoint_sha256']!=sha:raise ValueError('Random controls/checkpoint mismatch')
        encoder,model,reps,proto,settings=load_case(a.baseline,case,device)
        wrapper=OnlinePrefixEncoder(encoder)
        transform=OfficialVJEPA2Transform(settings['teacher_metadata']['encoder_dir'],preserve_field_of_view=True)
        groups=evaluation_groups(reps,settings)
        maps={}
        for path in sorted((a.experiment/case).glob('group_*.pt')):
            d=torch.load(path,map_location='cpu',weights_only=False)
            if d['checkpoint_sha256']!=sha:raise ValueError('Attribution/checkpoint mismatch')
            for j,i in enumerate(d['ids'].tolist()):maps[i]=d['logit_positive'][j].mean(0)
        if sorted(maps)!=list(range(80)):raise ValueError('Need completed all80-source maps')
        rows=[]
        for original in old['sources']:
            i=original['index'];rgb=c['rgb'][i];h,w=rgb.shape[-2:]
            if case=='frozen':
                ids=torch.tensor([i]);local=0
                px=normalize_bank(c['pixels'][i],c['normalization'])[None]
            else:
                ids=next(g for g in groups if i in g);local=int((ids==i).nonzero()[0])
                px=torch.stack([normalize_bank(c['pixels'][int(j),:2],c['normalization']) for j in ids])
            def forward(pixels):
                features=collect_prefix_features(wrapper,pixels,c['valid'][ids],device,chunk_sources=len(ids))[:,:2]
                return float(margins_for_group(model,reps,ids,features,proto,device)[local]),features[local].cpu()
            baseline,features=forward(px)
            error=float((features-reps['features'][i]).abs().max())
            if error>1e-6 or abs(baseline-original['baseline_margin'])>1e-5:
                raise RuntimeError('Unperturbed replay differs from reused controls')
            allowed=content_cells(h,w).flatten().nonzero().flatten()
            score=maps[i].sum(0).flatten()
            positions=allowed[score[allowed].argsort(descending=True,stable=True)[:max(1,round(len(allowed)*.05))]]
            mask=torch.zeros(256);mask[positions]=1
            raw_mask=inverse_letterbox_map(mask.reshape(16,16),h,w,interpolation='nearest')>.5
            fills=dict(mean=rgb.float().mean((-2,-1),keepdim=True).expand_as(rgb).round().byte(),
                blur33=F.avg_pool2d(F.pad(rgb.float(),(16,16,16,16),mode='reflect'),33,1).round().clamp(0,255).byte())
            records=[]
            for fill_name,fill in fills.items():
                changed=torch.where(raw_mask[None,None],fill,rgb)
                modified=px.clone();modified[local]=transform_bank(changed,c['augmentation_seeds'][i],transform,include_augmentations=case=='frozen')
                margin,_=forward(modified)
                old_records=[r for r in original['perturbations'] if r['fraction']==.05 and r['fill']==fill_name]
                random=[r['margin_drop'] for r in old_records if r['kind'].startswith('random_')]
                if len(random)!=3:raise ValueError('Expected three pre-existing matched random removals')
                old_top=next(r['margin_drop'] for r in old_records if r['kind']=='top')
                records.append(dict(fill=fill_name,margin_drop=baseline-margin,
                    random_mean_drop=sum(random)/len(random),old_probability_gradient_top_drop=old_top,
                    high_minus_random=baseline-margin-sum(random)/len(random),
                    old_high_minus_random=old_top-sum(random)/len(random),native_positions=positions.tolist(),
                    original_pixel_fraction=float(raw_mask.float().mean()),margin=margin))
            row=dict(id=original['id'],index=i,label=original['label'],split=original['split'],
                baseline_margin=baseline,replay_feature_error=error,records=records)
            rows.append(row);print(json.dumps(dict(case=case,**row)),flush=True)
            (output/f'{case}.json').write_text(json.dumps(dict(case=case,checkpoint_sha256=sha,sources=rows,
                removal_fraction=.05,random_control_provenance=str(a.baseline/'input_controls'/case/'results.json'),
                training_updates=0,localization_metric=False),indent=2)+'\n')
        del encoder,model,wrapper
        torch.cuda.empty_cache()
    (output/'completed.json').write_text(json.dumps(dict(sources_per_arm=17,extra_training_updates=0,
        method='all-layer positive logit sensitivity',spatial_masks_available=False),indent=2)+'\n')


if __name__=='__main__':main()
