#!/usr/bin/env python3
"""Synthetic constant-input encoder test; not a padded video or training loss."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from attribute_experiment81 import load_case,evaluation_groups
from prepare_experiment81 import normalize_bank,digest
from vision_model.training.online_trajectory import OnlinePrefixEncoder,collect_prefix_features


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocation',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available() or os.environ.get('SLURM_JOB_ID')!=a.allocation:raise RuntimeError('Use the verified Slurm job step')
    a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda',0)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    names=['cold_folderImage56','cold_folderImage10','cold_folderImage13','clean_videoFrame121'];rows=[];payload={}
    for case in ('frozen','full'):
        encoder,model,reps,proto,settings=load_case(a.baseline,case,device)
        wrapper=OnlinePrefixEncoder(encoder);groups=evaluation_groups(reps,settings)
        for name in names:
            i=c['ids'].index(name);ids=next(g for g in groups if i in g);local=int((ids==i).nonzero()[0])
            pixels=torch.stack([normalize_bank(c['pixels'][int(j),:2],c['normalization']) for j in ids])
            real=collect_prefix_features(wrapper,pixels,c['valid'][ids],device,chunk_sources=len(ids)).cpu()
            error=float((real-reps['features'][ids]).abs().max())
            if error>1e-6:raise RuntimeError('Real feature replay changed')
            synthetic=pixels.clone();synthetic[local]=pixels[local,:,:1].expand_as(pixels[local])
            if not torch.equal(synthetic[local,:,0],synthetic[local,:,-1]):raise RuntimeError('Constant input construction failed')
            constant=collect_prefix_features(wrapper,synthetic,c['valid'][ids],device,chunk_sources=len(ids)).cpu()[local]
            observations={}
            for label,features in [('real',real[local]),('constant',constant)]:
                unit=F.layer_norm(features,(features.shape[-1],));variation=[]
                for horizon in (1,2,4):
                    variation.append(F.smooth_l1_loss(unit[:,horizon:],unit[:,:-horizon],reduction='sum'))
                count=sum(features.shape[0]*(features.shape[1]-k)*features.shape[-1] for k in (1,2,4))
                observations[label]=dict(normalized_prefix_variation=float(torch.stack(variation).sum()/count),
                    first_last_feature_cosine=float(F.cosine_similarity(features[:,0],features[:,-1]).mean()),
                    first_last_max_feature_difference=float((features[:,0]-features[:,-1]).abs().max()))
            rows.append(dict(case=case,id=name,production_replay_error=error,observations=observations,
                constant_to_real_prefix_variation=observations['constant']['normalized_prefix_variation']/max(observations['real']['normalized_prefix_variation'],1e-20)))
            payload[case+'_'+name]=constant
        del encoder,model,wrapper
        torch.cuda.empty_cache()
    torch.save(payload,a.output/'constant_features.pt')
    result=dict(scope='Controlled encoder function test: same normalized observed image at all input positions, independently encoded prefixes of2,4,...12 frames',
        caveat='Synthetic diagnostic, not dataset padding, supported training augmentation, physical-motion performance, or a revised teacher target. Ratios are not an additive decomposition.',
        allocation=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),gpu=torch.cuda.get_device_name(),
        rows=rows,training_updates=0,manual_annotations=0)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
