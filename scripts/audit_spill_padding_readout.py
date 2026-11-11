#!/usr/bin/env python3
"""All80-source readout probe: full grid, image content, or letterbox tokens."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from attribute_experiment81 import load_case,evaluation_groups
from prepare_experiment81 import normalize_bank,digest
from vision_model.training.online_trajectory import autocast_for,prefix_batches
from control_experiment81 import content_cells
from diagnose_spill_representation import recognition


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocation',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available() or os.environ.get('SLURM_JOB_ID')!=a.allocation:raise RuntimeError('Use the verified Slurm step')
    if torch.cuda.device_count()!=1:raise RuntimeError('This bounded probe uses the current one-GPU allocation')
    a.output.mkdir(parents=True,exist_ok=False);device=torch.device('cuda',0)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    output={};cache={}
    for case in ('frozen','full'):
        encoder,model,reps,proto,settings=load_case(a.baseline,case,device)
        candidates={name:torch.zeros_like(reps['features']) for name in ('all256','content192','padding64')}
        patch_reference=None;patch_error=0.
        for ids in evaluation_groups(reps,settings):
            support=content_cells(*c['rgb'][int(ids[0])].shape[-2:]).flatten().to(device)
            if int(support.sum())!=192 or any(not torch.equal(support.cpu(),content_cells(*c['rgb'][int(i)].shape[-2:]).flatten()) for i in ids):
                raise ValueError('This probe requires the recorded4:3 letterbox geometry')
            px=torch.stack([normalize_bank(c['pixels'][int(i),:2],c['normalization']) for i in ids])
            captured={}
            handle=encoder.patch_embed.register_forward_hook(lambda module,args,out:captured.__setitem__('patch',out[:,-256:][:,~support].detach()))
            for t,_,clip in prefix_batches(px,c['valid'][ids],len(ids)):
                with autocast_for(device):tokens=encoder(clip.to(device))[:,-256:]
                raw=captured['patch'].float()
                if patch_reference is None:patch_reference=raw[0,0].clone()
                patch_error=max(patch_error,float((raw-patch_reference).abs().max()))
                for name,values in (('all256',tokens),('content192',tokens[:,support]),('padding64',tokens[:,~support])):
                    candidates[name][ids,:,t]=values.mean(1).float().reshape(len(ids),2,-1).cpu()
            handle.remove()
            print(json.dumps(dict(case=case,group_ids=ids.tolist())),flush=True)
        replay=float((candidates['all256']-reps['features']).abs().max())
        if replay>1e-6:raise RuntimeError(f'Original complete-grid feature replay failed: {replay}')
        measured={}
        for readout,features in candidates.items():
            embeddings=torch.zeros_like(reps['embeddings'])
            for split in ('train','selection'):
                ids=reps[split+'_indices'];v=reps['valid'][ids].to(device);t=reps['timestamps'][ids].to(device)
                n,views,length,width=features[ids].shape
                pred=model(features[ids].flatten(0,1).to(device),v[:,None].expand(n,views,length).flatten(0,1),
                    t[:,None].expand(n,views,length).flatten(0,1),detach_teacher=False)
                embeddings[ids]=pred['embedding'].reshape(n,views,-1).cpu()
            metrics={}
            for split in ('train','selection'):
                ids=reps[split+'_indices']
                metrics[split]=recognition(embeddings[:,0],reps['labels'],reps['train_indices'],ids,
                    exclude_self=split=='train',prototypes=proto.cpu())
            measured[readout]=metrics
            cache[case+'_'+readout]=dict(features=features,embeddings=embeddings)
        output[case]=dict(checkpoint_sha256=digest(a.baseline/case/'checkpoint.pt'),original_feature_replay_error=replay,
            patch_embed_padding_max_difference_across_all_sources_times=patch_error,readouts=measured)
        print(json.dumps(dict(case=case,selection_accuracy={k:v['selection']['accuracy'] for k,v in measured.items()},patch_embed_padding_error=patch_error)),flush=True)
        del encoder,model;torch.cuda.empty_cache()
    torch.save(dict(ids=c['ids'],labels=c['labels'],train_indices=c['train_indices'],selection_indices=c['selection_indices'],readouts=cache),a.output/'representations.pt')
    (a.output/'results.json').write_text(json.dumps(dict(models=output,source_count=80,allocation=os.environ['SLURM_JOB_ID'],
        step=os.environ.get('SLURM_STEP_ID'),gpu=torch.cuda.get_device_name(),training_updates=0,manual_spatial_annotations=0,
        scope='Internal mean-readout substitution at fixed original heads/prototypes; not input removal or a trained regional classifier',
        caveat='Padding tokens can carry information gathered from real image content; success does not prove the model ignores liquid.'),indent=2)+'\n')


if __name__=='__main__':main()
