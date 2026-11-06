#!/usr/bin/env python3
"""Input attribution under content-only pooling; original heads/prototypes fixed."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from attribute_experiment81 import load_case,evaluation_groups,margins_for_group
from prepare_experiment81 import normalize_bank,digest
from control_experiment81 import content_cells
from vision_model.training.online_trajectory import OnlinePrefixEncoder
from vision_model.interpretability.prefix_input_gradients import prefix_input_gradient


class ContentPrefixEncoder(OnlinePrefixEncoder):
    """Diagnostic readout only; backbone attention and checkpoints unchanged."""
    def __init__(self,encoder,support):
        super().__init__(encoder)
        if support.ndim!=2 or not support.any():raise ValueError('Need nonempty spatial support')
        self.register_buffer('support',support.bool(),persistent=False)

    def forward(self,clip):
        patch=self.encoder.architecture.patch_size
        if (clip.shape[-2]//patch,clip.shape[-1]//patch)!=tuple(self.support.shape):raise ValueError('Spatial support/grid mismatch')
        tokens=self.encoder(clip)[:,-self.support.numel():]
        return tokens[:,self.support.flatten()].mean(1).float()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--pooling',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allocation',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available() or os.environ.get('SLURM_JOB_ID')!=a.allocation:raise RuntimeError('Use verified Slurm job step')
    a.output.mkdir(parents=True,exist_ok=False);device=torch.device('cuda',0)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    probes=torch.load(a.pooling/'representations.pt',map_location='cpu',weights_only=False)
    rows=[]
    for case in ('frozen','full'):
        encoder,model,reps,proto,settings=load_case(a.baseline,case,device)
        # All-source probes used the original training prototypes; retain those.
        reps['features']=probes['readouts'][case+'_content192']['features']
        groups=evaluation_groups(reps,settings)
        for name in ('cold_folderImage56','cold_folderImage10','cold_folderImage13','clean_videoFrame121'):
            i=c['ids'].index(name);ids=next(g for g in groups if i in g);local=int((ids==i).nonzero()[0])
            support=content_cells(*c['rgb'][i].shape[-2:])
            wrapper=ContentPrefixEncoder(encoder,support.to(device))
            px=torch.stack([normalize_bank(c['pixels'][int(j),:2],c['normalization']) for j in ids])
            result=prefix_input_gradient(wrapper,px,c['valid'][ids],
                lambda f:margins_for_group(model,reps,ids,f,proto,device)[local],device)
            error=float((result['features']-reps['features'][ids]).abs().max())
            if error>1e-6:raise RuntimeError('Content-only forward differs from saved all80 probe')
            grid=torch.nn.functional.avg_pool2d(result['gradient'][local,0,-1].abs().mean(0)[None,None],16,16)[0,0]
            allowed=support.flatten().nonzero().flatten();v=grid.flatten()[allowed];pos=int(allowed[v.argmax()])
            row=dict(case=case,id=name,pool='content192',prototype_margin=result['score'],replay_error=error,
                prefix_gradient_replay_error=result['replay_max_error'],native_input_gradient_peak=[pos//16,pos%16],
                peak_to_median=float(v.max()/v.median().clamp_min(1e-30)),checkpoint_sha256=digest(a.baseline/case/'checkpoint.pt'))
            torch.save(dict(summary=row,input_gradient=result['gradient'][local,0]),a.output/f'{name}_{case}.pt')
            rows.append(row);print(json.dumps(row),flush=True)
        del wrapper,encoder,model;torch.cuda.empty_cache()
    (a.output/'results.json').write_text(json.dumps(dict(rows=rows,allocation=os.environ['SLURM_JOB_ID'],
        step=os.environ.get('SLURM_STEP_ID'),gpu=torch.cuda.get_device_name(),training_updates=0,
        caveat='Inference readout intervention, not retraining. Removing padding from pooling does not prevent contextual mixing among image tokens.'),indent=2)+'\n')


if __name__=='__main__':main()
