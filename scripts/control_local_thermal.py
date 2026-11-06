#!/usr/bin/env python3
"""Recompute local features under input removal and distant-context changes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from vision_model.training.local_thermal import LocalThermalReadout,local_relative_features
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from vision_model.models.dense_encoder import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',default='joint');p.add_argument('--output-name',default='input_controls');a=p.parse_args()
    torch.set_num_threads(4);ctx=initialize_distributed();out=a.root/a.output_name
    if ctx.is_main:out.mkdir(exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.root/'prepared/cache.pt',map_location='cpu',mmap=True,weights_only=False)
    ck=torch.load(a.root/a.case/'checkpoint.pt',map_location='cpu',weights_only=False)
    m=LocalThermalReadout(**ck['metadata']['configuration']).to(ctx.device);m.load_state_dict(ck['model'],strict=True);m.eval()
    original=torch.load(a.root/a.case/'maps.pt',map_location='cpu',weights_only=False)
    threshold=json.loads((a.root/a.case/'report.json').read_text())['calibration']['clean_q995']
    val=c['validation_indices'].tolist();positions=list(range(ctx.rank,len(val),ctx.world_size));_,counts=audit_distributed_indices(positions,len(val))
    rows=[]
    @torch.no_grad()
    def forward(pixels):
        # Match the actual FP16 cache boundary before the FP32 readout. Larger
        # learned weights expose otherwise small feature-quantization errors.
        features=local_relative_features(pixels).half().float()
        # Saved maps are evaluated as two 12-frame views. Match that batch
        # shape as well: cuDNN can select a different TF32 kernel for batch12.
        return m(torch.cat((features,features),0))[:len(features)]
    for pos in positions:
        i=val[pos];x=c['gray'][i].to(ctx.device).float()/255
        base=forward(x);error=float((base.cpu()-original[i,0]).abs().max())
        if error>.003:raise RuntimeError(f'Cached-feature replay failed: {error}')
        h,w=base.shape[-2:];n=max(1,int(h*w*.1));mean=base.mean(0).flatten()
        choices={'high':mean.argsort(descending=True)[:n],
            **{f'random{j}':torch.randperm(h*w,generator=torch.Generator().manual_seed(239+17*i+j))[:n].to(ctx.device) for j in range(3)}}
        mass=lambda z:float((z-threshold).clamp_min(0).mean())
        baseline=mass(base);controls={}
        for name,idx in choices.items():
            mask=torch.zeros(h*w,device=ctx.device);mask[idx]=1;mask=F.interpolate(mask.reshape(1,1,h,w),size=x.shape[-2:],mode='nearest').bool()
            for fill in ['mean','blur']:
                replacement=x.mean((-2,-1),keepdim=True).expand_as(x) if fill=='mean' else F.avg_pool2d(F.pad(x,(16,16,16,16),mode='reflect'),33,stride=1)
                changed=torch.where(mask,replacement,x)
                controls[name+'_'+fill]=baseline-mass(forward(changed))
        peak=int(mean.argmax());yy,xx=divmod(peak,w)
        protected=torch.zeros_like(x,dtype=torch.bool)
        radius=32+m.extra_receptive_radius
        protected[:,:,max(0,yy*8-radius):min(x.shape[-2],(yy+1)*8+radius),max(0,xx*8-radius):min(x.shape[-1],(xx+1)*8+radius)]=True
        distant=torch.where(protected,x,1-x)
        delta=float((forward(distant)[:,yy,xx]-base[:,yy,xx]).abs().max())
        if delta>2e-5:raise RuntimeError('Global context leaked into bounded-field score')
        style=float((forward(.7*x+.1)-base).abs().mean())
        rows.append(dict(id=c['ids'][i],label=int(c['labels'][i]),replay_error=error,selected_cells=n,total_cells=h*w,
            evidence_mass=baseline,removal_drop=controls,distant_context_selected_cell_error=delta,
            affine_style_mean_absolute_score_change=style,peak_cell=[yy,xx],peak_source_coordinate_resized=[xx*8+4,yy*8+4]))
        print(json.dumps(dict(rank=ctx.rank,id=c['ids'][i],distant_error=delta)),flush=True)
    pieces=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(rows,pieces,dst=0)
    else:pieces=[rows]
    if ctx.is_main:
        rows=sorted([r for part in pieces for r in part],key=lambda r:r['id'])
        if len(rows)!=len(val) or len({r['id'] for r in rows})!=len(val):raise RuntimeError('Repeated evaluation')
        summary={};rng=np.random.default_rng(239)
        for fill in ['mean','blur']:
            values=np.array([r['removal_drop']['high_'+fill]-np.mean([r['removal_drop'][f'random{j}_{fill}'] for j in range(3)]) for r in rows if r['label']==1])
            means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            summary[fill]=dict(cold_sources=len(values),high_minus_random_evidence_mass_drop=float(values.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist())
        (out/'report.json').write_text(json.dumps(dict(metadata=dict(world_size=ctx.world_size,rank_source_counts=counts,
            device=torch.cuda.get_device_name() if ctx.device.type=='cuda' else 'CPU',checkpoint_sha256=file_sha256(a.root/a.case/'checkpoint.pt'),
            code_sha256=file_sha256(__file__),input='Freshly recomputed features from actual saved grayscale detector-input pixels',
            scope='Local detector sensitivity, not original trajectory attribution or ground-truth localization'),summary=summary,rows=rows),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
