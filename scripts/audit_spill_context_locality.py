#!/usr/bin/env python3
"""Local-pixel invariance versus contextual-token changes in all24 JEPA blocks."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from attribute_experiment81 import load_case,evaluation_groups,margins_for_group
from control_experiment81 import transform_bank
from prepare_experiment81 import normalize_bank,digest
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.distributed import initialize_distributed
from vision_model.training.online_trajectory import OnlinePrefixEncoder,collect_prefix_features
from vision_model.interpretability.context_controls import protected_tiles,change_distant_context


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allocation',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    ctx=initialize_distributed()
    if ctx.device.type!='cuda' or os.environ.get('SLURM_JOB_ID')!=a.allocation:
        raise RuntimeError('Run inside the verified allocation using srun; do not fake its environment')
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    names=['cold_folderImage56','cold_folderImage10','cold_folderImage13','clean_videoFrame121']
    sources=[c['ids'].index(n) for n in names]
    if not c['valid'][sources].all():raise ValueError('This study requires complete observed prefixes')
    settings=dict(allocation=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),
        host=os.uname().nodename,gpu=torch.cuda.get_device_name(ctx.device),world_size=ctx.world_size,
        sources=names,probe_tiles='All twelve fixed4x4 content tiles; measure inner2x2 patches',
        intervention_fills=['gain09','blur33'],scope='Conditional spatial feature locality, not liquid localization',
        manual_annotations=0,training_updates=0,baseline=str(a.baseline.resolve()))
    if ctx.is_main:(a.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(settings),flush=True)
    all_rows=[]
    for case in ('frozen','full'):
        encoder,model,reps,proto,training_settings=load_case(a.baseline,case,ctx.device)
        transform=OfficialVJEPA2Transform(training_settings['teacher_metadata']['encoder_dir'],preserve_field_of_view=True)
        wrapper=OnlinePrefixEncoder(encoder)
        groups=evaluation_groups(reps,training_settings)
        checkpoint_sha=digest(a.baseline/case/'checkpoint.pt')
        for i in sources[ctx.rank::ctx.world_size]:
            ids=next(g for g in groups if i in g);local=int((ids==i).nonzero()[0]);identity=local*2
            px=torch.stack([normalize_bank(c['pixels'][int(j),:2],c['normalization']) for j in ids])
            captured={}
            def capture(key):
                def hook(module,args,out):
                    if out.shape[1]==1536:captured[key]=out[identity,-256:].detach().cpu().float()
                return hook
            handles=[encoder.patch_embed.register_forward_hook(capture('patch_embed'))]
            handles += [block.register_forward_hook(capture(f'block_{l:02d}')) for l,block in enumerate(encoder.blocks)]
            handles += [encoder.norm.register_forward_hook(capture('final_norm'))]
            def forward(pixels):
                captured.clear()
                features=collect_prefix_features(wrapper,pixels,c['valid'][ids],ctx.device,chunk_sources=len(ids))
                margin=float(margins_for_group(model,reps,ids,features,proto,ctx.device)[local])
                if len(captured)!=26:raise RuntimeError('Missing captured spatial stage')
                return features,margin,dict(captured)
            original,base_margin,base_tokens=forward(px)
            error=float((original.cpu()-reps['features'][ids]).abs().max())
            if error>1e-6:raise RuntimeError(f'Original input replay mismatch: {error}')
            rows=[]
            for tile_index,tile in enumerate(protected_tiles()):
                core=tile['core'].flatten()
                pixel_core=tile['core'].repeat_interleave(16,0).repeat_interleave(16,1)
                for fill in ('gain09','blur33'):
                    rgb,raw_mask=change_distant_context(c['rgb'][i],tile['protected'],fill)
                    replacement=transform_bank(rgb,[],transform,include_augmentations=False)
                    unchanged_error=float((replacement[0,...,pixel_core].float()-px[local,0,...,pixel_core].float()).abs().max())
                    if unchanged_error!=0:raise RuntimeError('Resize changed protected core pixels')
                    modified=px.clone();modified[local]=replacement
                    _,margin,tokens=forward(modified)
                    embedding_error=float((tokens['patch_embed'][core]-base_tokens['patch_embed'][core]).abs().max())
                    if embedding_error!=0:raise RuntimeError('Protected patch embedding changed')
                    for stage in base_tokens:
                        before,after=base_tokens[stage][core],tokens[stage][core]
                        rows.append(dict(case=case,id=c['ids'][i],index=i,
                            split='train' if i in reps['train_indices'] else 'selection',label=int(c['labels'][i]),
                            tile=tile_index,row=tile['row'],column=tile['column'],fill=fill,stage=stage,
                            protected_input_error=unchanged_error,protected_patch_embed_error=embedding_error,
                            cosine_change=float((1-F.cosine_similarity(before,after,dim=-1)).mean()),
                            relative_l2_change=float((after-before).norm()/before.norm().clamp_min(1e-20)),
                            changed_original_pixel_fraction=float((~raw_mask).float().mean()),
                            baseline_margin=base_margin,changed_margin=margin,margin_drop=base_margin-margin,
                            production_replay_error=error,checkpoint_sha256=checkpoint_sha))
            for handle in handles:handle.remove()
            all_rows.extend(rows)
            print(json.dumps(dict(case=case,id=c['ids'][i],interventions=24,stages=26,production_replay_error=error)),flush=True)
        del encoder,model,wrapper
        torch.cuda.empty_cache()
    gathered=[None]*ctx.world_size if ctx.is_main else None
    if ctx.world_size>1:dist.gather_object(all_rows,gathered,dst=0)
    else:gathered=[all_rows]
    if ctx.is_main:
        rows=[r for part in gathered for r in part]
        if len(rows)!=2*4*12*2*26:raise RuntimeError('Missing source/probe/stage observations')
        (a.output/'results.json').write_text(json.dumps(dict(settings=settings,rows=rows,
            warning='Distant interventions can be out of distribution. Nonlocal feature change does not prove absence of local semantics or identify liquid.'),indent=2)+'\n')
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
