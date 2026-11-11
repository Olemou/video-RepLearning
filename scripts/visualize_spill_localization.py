#!/usr/bin/env python3
"""Show a joint checkpoint's local scores and actual trajectory sensitivities.

The local field is supervised FG/BG prediction; the other field explains the
same checkpoint's trajectory prototype decision, conditional on cached pre23.
Neither is presented as verified pixel segmentation or raw attention.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
import torch
from vision_model.training.spill_localization import SpillPatchHead
from vision_model.training.trajectory import TrajectoryModel
from vision_model.interpretability.spill_localization import explain_trajectory_spatial
from vision_model.interpretability.map_geometry import (inverse_letterbox_map,
    nonnegative_evidence_colors,blend_rgb,spatial_evidence_overlay)
from train_spill_localization import build_student,DEFAULT_PREFIX,DEFAULT_TEACHER
from diagnose_spill_representation import digest
from visualize_trajectory_video import decode_selected_frames


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix-cache',type=Path,default=DEFAULT_PREFIX)
    parser.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    parser.add_argument('--decision-scale',type=float,help='Provide one common scale when comparing multiple sources')
    args=parser.parse_args(argv);torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('format')!='spill_localization_v1' or saved['metadata']['stage']!='joint':
        raise ValueError('Need the trained joint checkpoint, not a detached auxiliary probe')
    metadata=saved['metadata']
    automatic=metadata.get('supervision_kind')=='automatic_unverified'
    if digest(args.prefix_cache)!=metadata['prefix_cache_sha256'] or digest(args.teacher_cache)!=metadata['teacher_cache_sha256']:
        raise ValueError('Checkpoint/cache provenance mismatch')
    raw=torch.load(args.teacher_cache,map_location='cpu',weights_only=False)
    cache=torch.load(args.prefix_cache,map_location='cpu',weights_only=False,mmap=True)
    rep=torch.load(args.checkpoint.parent/'representations.pt',map_location='cpu',weights_only=False)
    if rep['ids']!=raw['ids'] or cache['ids']!=raw['ids']:raise ValueError('Source identities differ')
    i=raw['ids'].index(args.source_id);n=int(raw['valid'][i].sum());grid=tuple(cache.get('grid',(16,16)))
    head=SpillPatchHead(**metadata['head_configuration'])
    student=build_student(cache,head,train_attention=False).eval().requires_grad_(False)
    student.load_state_dict(saved['model'],strict=True)
    # The trajectory architecture is inferred from the explicitly saved config.
    cfg=saved['configuration'];original=torch.load(cfg['trajectory_checkpoint'],map_location='cpu',weights_only=False)
    if digest(Path(cfg['trajectory_checkpoint']))!=metadata['trajectory_checkpoint_sha256']:raise ValueError('Original trajectory changed')
    trajectory=TrajectoryModel(original['width'],original['hidden'],original['embedding_dim']).eval().requires_grad_(False)
    trajectory.load_state_dict(saved['trajectory'],strict=True)
    result=explain_trajectory_spatial(student,trajectory,[h[None].float() for h in cache['rows'][i]['prefixes'][0][:n]],
        raw['timestamps'][i,:n],rep['prototypes'],height=grid[0],width=grid[1])
    embedding_error=float((result['embedding']-rep['embeddings'][i]).abs().max())
    local_error=float((result['local_probabilities']-rep['local_logits'][i,0,:n].sigmoid()).abs().max())
    if max(embedding_error,local_error)>1e-4:raise ValueError('Stored predictions do not replay from this checkpoint')
    scale=args.decision_scale if args.decision_scale is not None else float(result['maps']['positive_gradient'].max())
    if not np.isfinite(scale) or scale<=0:raise ValueError('Need a positive finite decision-gradient scale')
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    record=records[args.source_id];source=Path(raw['dataset_root'])/record['video_path']
    if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source video changed')
    indices=raw['frame_indices'][i,:n].tolist()
    frames=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
    args.output.mkdir(parents=True,exist_ok=False)
    canvas=Image.new('RGB',(1280,55+n*260),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,3),f'{args.source_id}: actual joint trajectory margin {result["margin"]:.5f}; no mask accuracy claim',fill='black')
    for column,title in enumerate(('Original','Automatic-candidate score' if automatic else 'Verified-point local score','Local score overlay','Trajectory conditional gradient')):
        draw.text((column*320+5,28),title,fill='black')
    for t,index in enumerate(indices):
        original_image=Image.fromarray(frames[index]).resize((320,240),Image.Resampling.BILINEAR)
        pixels=torch.from_numpy(np.asarray(original_image).copy())
        local_grid=result['local_probabilities'][t].reshape(grid)
        local=inverse_letterbox_map(local_grid,240,320,raw.get('checkpoint_image_size',256),interpolation='nearest')
        colors=nonnegative_evidence_colors(local,scale=1.)
        overlay=blend_rgb(pixels,colors,.4)
        decision=spatial_evidence_overlay(pixels,result['maps']['positive_gradient'][t].reshape(grid),scale=scale,
            scope='conditional_attention_key_gradient',signed=False,style='blue_red',checkpoint_size=raw.get('checkpoint_image_size',256))
        for column,image in enumerate((original_image,Image.fromarray(colors.numpy()),Image.fromarray(overlay.numpy()),Image.fromarray(decision.numpy()))):
            canvas.paste(image,(column*320,55+t*260))
        draw.text((5,55+t*260+242),f'Frame {index}, {index/record["fps"]:.3f}s; local scale [0,1], gradient scale {scale:.6g}; distinct mechanisms',fill='black')
    canvas.save(args.output/'comparison.png')
    torch.save({**result,'source_id':args.source_id,'checkpoint_sha256':digest(args.checkpoint)},args.output/'maps.pt')
    (args.output/'metadata.json').write_text(json.dumps(dict(source_id=args.source_id,source_sha256=raw['video_sha256'][i],
        checkpoint_sha256=digest(args.checkpoint),frame_indices=indices,timestamps=[j/record['fps'] for j in indices],
        local_field=('Sigmoid trained on UNVERIFIED automatic candidates' if automatic else 'Sigmoid trained on verified liquid/background points')+'; not a validated mask or causal explanation',
        supervision_kind=metadata.get('supervision_kind','verified_points'),
        target=result['target'],scope=result['attribution_scope'],decision_scale=scale,
        decision_scale_policy='provided comparison scale' if args.decision_scale is not None else 'single-source maximum',
        local_scale=[0,1],interpolation='nearest native cells',embedding_replay_max_error=embedding_error,
        local_probability_replay_max_error=local_error,localization_validated=False),indent=2)+'\n')


if __name__=='__main__':main()
