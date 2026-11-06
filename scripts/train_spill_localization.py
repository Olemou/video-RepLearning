#!/usr/bin/env python3
"""Verified FG/BG probe or shared-suffix trajectory pilot, without MIL/decoder.

Uses verified points OR an explicitly selected unverified automatic-candidate
cache. The latter requires no manual annotations and is reported separately.
--validate-only reports readiness without fitting. Originals are read only.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vision_model.data.spill_points import annotation_template, load_spill_points, require_training_points
from vision_model.models.encoder import Block
from vision_model.training.distributed import initialize_distributed, audit_distributed_indices
from vision_model.training.trajectory import TrajectoryModel, trajectory_loss_from_output
from vision_model.training.dense_spill import candidate_coverage, require_candidate_coverage
from vision_model.training.spill_localization import (SpillPatchHead, SpillPrefixStudent,
    foreground_background_loss, localized_trajectory_loss)
from diagnose_spill_representation import digest, recognition
from train_spatial_semantic import DEFAULT_SPATIAL, DEFAULT_TEACHER, EXPERIMENT

DEFAULT_PREFIX = ROOT/'reports/experiments/final_attention_80_sources_20261021/cache_verified/pre23.pt'


def build_student(cache, head, *, train_attention):
    architecture = cache.get('suffix_architecture', dict(width=1024, heads=16, mlp_ratio=4., rope_dim=16))
    block = Block(architecture['width'], architecture['heads'], architecture['mlp_ratio'], architecture['rope_dim'])
    student = SpillPrefixStudent(block, torch.nn.LayerNorm(architecture['width'], eps=1e-6),
                                 train_attention=train_attention, head=head)
    student.load_state_dict({**cache['suffix_state'], **{'local_head.'+k: v for k,v in head.state_dict().items()}}, strict=True)
    return student


def point_metrics(logits, targets):
    active = targets >= 0
    scores = logits[active].sigmoid().double().cpu()
    truth = targets[active].cpu()
    result = dict(verified_cells=len(truth), accuracy=None, foreground_recall=None,
                  background_recall=None, roc_auc=None, average_precision=None)
    if len(truth):
        result['accuracy'] = float(((scores >= .5) == truth).double().mean())
        for label, key in ((0, 'background_recall'), (1, 'foreground_recall')):
            if (truth == label).any():
                result[key] = float(((scores[truth == label] >= .5) == label).double().mean())
        if len(truth.unique()) == 2:
            from sklearn.metrics import roc_auc_score, average_precision_score
            result.update(roc_auc=float(roc_auc_score(truth, scores)),
                          average_precision=float(average_precision_score(truth, scores)))
    result['scope'] = 'Identity-view verified point cells only; not mask IoU or independent point samples'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('probe', 'joint'), default='probe')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--pseudo-cache', type=Path, help='Opt-in automatic candidates from benchmark_normal_spill.py; NOT verified labels')
    parser.add_argument('--pseudo-case', default='block5_raw')
    parser.add_argument('--trajectory-only', action='store_true', help='Joint control: optimize original four terms only; local probe head is monitored')
    parser.add_argument('--write-template', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--teacher-cache', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--prefix-cache', type=Path, default=DEFAULT_PREFIX)
    parser.add_argument('--spatial-cache', type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument('--trajectory-checkpoint', type=Path, default=EXPERIMENT/'training/trial_01_checkpoint.pt')
    parser.add_argument('--probe-checkpoint', type=Path)
    parser.add_argument('--layer', type=int, choices=(5, 11, 17, 23), default=23)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=239)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--attention-lr', type=float, default=3e-6)
    parser.add_argument('--fg-weight', type=float, help='Required for joint training; calibrate with training-only data')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    for value in (args.head_lr, args.attention_lr):
        if not math.isfinite(value) or value <= 0:
            parser.error('Learning rates must be positive and finite')
    if args.updates < 1 or args.eval_every < 1:
        parser.error('Budgets must be positive')
    automatic = args.pseudo_cache is not None
    if args.trajectory_only and args.stage!='joint':parser.error('--trajectory-only is a joint control')
    if automatic and args.annotations is not None:
        parser.error('Choose automatic candidates OR verified points, never mix their provenance')
    if not args.write_template and not automatic and (args.annotations is None or not args.annotations.is_file()):
        parser.error('Supply verified --annotations; no foreground/background labels are inferred from video names')
    raw = torch.load(args.teacher_cache, map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.trajectory_checkpoint, map_location='cpu', weights_only=False)
    if raw['temporal_context_policy'] != 'past_only_prefix_last_tubelet_v1':
        raise ValueError('Require separately encoded past-only prefixes')
    records_list = [json.loads(line) for line in Path(raw['manifest']).read_text().splitlines() if line.strip()]
    records = {r['video_id']: r for r in records_list}
    if len(records) != len(records_list):
        raise ValueError('Duplicate manifest source IDs')
    train = torch.tensor([raw['ids'].index(name) for name in checkpoint['train_source_ids']])
    val = torch.tensor([raw['ids'].index(name) for name in checkpoint['validation_source_ids']])
    if sorted(train.tolist()+val.tolist()) != list(range(len(raw['ids']))):
        raise ValueError('Source split omitted or repeated a video')
    if args.write_template:
        # Template creation is a rank-zero standalone operation, not training.
        import os
        if int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0'))) == 0:
            args.write_template.parent.mkdir(parents=True, exist_ok=True)
            with args.write_template.open('x') as stream:
                json.dump(annotation_template(raw, records), stream, indent=2)
            print(json.dumps(dict(template=str(args.write_template),generated_labels=0,ready_for_training=False)))
        return
    cache = torch.load(args.prefix_cache, map_location='cpu', weights_only=False, mmap=True)
    grid = tuple(cache.get('grid', (16, 16)));size = raw.get('checkpoint_image_size', 256)
    if automatic:
        candidate=torch.load(args.pseudo_cache,map_location='cpu',weights_only=False)
        candidate_meta=json.loads(args.pseudo_cache.with_name('results.json').read_text())
        if (candidate['ids']!=raw['ids'] or not torch.equal(candidate['valid'],raw['valid'])
            or not torch.equal(candidate['timestamps'],raw['timestamps'])
            or not torch.equal(candidate['train_indices'],train) or not torch.equal(candidate['validation_indices'],val)
            or candidate_meta['metadata']['teacher_cache_sha256']!=digest(args.teacher_cache)
            or candidate_meta['metadata']['trajectory_checkpoint_sha256']!=digest(args.trajectory_checkpoint)
            or candidate_meta['metadata']['source_sha256']!=raw['video_sha256']):
            raise ValueError('Automatic-candidate data/split/provenance mismatch')
        reference_sources=candidate_meta['cases'][args.pseudo_case]['normal_sources']
        expected_clean={raw['ids'][i] for i in train if raw['labels'][i]==0}
        if set(reference_sources)!=expected_clean:
            raise ValueError('Automatic reference must contain exactly training clean sources')
        targets=candidate['pseudo_targets'][args.pseudo_case].clone()
        content=candidate['content'].reshape(len(raw['ids']),*grid)
        if targets.shape!=(*raw['features'].shape[:-1],grid[0]*grid[1]):raise ValueError('Wrong automatic candidate shape')
        if targets.dtype!=torch.long or not ((targets==-1)|(targets==0)|(targets==1)).all():
            raise ValueError('Automatic candidates must be long -1/0/1')
        eligible=raw['valid'][:,None,:,None]&content.flatten(1)[:,None,None,:]
        if (targets.masked_select(~eligible.expand_as(targets))!=-1).any():
            raise ValueError('Automatic candidates cannot label padding')
        if (targets[raw['labels']==0]==1).any():raise ValueError('Automatic candidates cannot label clean foreground')
        coverage=dict(supervision='automatic_unverified',manual_annotations=0,reference_clean_sources=len(reference_sources),
            foreground_candidate_cells=int((targets[:,0]==1).sum()),background_candidate_cells=int((targets[:,0]==0).sum()),
            all_other_positions='ignored; positive episode labels never become dense patch labels')
        coverage['within_spill']=candidate_coverage(targets,raw['labels'],train)
    else:
        targets, content, coverage = load_spill_points(args.annotations, raw, records, size=size, grid=grid)
    readiness = dict(coverage=coverage,train_sources=len(train),validation_sources=len(val),
        training_foreground=int((targets[train, 0] == 1).sum()),
        training_background=int((targets[train, 0] == 0).sum()),
        validation_foreground=int((targets[val, 0] == 1).sum()),
        validation_background=int((targets[val, 0] == 0).sum()))
    readiness['ready_for_training'] = readiness['training_foreground'] > 0 and readiness['training_background'] > 0
    if automatic:
        readiness['ready_for_training'] &= coverage['within_spill']['ready_for_training']
    if args.validate_only:
        print(json.dumps(readiness, indent=2))
        if not readiness['ready_for_training']:
            raise SystemExit(2)
        return readiness
    if automatic:
        require_candidate_coverage(targets,raw['labels'],train)
        if not readiness['ready_for_training']:
            raise ValueError('Automatic seed generator abstained or lacks a class; no manual annotations are required, but this candidate set cannot fit a two-class local objective')
    else:require_training_points(targets, train)
    if args.output is None:
        parser.error('Training requires a new --output directory')
    if args.stage == 'joint' and (args.layer != 23 or args.probe_checkpoint is None or args.fg_weight is None or not math.isfinite(args.fg_weight) or args.fg_weight <= 0):
        parser.error('Joint training requires layer 23, --probe-checkpoint and an explicit positive --fg-weight')
    teacher_hash = digest(args.teacher_cache);original_hash = digest(args.trajectory_checkpoint)
    if checkpoint['cache_sha256'] != teacher_hash or cache['teacher_cache_sha256'] != teacher_hash:
        raise ValueError('Teacher/trajectory/prefix provenance mismatch')
    if (cache['ids'] != raw['ids'] or not torch.equal(cache['labels'],raw['labels'])
        or not torch.equal(cache['valid'],raw['valid']) or not torch.equal(cache['timestamps'],raw['timestamps'])
        or cache['views'] != ['identity','mirror'] or cache['encoder_sha256'] != raw['checkpoint_sha256']
        or cache['temporal_context_policy'] != raw['temporal_context_policy']
        or not torch.equal(cache['train_indices'],train) or not torch.equal(cache['validation_indices'],val)):
        raise ValueError('Cache identities, split, views, timestamps or encoder mismatch')
    for i, name in enumerate(raw['ids']):
        if digest(Path(raw['dataset_root'])/records[name]['video_path']) != raw['video_sha256'][i]:
            raise ValueError('Source changed since feature extraction')
    annotation_hash = digest(args.pseudo_cache if automatic else args.annotations)
    torch.manual_seed(args.seed)
    head = SpillPatchHead(checkpoint['width'])
    if args.probe_checkpoint:
        probe = torch.load(args.probe_checkpoint,map_location='cpu',weights_only=False)
        if (probe['format'] != 'spill_localization_v1' or probe['metadata']['stage'] != 'probe'
            or probe['metadata']['layer'] != 23 or probe['metadata'].get('supervision_sha256',probe['metadata']['annotation_sha256']) != annotation_hash
            or probe['metadata'].get('supervision_kind','verified_points') != ('automatic_unverified' if automatic else 'verified_points')
            or probe['metadata']['teacher_cache_sha256'] != teacher_hash
            or probe['metadata']['train_source_ids'] != checkpoint['train_source_ids']):
            raise ValueError('Probe must fit these training annotations, split and final-layer features')
        head = SpillPatchHead(**probe['metadata']['head_configuration'])
        head.load_state_dict(probe['local_head'],strict=True)
    if head.configuration['width'] != checkpoint['width']:
        raise ValueError('Local readout width differs from the preserved trajectory input')
    spatial = None
    if args.layer != 23:
        spatial = torch.load(args.spatial_cache,map_location='cpu',weights_only=False,mmap=True)
        if (spatial['ids'] != raw['ids'] or spatial['cache_sha256'] != teacher_hash
            or spatial['encoder_sha256'] != raw['checkpoint_sha256']):
            raise ValueError('Intermediate spatial cache provenance mismatch')
    context = initialize_distributed()
    local_train = train[context.rank::context.world_size].tolist()
    assigned = sorted(local_train+val[context.rank::context.world_size].tolist())
    _, rank_counts = audit_distributed_indices(assigned,len(raw['ids']))
    if len(train) < context.world_size:
        raise ValueError('Every rank needs a distinct training source')
    if context.is_main:
        args.output.mkdir(parents=True,exist_ok=False)
    if context.world_size > 1:dist.barrier()
    trajectory = TrajectoryModel(checkpoint['width'],checkpoint['hidden'],checkpoint['embedding_dim']).to(context.device).eval().requires_grad_(False)
    trajectory.load_state_dict(checkpoint['model'],strict=True)
    model = (build_student(cache,head,train_attention=True) if args.stage == 'joint' else head).to(context.device).eval()
    initial_frozen = {name:p.detach().cpu().clone() for name,p in model.named_parameters() if not p.requires_grad}
    wrapped = DDP(model,device_ids=[context.device.index] if context.device.type=='cuda' else None,
                  broadcast_buffers=False) if context.world_size>1 else model
    groups = [dict(params=(model.local_head if args.stage=='joint' else model).parameters(),lr=args.head_lr)]
    if args.stage=='joint':groups.append(dict(params=model.blocks[0].attn.parameters(),lr=args.attention_lr))
    def make_optimizer():return torch.optim.AdamW(groups,weight_decay=0.,eps=1e-12)
    optimizer = make_optimizer()

    def compute(student, i, t):
        if args.stage == 'joint':
            hidden = torch.stack([cache['rows'][i]['prefixes'][view][t] for view in (0,1)]).to(context.device)
            return student(hidden,frames=t+1,height=grid[0],width=grid[1])
        features = (cache['rows'][i]['reference'][:,t] if spatial is None else
                    spatial['tokens'][i,:,t,spatial['layers'].index(args.layer)]).to(context.device).float()
        return dict(student_features=raw['features'][i,:,t].to(context.device),local_logits=student(features))

    @torch.no_grad()
    def collect():
        rows=[]
        for i in assigned:
            means=torch.zeros(2,raw['valid'].shape[1],checkpoint['width'])
            logits=torch.zeros(2,raw['valid'].shape[1],grid[0]*grid[1])
            for t in range(int(raw['valid'][i].sum())):
                result=compute(model,i,t)
                means[:,t]=result['student_features'].cpu();logits[:,t]=result['local_logits'].cpu()
            rows.append((i,means,logits))
        parts=[None]*context.world_size if context.is_main else None
        if context.world_size>1:dist.gather_object(rows,parts,dst=0)
        else:parts=[rows]
        if not context.is_main:return None
        ordered=sorted([row for part in parts for row in part],key=lambda row:row[0])
        if [r[0] for r in ordered]!=list(range(len(raw['ids']))):raise RuntimeError('Repeated/missing evaluated source')
        return torch.stack([r[1] for r in ordered]),torch.stack([r[2] for r in ordered])

    @torch.no_grad()
    def trajectory_output(features):
        mask=raw['valid'][:,None].expand(-1,2,-1).flatten(0,1).to(context.device)
        times=raw['timestamps'][:,None].expand(-1,2,-1).flatten(0,1).to(context.device)
        return {k:v.cpu() for k,v in trajectory(features.flatten(0,1).to(context.device),mask,times).items()}

    original_output=trajectory_output(raw['features']) if context.is_main else None
    history=[];evaluations=[];initial_output=None
    @torch.no_grad()
    def evaluate(update):
        nonlocal initial_output
        collected=collect()
        if not context.is_main:return None
        means,logits=collected;output=trajectory_output(means)
        if initial_output is None:initial_output={k:v.clone() for k,v in output.items()}
        embeddings=output['embedding'].reshape(len(raw['ids']),2,-1)[:,0]
        result=dict(update=update,
            original_state_max_drift=float((output['states']-original_output['states']).abs().max()),
            matched_initial_state_max_drift=float((output['states']-initial_output['states']).abs().max()),
            original_embedding_max_drift=float((output['embedding']-original_output['embedding']).abs().max()))
        for split,indices in (('train',train),('validation',val)):
            selected=(indices[:,None]*2+torch.arange(2)[None]).flatten()
            subset={k:v[selected].to(context.device) for k,v in output.items()}
            loss_inputs=[raw[key][indices].to(context.device) for key in ('features','valid','timestamps','labels')]
            _,details,_=trajectory_loss_from_output(trajectory,*loss_inputs,subset,epoch=20,curriculum_epochs=20)
            spatial_metrics=point_metrics(logits[indices,0],targets[indices,0])
            if automatic:
                spatial_metrics['candidate_cells']=spatial_metrics.pop('verified_cells')
                spatial_metrics['scope']='Agreement with automatic UNVERIFIED candidates only; not spill-localization accuracy'
            result[split]=dict(classification=recognition(embeddings,raw['labels'],train,indices,exclude_self=split=='train'),
                **({'pseudo_candidate_agreement':spatial_metrics} if automatic else {'points':spatial_metrics}),
                trajectory_losses={k:float(details[k]) for k in ('total_loss','semantic_loss','alignment_loss','prediction_loss','retention_loss','persistence_loss')})
            horizons={}
            for horizon in (1,2,4):
                valid_pairs=raw['valid'][indices,:-horizon]&raw['valid'][indices,horizon:] if horizon<raw['valid'].shape[1] else torch.tensor([])
                if valid_pairs.any():
                    _,d,_=trajectory_loss_from_output(trajectory,*loss_inputs,subset,epoch=20,curriculum_epochs=20,horizons=(horizon,))
                    horizons[str(horizon)]=dict(prediction=float(d['prediction_loss']),persistence=float(d['persistence_loss']))
            result[split]['horizons']=horizons
        evaluations.append(result)
        torch.save(dict(ids=raw['ids'],labels=raw['labels'],valid=raw['valid'],timestamps=raw['timestamps'],
            embeddings=embeddings,prototypes=torch.stack([F.normalize(embeddings[train][raw['labels'][train]==label].mean(0),dim=0) for label in (0,1)]),
            local_logits=logits,content=content,train_indices=train,validation_indices=val,
            scope=('Automatic-candidate local scores; NOT verified liquid probabilities' if automatic else 'Supervised local FG/BG scores')+'; not original-decision attribution or raw self-attention'),args.output/'representations.pt')
        return result

    evaluate(0)
    if context.world_size>1:dist.barrier()
    exposures=torch.zeros(len(raw['ids']),2,raw['valid'].shape[1],dtype=torch.long,device=context.device)
    local_valid=raw['valid'][local_train].to(context.device);local_targets=targets[local_train].to(context.device)
    for update in range(1,args.updates+1):
        if update==21:optimizer=make_optimizer()
        with torch.no_grad():
            means=torch.zeros(len(local_train),2,raw['valid'].shape[1],checkpoint['width'],device=context.device)
            logits=torch.zeros(len(local_train),2,raw['valid'].shape[1],grid[0]*grid[1],device=context.device)
            calls=[]
            for j,i in enumerate(local_train):
                for t in range(int(raw['valid'][i].sum())):
                    result=compute(model,i,t);means[j,:,t]=result['student_features'];logits[j,:,t]=result['local_logits'];calls.append((j,i,t))
        means.requires_grad_(True);logits.requires_grad_(True)
        if args.stage=='joint':
            loss,details,_=localized_trajectory_loss(trajectory,raw['features'][local_train].to(context.device),means,logits,
                local_targets,local_valid,raw['timestamps'][local_train].to(context.device),raw['labels'][local_train].to(context.device),
                fg_weight=args.fg_weight,epoch=update)
            if args.trajectory_only:
                loss=details['trajectory_loss'];details['joint_loss']=loss
        else:
            loss,_=foreground_background_loss(logits,local_targets);details=dict(foreground_background_loss=loss)
        loss.backward()
        grad_means=means.grad if means.grad is not None else torch.zeros_like(means)
        grad_logits=logits.grad if logits.grad is not None else torch.zeros_like(logits)
        optimizer.zero_grad(set_to_none=True)
        for position,(j,i,t) in enumerate(calls):
            sync=contextlib.nullcontext() if position==len(calls)-1 or context.world_size==1 else wrapped.no_sync()
            with sync:
                result=compute(wrapped,i,t)
                ((result['student_features']*grad_means[j,:,t]).sum()+(result['local_logits']*grad_logits[j,:,t]).sum()).backward()
            exposures[i,:,t]+=1
        gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
        optimizer.step()
        values=torch.stack([loss.detach(),details['foreground_background_loss'].detach()])
        if context.world_size>1:dist.all_reduce(values);values/=context.world_size
        history.append(dict(update=update,loss=float(values[0]),foreground_background_loss=float(values[1]),gradient_norm=float(gradient)))
        if update%args.eval_every==0 or update==args.updates:
            result=evaluate(update)
            if context.is_main:
                print(json.dumps(dict(update=update,loss=history[-1]['loss'],validation=result['validation'])),flush=True)
                (args.output/'evaluation_history.json').write_text(json.dumps(evaluations,indent=2)+'\n')
            if context.world_size>1:dist.barrier()
    drift=max([float((p.detach().cpu()-initial_frozen[n]).abs().max()) for n,p in model.named_parameters() if n in initial_frozen] or [0.])
    if drift or any(p.grad is not None for p in trajectory.parameters()):raise RuntimeError('Frozen parameters received updates/gradients')
    if digest(args.trajectory_checkpoint)!=original_hash or digest(args.teacher_cache)!=teacher_hash:
        raise RuntimeError('Original trajectory/cache changed')
    if context.world_size>1:dist.all_reduce(exposures)
    if context.is_main:
        metadata=dict(stage=args.stage,layer=args.layer,seed=args.seed,updates=args.updates,
            head_configuration=(model.local_head if args.stage=='joint' else model).configuration,
            objective=('L_trajectory only' if args.trajectory_only else 'L_trajectory + lambda_FGBG * partial_BCE') if args.stage=='joint' else 'partial_BCE; original trajectory untouched',
            fg_weight=(0. if args.trajectory_only else args.fg_weight) if args.stage=='joint' else None,
            teacher_cache_sha256=teacher_hash,trajectory_checkpoint_sha256=original_hash,annotation_sha256=None if automatic else annotation_hash,
            supervision_sha256=annotation_hash,supervision_kind='automatic_unverified' if automatic else 'verified_points',
            pseudo_case=args.pseudo_case if automatic else None,manual_annotations_used=0 if automatic else 'verified file',
            prefix_cache_sha256=digest(args.prefix_cache),train_source_ids=checkpoint['train_source_ids'],
            validation_source_ids=checkpoint['validation_source_ids'],rank_eval_source_counts=rank_counts,
            optimizer_reset_before_update=21 if args.updates>=21 else None,world_size=context.world_size,frozen_parameter_max_drift=drift,
            trajectory_frozen=True,teacher_targets_detached=True,student_input_detached=False if args.stage=='joint' else None,
            source_validation_is_selection=True,independent_scene_test=False,mil=False,decoder=False,mirror_loss=False,
            temporal_context_policy=raw['temporal_context_policy'],annotation_readiness=readiness,
            model_id=raw['model_id'],teacher_normalization=raw.get('normalization'),
            views='fixed actual identity and whole-clip mirror; no fresh augmentation',
            feature_precision=cache.get('precision','unrecorded'),
            local_feature_source='FP32 frozen block23 reference' if args.layer==23 else f'accepted block-{args.layer} cache',
            source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                'src/vision_model/training/spill_localization.py':digest(ROOT/'src/vision_model/training/spill_localization.py'),
                'src/vision_model/training/trajectory.py':digest(ROOT/'src/vision_model/training/trajectory.py'),
                'src/vision_model/data/spill_points.py':digest(ROOT/'src/vision_model/data/spill_points.py')})
        torch.save(dict(format='spill_localization_v1',model=model.state_dict(),
            local_head=(model.local_head if args.stage=='joint' else model).state_dict(),trajectory=trajectory.state_dict(),
            optimizer=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),view_time_exposures=exposures.cpu(),
            metadata=metadata,configuration=vars(args),history=history),args.output/'checkpoint.pt')
        (args.output/'report.json').write_text(json.dumps(dict(metadata=metadata,evaluations=evaluations),indent=2)+'\n')
        with (args.output/'history.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    if context.world_size>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
