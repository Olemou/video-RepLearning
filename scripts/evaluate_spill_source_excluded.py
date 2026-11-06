#!/usr/bin/env python3
"""Read-only evaluation of every byte-unique source outside the original80."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from attribute_experiment81 import load_case
from control_experiment81 import transform_bank
from prepare_experiment81 import digest
from diagnose_spill_representation import recognition
from vision_model.data.video_dataset import VideoManifestDataset
from vision_model.data.sampling import TemporalSamplingConfig
from vision_model.data.transforms import OfficialVJEPA2Transform
from vision_model.training.online_trajectory import OnlinePrefixEncoder,collect_prefix_features
from vision_model.training.relative_clock import trajectory_model_for_clock


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--relative',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocation',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available() or os.environ.get('SLURM_JOB_ID')!=a.allocation:
        raise RuntimeError('Use the verified Slurm allocation with srun')
    if torch.cuda.device_count()!=1:raise RuntimeError('This diagnostic uses the verified one-GPU allocation')
    a.output.mkdir(parents=True,exist_ok=False);device=torch.device('cuda',0)
    c=torch.load(a.baseline/'prepared/prepared.pt',map_location='cpu',weights_only=False,mmap=True)
    dataset=json.loads(a.audit.read_text());known=set(c['source_sha256']);selected={}
    for row in dataset['source_rows']:
        if row['sha256'] not in known:selected.setdefault(row['sha256'],row)
    rows=list(selected.values())
    if len(rows)!=76:raise ValueError('Expected the audited76 unique source-excluded videos')
    if any(r['video_id'] in c['ids'] for r in rows):raise ValueError('Source ID overlap')
    ds=VideoManifestDataset(dataset['manifest'],sampling=TemporalSamplingConfig(clip_length=12,target_fps=4,training=False))
    lookup={r.video_id:i for i,r in enumerate(ds.records)}
    encoder,model,reps,proto,settings=load_case(a.baseline,'frozen',device)
    transform=OfficialVJEPA2Transform(settings['teacher_metadata']['encoder_dir'],preserve_field_of_view=True)
    pixels=[];times=[];indices=[];valid=[]
    for row in rows:
        if digest(ds.dataset_root/row['video_path'])!=row['sha256']:raise ValueError('Source hash changed')
        item=ds[lookup[row['video_id']]]
        mask=item['temporal_valid_mask'].reshape(6,2).all(1)
        if not mask.all():raise ValueError('This fixed observed-window audit excludes padded tails')
        px=transform_bank(item['video'],[],transform,include_augmentations=False)
        # Exact reconstruction of the existing uint8 normalized boundary.
        mean=torch.tensor(c['normalization']['mean'])[None,None,:,None,None]
        std=torch.tensor(c['normalization']['std'])[None,None,:,None,None]
        # Store already normalized BF16 values to avoid an inverse-normalization rounding change.
        pixels.append(px);times.append(item['frame_indices'][1::2].float()/item['source_fps']);valid.append(mask);indices.append(item['frame_indices'])
    pixels=torch.stack(pixels);times=torch.stack(times);valid=torch.stack(valid)
    features={}
    for case in ('frozen','full'):
        if case=='full':
            del encoder,model
            torch.cuda.empty_cache()
            encoder,model,reps,proto,settings=load_case(a.baseline,'full',device)
        wrapper=OnlinePrefixEncoder(encoder)
        values=[]
        for begin in range(0,len(rows),2):
            values.append(collect_prefix_features(wrapper,pixels[begin:begin+2],valid[begin:begin+2],device,chunk_sources=2).cpu())
            print(json.dumps(dict(stage='encode',case=case,completed=min(begin+2,len(rows)),total=len(rows),allocation=os.environ['SLURM_JOB_ID'])),flush=True)
        features[case]=torch.cat(values)
    del encoder,model,wrapper
    torch.cuda.empty_cache()
    labels=torch.tensor([r['label'] for r in rows]);results={};payload={}
    for arm,root,case,feature_key in [('frozen_absolute',a.baseline,'frozen','frozen'),('full_absolute',a.baseline,'full','full'),('frozen_elapsed',a.relative,'frozen','frozen')]:
        saved=torch.load(root/case/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        reps=torch.load(root/case/'representations.pt',map_location='cpu',weights_only=False)
        model=trajectory_model_for_clock(1024,saved['settings'].get('clock_policy','absolute')).to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['trajectory'],strict=True)
        train=reps['train_indices'];selection=reps['selection_indices']
        prototypes=torch.stack([torch.nn.functional.normalize(reps['embeddings'][train,0][reps['labels'][train]==label].mean(0),dim=0) for label in (0,1)])
        # Replay the known selection batch before accepting additional-source evaluation.
        def forward(f,v,t):
            n,views,length,width=f.shape
            return model(f.flatten(0,1).to(device),v[:,None].expand(n,views,length).flatten(0,1).to(device),
                t[:,None].expand(n,views,length).flatten(0,1).to(device),detach_teacher=False)['embedding'].reshape(n,views,-1).cpu()
        baseline=forward(reps['features'][selection],reps['valid'][selection],reps['timestamps'][selection])
        error=float((baseline-reps['embeddings'][selection]).abs().max())
        if error>1e-6:raise RuntimeError(f'Known head replay failed: {arm},{error}')
        details={}
        for clock in ('actual','relative_zero'):
            embeddings=[]
            for begin in range(0,len(rows),16):
                count=min(16,len(rows)-begin)
                f=reps['features'][selection].clone();v=reps['valid'][selection].clone();t=reps['timestamps'][selection].clone()
                f[:count]=features[feature_key][begin:begin+count];v[:count]=valid[begin:begin+count];t[:count]=times[begin:begin+count]
                if clock=='relative_zero':t[:count]-=t[:count,:1].clone()
                embeddings.append(forward(f,v,t)[:count,0])
            embedding=torch.cat(embeddings)
            all_embeddings=torch.cat((reps['embeddings'][train,0],embedding))
            all_labels=torch.cat((reps['labels'][train],labels))
            metrics=recognition(all_embeddings,all_labels,torch.arange(len(train)),torch.arange(len(train),len(train)+len(rows)),exclude_self=False,prototypes=prototypes)
            details[clock]=metrics;payload[arm+'_'+clock]=embedding
        results[arm]=dict(checkpoint_sha256=digest(root/case/'checkpoint.pt'),known_selection_replay_error=error,metrics=details)
        print(json.dumps(dict(stage='metrics',arm=arm,actual_accuracy=details['actual']['accuracy'],actual_auc=details['actual']['roc_auc'],relative_clock_accuracy=details['relative_zero']['accuracy'])),flush=True)
        del model,saved
        torch.cuda.empty_cache()
    torch.save(dict(ids=[r['video_id'] for r in rows],source_sha256=[r['sha256'] for r in rows],labels=labels,
        features=features,embeddings=payload,timestamps=times,frame_indices=torch.stack(indices),valid=valid),a.output/'representations.pt')
    (a.output/'results.json').write_text(json.dumps(dict(source_count=len(rows),sources=rows,models=results,
        protocol='One fixed centered12-frame window per source, all outside original80 IDs and byte hashes; frozen original training prototypes',
        scene_excluded=False,previous_research_exposure_unknown=True,independent_final_test_claim=False,
        acquisition_groups_are_not_scene_ids=True,allocation=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),
        host=os.uname().nodename,gpu=torch.cuda.get_device_name(),training_updates=0,manual_spatial_annotations=0),indent=2)+'\n')


if __name__=='__main__':main()
