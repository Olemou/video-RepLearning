#!/usr/bin/env python3
"""Recheck every saved source and distinguish recognition from localization.

This CPU artifact audit consolidates completed, source-partitioned DDP results.
It does not refit representations, select favorable heads or invent annotations.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,trustworthiness
from sklearn.metrics import roc_auc_score,average_precision_score
from diagnose_spill_representation import digest
from train_spatial_semantic import DEFAULT_TEACHER
from verify_and_export_spill_overlays import evidence_peak


def independent_recognition(rep,indices):
    """Identity queries, training-only prototypes/gallery; no self retrieval."""
    z=rep['embeddings'][:,0].double();labels=rep['labels'];train=rep['train_indices']
    prototype=F.normalize(torch.stack([z[train][labels[train]==k].mean(0) for k in (0,1)]),dim=-1)
    scores=z[indices]@prototype.T
    margin=scores[:,1]-scores[:,0];truth=labels[indices]
    retrieval=[];ap=[]
    for index in indices:
        candidates=train[train!=index]
        order=(z[index]@z[candidates].T).argsort(descending=True,stable=True)
        relevant=labels[candidates[order]]==labels[index]
        retrieval.append(float(relevant[0]))
        precision=relevant.double().cumsum(0)/torch.arange(1,len(relevant)+1)
        ap.append(float(precision[relevant].mean()))
    return dict(sources=len(indices),accuracy=float((scores.argmax(-1)==truth).double().mean()),
        roc_auc=float(roc_auc_score(truth,margin)),class_average_precision=float(average_precision_score(truth,margin)),
        retrieval_top1=float(np.mean(retrieval)),retrieval_mean_average_precision=float(np.mean(ap)),
        training_prototype_max_error=float((prototype-rep['prototypes'].double()).abs().max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if int(os.environ.get('RANK','0'))!=0:raise RuntimeError('Consolidated reporting is rank zero only')
    args.output.mkdir(parents=True,exist_ok=False)
    experiment=args.experiment;run=experiment/'attention_lr1em4_100'
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    attention_path=run/'spatial_evaluation/attention_maps.pt'
    attention=torch.load(attention_path,map_location='cpu',weights_only=False)
    figure_manifest=json.loads((run/'figures_v2/manifest.json').read_text())
    assert digest(attention_path)==figure_manifest['attention_sha256']
    assert attention['ids']==raw['ids'] and len(set(raw['ids']))==80
    assert len(set(raw['video_sha256']))==80
    assert sum(attention['source_counts_per_rank'])==80
    assert len(attention['rows'])==160 and len({(r['case'],r['index']) for r in attention['rows']})==160
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    scale=figure_manifest['scales']['positive_gradient'];models={};representations={}
    for name in ('FT0_frozen','FT1_attention'):
        checkpoint=run/name/'checkpoint.pt'
        assert digest(checkpoint)==figure_manifest['checkpoint_sha256'][name]
        rep=torch.load(run/name/'representations.pt',map_location='cpu',weights_only=False,mmap=True)
        representations[name]=rep
        train,val=rep['train_indices'],rep['validation_indices']
        assert rep['ids']==raw['ids'] and torch.equal(rep['labels'],raw['labels'])
        assert set(train.tolist()).isdisjoint(val.tolist())
        assert sorted(train.tolist()+val.tolist())==list(range(80))
        assert len(train)==64 and len(val)==16
        metrics=json.loads((run/name/'metrics.json').read_text())
        models[name]=dict(checkpoint_sha256=digest(checkpoint),metadata=metrics['metadata'])
        for split,indices in (('train',train),('validation',val)):
            measured=independent_recognition(rep,indices)
            assert measured['training_prototype_max_error']<1e-6
            for key in ('accuracy','roc_auc','retrieval_top1','retrieval_mean_average_precision'):
                assert abs(measured[key]-metrics['final'][split][key])<1e-6
            models[name][split]=measured
    # Confirm the two arms actually have identical frozen operations.
    states=[torch.load(run/name/'checkpoint.pt',map_location='cpu',weights_only=False)['model'] for name in models]
    frozen_drift=max(float((states[0][key]-states[1][key]).abs().max()) for key in states[0]
                     if not key.startswith(('projector.','blocks.0.attn.')))
    assert frozen_drift==0
    rows=[];aggregate_error=0.;raw_probability_error=0.;target_replay_error=0.
    for row in attention['rows']:
        i=row['index'];name=row['case'];rep=representations[name]
        valid=rep['valid'][i];n=int(valid.sum());support=rep['content'][i]
        assert torch.equal(valid,torch.arange(len(valid))<n)
        assert n==6 and int(support.sum())==192
        assert row['checkpoint_replay_max_error']<2e-5 and min(row['gradient_norms'])>0
        for kind,accumulated in row['maps'].items():
            reconstructed=torch.zeros_like(accumulated)
            assert len(row['prefix_maps'][kind])==n
            for end,values in enumerate(row['prefix_maps'][kind],1):
                assert values.shape==(end,256) and torch.isfinite(values).all()
                reconstructed[:end]+=values
                if kind=='raw':raw_probability_error=max(raw_probability_error,abs(float(values.sum())-1))
            error=float((reconstructed-accumulated).abs().max())
            aggregate_error=max(aggregate_error,error)
            assert error==0
        gradients=row['maps']['positive_gradient'][:,support].double()
        clean=torch.zeros_like(row['maps']['positive_gradient'])
        for end,values in enumerate(row['clean_target_prefix_positive_gradients'],1):clean[:end]+=values
        target_cos=float(F.cosine_similarity(gradients.flatten(),clean[:,support].double().flatten(),dim=0))
        target_replay_error=max(target_replay_error,abs(target_cos-row['spill_clean_positive_gradient_cosine']))
        assert target_replay_error<1e-6
        record=records[raw['ids'][i]]
        peaks=[evidence_peak(g.reshape(16,16),support,record['height'],record['width']) for g in row['maps']['positive_gradient']]
        mass=gradients.sum(-1);distribution=mass/mass.sum()
        spatial=gradients.sum(0);spatial/=spatial.sum()
        entropy=float(-(spatial*spatial.clamp_min(1e-30).log()).sum()/np.log(len(spatial)))
        rows.append(dict(id=raw['ids'][i],index=i,case=name,label=int(rep['labels'][i]),
            split='train' if i in rep['train_indices'] else 'validation',
            gradient_min=float(gradients.min()),gradient_max=float(gradients.max()),
            fraction_above_half_shared_scale=float((gradients>scale*.5).double().mean()),
            spatial_entropy=entropy,positive_gradient_mass_by_key_time=distribution.tolist(),
            spill_clean_target_cosine=target_cos,peaks=peaks))
    assert raw_probability_error<1e-5
    summary={}
    for name in models:
        summary[name]={}
        for label in (0,1):
            subset=[r for r in rows if r['case']==name and r['label']==label]
            summary[name][str(label)]={key:float(np.mean([r[key] for r in subset])) for key in (
                'fraction_above_half_shared_scale','spatial_entropy','spill_clean_target_cosine')}
            summary[name][str(label)]['mean_mass_by_key_time']=np.mean([r['positive_gradient_mass_by_key_time'] for r in subset],axis=0).tolist()
    # Preserve the existing checkpoint-linked all-80 projection; add a clearly
    # labelled projection restricted to the 16 selection sources.
    for filename in ('tsne_80.png','tsne_80.json','loss_curves.png'):
        shutil.copyfile(run/'figures_v2'/filename,args.output/filename)
    fig,axes=plt.subplots(1,2,figsize=(11,4));coordinates={}
    for ax,(name,rep) in zip(axes,representations.items()):
        val=rep['validation_indices'];z=rep['embeddings'][val,0].numpy();labels=rep['labels'][val].numpy()
        xy=TSNE(perplexity=5,init='pca',learning_rate='auto',random_state=239,max_iter=1500).fit_transform(z)
        for label,color in ((0,'tab:blue'),(1,'tab:red')):
            ax.scatter(xy[labels==label,0],xy[labels==label,1],c=color,label='cold' if label else 'clean',s=55)
        ax.set_title(name);ax.legend();ax.set_xticks([]);ax.set_yticks([])
        coordinates[name]=dict(ids=[rep['ids'][j] for j in val],xy=xy.tolist(),
            trustworthiness=float(trustworthiness(z,xy,n_neighbors=3,metric='cosine')))
    fig.suptitle('16 source-validation selection videos only; not an independent scene test')
    fig.tight_layout();fig.savefig(args.output/'tsne_validation16.png',dpi=150);plt.close(fig)
    (args.output/'tsne_validation16.json').write_text(json.dumps(coordinates,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in models:
        history=json.loads((run/name/'evaluation_history.json').read_text())
        for split,style in (('train','-'),('validation','--')):
            for ax,key in zip(axes,('vico_loss','minimum_pair_separation','accuracy')):
                ax.plot([r['update'] for r in history],[r[split][key] for r in history],style,label=f'{name} {split}')
                ax.set_title(key);ax.set_xlabel('Spatial continuation update')
    axes[0].set_yscale('log');axes[2].set_ylim(0,1.05);axes[0].legend(fontsize=7)
    fig.suptitle('Fixed-curriculum episode evaluation: class discrimination, not local accuracy')
    fig.tight_layout();fig.savefig(args.output/'classification_curves.png',dpi=150);plt.close(fig)
    temporal=ROOT/'reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/training/loss_history.csv'
    temporal_rows=list(csv.DictReader(temporal.open()));updates=[int(r['update']) for r in temporal_rows]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,keys,title in zip(axes,[('loss','semantic_loss'),('prediction_loss','persistence_loss'),('alignment_loss','retention_loss')],
                             ['Episode/total validation','Future prediction vs persistence','Alignment/retention validation']):
        for key in keys:ax.plot(updates,[float(r['validation_'+key]) for r in temporal_rows],label=key)
        ax.set_title(title);ax.set_xlabel('Original temporal optimizer update');ax.legend(fontsize=8)
    axes[0].set_yscale('log');fig.suptitle('Original temporal branch: 100 updates, 50 complete passes; unchanged during spatial continuation')
    fig.tight_layout();fig.savefig(args.output/'temporal_curves.png',dpi=150);plt.close(fig)
    original=json.loads((experiment/'verification.json').read_text())['original_artifacts']
    for entry in original:assert digest(ROOT/entry['path'])==entry['sha256']
    result=dict(models=models,map_statistics=summary,source_rows=rows,shared_scale=scale,
        checks=dict(unique_sources=80,train_sources=64,validation_sources=16,
            attributed_source_model_pairs=160,duplicate_evaluations=0,
            prefix_map_aggregation_max_error=aggregate_error,raw_prefix_probability_sum_max_error=raw_probability_error,
            target_reversal_statistic_replay_max_error=target_replay_error,matched_frozen_parameter_max_drift=frozen_drift,
            original_artifacts_unchanged=len(original)),
        requirements=dict(recognition_on_selection_sources='passed',causal_prefix_map_accounting='passed',
            learned_spill_localization='not demonstrated; known qualitative failures and unsupported removal advantage',
            mask_metrics='unavailable: no verified masks or points',independent_scene_generalization='unverified',
            temporal_convergence='not established at the 100-update budget',warm_spill_evaluation='unavailable in this dataset'),
        interpretation='All source videos audited; conditional key sensitivity is not an input-pixel spill mask. Blue denotes low sensitivity, not a verified background label.',
        checkpoint_manifest=figure_manifest['checkpoint_sha256'],attention_sha256=digest(attention_path),
        teacher_cache_sha256=digest(DEFAULT_TEACHER),audit_source_sha256=digest(Path(__file__)))
    (args.output/'requirements.json').write_text(json.dumps(result,indent=2)+'\n')
    snapshots=args.output/'source_snapshot';snapshots.mkdir()
    for relative in ('scripts/audit_spill_requirements.py','scripts/verify_and_export_spill_overlays.py',
                     'src/vision_model/interpretability/map_geometry.py','tests/test_map_geometry.py',
                     'tests/test_spill_overlay_export.py','tests/test_final_attention.py'):
        destination=snapshots/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    print(json.dumps(dict(output=str(args.output),checks=result['checks'],requirements=result['requirements'],statistics=summary)),flush=True)


if __name__=='__main__':main()
