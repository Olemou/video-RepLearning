#!/usr/bin/env python3
"""Summarize saved ViCo Eq. 10 runs, plot convergence, and replay the model.

Run from the project directory after scripts/train.py and the documented
ablations. Historical SupCon reference fields are renamed without changing
any saved numerical outcome, to distinguish them from the active Eq. 10 loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from vision_model.training.video_contrastive import VideoRepresentationModel
from vision_model.training.vico import vico_pairwise_contrastive


def clarify_history(row):
    if 'vico_loss' not in row or 'supcon_contrastive_loss' in row:
        return
    for key in ('contrastive_loss', 'positive_entropy', 'contrastive_excess', 'max_anchor_excess',
                'negative_leakage_loss', 'positive_nonuniformity_loss', 'mean_negative_probability',
                'binary_unit_sphere_minimum_excess'):
        if key in row:
            row['supcon_' + key] = row.pop(key)
    row['contrastive_loss'] = row['vico_loss']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/vico_eq10')
    parser.add_argument('--cache', type=Path, default=ROOT / 'reports/experiments/unique_7_clean_7_spill/checkpoints/checkpoint_full_frame_flip_features.pt')
    args = parser.parse_args()
    torch.set_num_threads(4)
    reports = {}
    for path in sorted(args.output.glob('*/training_report.json')):
        r = json.loads(path.read_text())
        if r['settings']['objective'] != 'vico':
            continue
        for row in r['history']:
            clarify_history(row)
        clarify_history(r['final_objective_metrics'])
        for fold in r['fold_convergence']:
            if 'max_anchor_excess' in fold:
                fold['supcon_max_anchor_excess'] = fold.pop('max_anchor_excess')
        for old, new in (('final_training_diagnostics', 'supcon_reference_on_original_views'),
                         ('pretrained_training_diagnostics', 'pretrained_supcon_reference')):
            if old in r:
                r[new] = r.pop(old)
        r['optimized_equation'] = 'loss_doc/VicoLWIR.pdf Eq. 10'
        r['augmentation'].pop('same_source_positive_pairs', None)
        r['augmentation']['training_positive_policy'] = 'same-source positive views included with weight one'
        r['report_schema'] = 'vico-explicit-objective-v1'
        path.write_text(json.dumps(r, indent=2) + '\n')
        reports[path.parent.name] = r
    r = reports['final']
    reference = {p['video_id']: p['fold'] for p in r['predictions']}
    comparison = []
    for name, run in reports.items():
        assert {p['video_id']: p['fold'] for p in run['predictions']} == reference
        final = run['final_objective_metrics']
        comparison.append({'variant': name, 'epochs': run['epochs_run'], 'loss': final['vico_loss'],
                           'max_pair_loss': final['max_positive_pair_loss'],
                           'tolerance': run['settings']['convergence_tolerance'],
                           'converged': run['converged'],
                           'all_folds_converged': all(f['converged'] for f in run['fold_convergence']),
                           'held_out_correct': sum(p['prediction'] == p['label'] for p in run['predictions'])})
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'loss_doc').glob('*.pdf')}
    (args.output / 'document_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    history = r['history']
    x = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].semilogy(x, [h['vico_loss'] for h in history], label='Raw Eq. 10 loss')
    axes[0].semilogy(x, [h['max_positive_pair_loss'] for h in history], label='Worst positive pair')
    axes[0].axhline(r['settings']['convergence_tolerance'], color='black', linestyle='--', label='Stopping tolerance')
    axes[0].axvline(r['settings']['curriculum_epochs'], color='gray', linestyle=':', label='Curriculum ends; Adam reset')
    axes[0].set(xlabel='Optimizer update', ylabel='Unshifted pairwise contrastive loss', title='Selected ViCo-LWIR Eq. 10')
    axes[1].plot(x, [h['minimum_positive_cosine'] for h in history], label='Worst positive cosine')
    axes[1].plot(x, [h['maximum_negative_cosine'] for h in history], label='Worst negative cosine')
    axes[1].set(xlabel='Optimizer update', ylabel='Cosine similarity', ylim=(-1.05, 1.05), title='Both training views: all pair constraints')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(args.output / 'convergence.png', dpi=180)
    fig.savefig(args.output / 'convergence.pdf')
    plt.close(fig)
    with (args.output / 'loss_series.csv').open('w') as handle:
        keys = ['epoch', 'vico_loss', 'max_positive_pair_loss', 'minimum_positive_cosine', 'maximum_negative_cosine', 'optimizer_resets']
        writer = csv.DictWriter(handle, keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(args.output / 'final/video_representation.pt', map_location='cpu', weights_only=True)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    assert hashlib.sha256(args.cache.read_bytes()).hexdigest() == ckpt['feature_cache_sha256']
    model = VideoRepresentationModel(ckpt['width'], ckpt['bottleneck'], head_type=ckpt['head_type'], embedding_dim=ckpt['embedding_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    index = [cache['ids'].index(video_id) for video_id in ckpt['training_video_ids']]
    original = cache['features'][index]
    augmented = cache['horizontal_flip_features'][index]
    valid, labels = cache['valid'][index], cache['labels'][index]
    with torch.no_grad():
        z = torch.cat((model(original, valid), model(augmented, valid)))
        loss, metrics = vico_pairwise_contrastive(z, labels.repeat(2), source_ids=torch.arange(len(index)).repeat(2),
            epoch=r['settings']['curriculum_epochs'], curriculum_epochs=r['settings']['curriculum_epochs'],
            temperature=r['settings']['temperature'])
        assert float(metrics['max_positive_pair_loss']) <= r['settings']['convergence_tolerance']
        assert float(metrics['minimum_pair_separation']) > 0
        prediction = model.predict(original, valid)
    saved = torch.load(args.output / 'final/video_embeddings.pt', weights_only=True)
    for key in prediction:
        torch.testing.assert_close(prediction[key], saved[key])
    replay = {'loss': float(loss), 'max_pair_loss': float(metrics['max_positive_pair_loss']),
              'training_correct': int((prediction['labels'] == labels).sum()),
              'embedding_replay': 'passed', 'strict_convergence_replay': 'passed'}
    (args.output / 'checkpoint_replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    similarity = prediction['embeddings'] @ prediction['embeddings'].T
    order = labels.argsort(stable=True)
    im = ax.imshow(similarity[order][:, order].numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    n_clean = int((labels == 0).sum())
    ax.axhline(n_clean - .5, color='black', linewidth=.8)
    ax.axvline(n_clean - .5, color='black', linewidth=.8)
    ax.set(title='Training cosine similarities: original videos', xlabel='Clean videos, then spill videos', ylabel='Clean videos, then spill videos')
    fig.colorbar(im, ax=ax, label='Cosine similarity')
    fig.savefig(args.output / 'embedding_similarity.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'comparison': comparison, 'checkpoint_replay': replay}, indent=2))


if __name__ == '__main__':
    main()
