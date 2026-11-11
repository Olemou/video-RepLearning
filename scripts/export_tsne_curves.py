#!/usr/bin/env python3
"""Export named curves from saved histories/coordinates without another run."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args()
    out=a.experiment/'tsne_curves';out.mkdir(exist_ok=False)
    histories={'absolute clock':json.loads((a.baseline/'frozen/history.json').read_text()),
               'elapsed clock':json.loads((a.experiment/'frozen/history.json').read_text())}
    inventory=[]
    def export(fig,name,title,kind):
        for ext in ('png','pdf','svg'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        inventory.append(dict(file=name+'.png',full_name=title,kind=kind,formats=['png','pdf','svg']))
    series=[('total_loss','total_loss',1.,'Total combined trajectory loss'),
        ('vico_semantic_loss','semantic_loss',1.,'ViCo-LWIR episode semantic contrastive loss'),
        ('alignment_loss','alignment_loss',1.,'Matched-view temporal alignment loss (raw)'),
        ('prediction_loss','prediction_loss',1.,'Frozen-future-feature prediction loss'),
        ('retention_loss','retention_loss',1.,'Frozen-teacher feature-retention loss (raw)'),
        ('weighted_alignment_loss','alignment_loss',.1,'Weighted temporal alignment contribution: 0.1 × alignment'),
        ('weighted_retention_loss','retention_loss',.1,'Weighted feature-retention contribution: 0.1 × retention')]
    for name,key,weight,title in series:
        fig,ax=plt.subplots(figsize=(9,5))
        for color,(arm,history) in zip(('tab:blue','tab:orange'),histories.items()):
            for split,style in (('train','-'),('selection','--')):
                ax.plot([r['update'] for r in history],[weight*r[split][key] for r in history],style,color=color,label=f'{arm}: {split}')
        if key=='semantic_loss':ax.set_yscale('log')
        ax.axvline(20.5,color='gray',ls=':',lw=1,label='Optimizer reset before update21')
        ax.set(xlabel='Full-batch optimizer update',ylabel='Loss',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.text(.5,.01,'Frozen V-JEPA; 64 training / 16 reused selection sources; 100 updates per arm',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,1));export(fig,name,title,'loss_curve')
    coords=json.loads((a.experiment/'figures/tsne_coordinates.json').read_text())
    for arm,name in [('absolute clock','tsne_absolute_clock'),('elapsed clock','tsne_elapsed_clock')]:
        d=coords[arm];xy=np.asarray(d['coordinates']);train=set(d['train_indices'])
        # Labels are the recorded identity/source IDs: this study has clean/cold only.
        labels=[0 if source.startswith('clean_') else 1 for source in d['ids']]
        fig,ax=plt.subplots(figsize=(7,6))
        for label,color in ((0,'tab:blue'),(1,'tab:red')):
            for split,marker in (('train','o'),('selection','s')):
                ids=[i for i in range(80) if labels[i]==label and (i in train)==(split=='train')]
                ax.scatter(xy[ids,0],xy[ids,1],color=color,marker=marker,edgecolors='black',linewidths=.5,
                    label=f'{"clean" if label==0 else "cold spill"}: {split}')
        title=f't-SNE of episode embeddings: {arm}'
        ax.set(title=title);ax.set_xticks([]);ax.set_yticks([]);ax.legend(fontsize=8)
        fig.text(.5,.01,'Saved fit: seed239, perplexity15; class separation does not establish spill localization',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,1));export(fig,name,title,'tsne')
    for source,name,title,kind in [('loss_components','weighted_loss_components','Four weighted contributions to the combined trajectory objective','loss_components'),
        ('tsne_comparison','tsne_comparison','t-SNE comparison: absolute-clock versus elapsed-clock adapter','tsne')]:
        for ext in ('png','pdf','svg'):shutil.copy2(a.experiment/'figures'/f'{source}.{ext}',out/f'{name}.{ext}')
        inventory.append(dict(file=name+'.png',full_name=title,kind=kind,formats=['png','pdf','svg']))
    shutil.copy2(a.experiment/'figures/loss_history.csv',out/'loss_history.csv')
    shutil.copy2(a.experiment/'figures/tsne_coordinates.json',out/'tsne_coordinates.json')
    manifest=dict(experiment=str(a.experiment.resolve()),matched_baseline=str(a.baseline.resolve()),
        objective='ViCo + 0.1 alignment + prediction + 0.1 retention',frozen_backbone=True,
        optimizer_updates_per_arm=100,source_count=80,training_sources=64,reused_selection_sources=16,
        fresh_training=False,fresh_tsne_fit=False,inventory=inventory,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (a.baseline/'frozen/history.json',a.experiment/'frozen/history.json',a.experiment/'figures/tsne_coordinates.json')})
    (out/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Named t-SNE and loss curves','',
        'Completed80-source comparison: pretrained V-JEPA frozen; original absolute-clock adapter versus opt-in elapsed-clock adapter. '
        'Both use100 updates, seed239,64 training and16 reused selection sources. These figures are separate from full-backbone Experiment81.','',
        '`L_total = L_ViCo + 0.1 L_align + L_predict + 0.1 L_keep`.','',
        'Each plot is available as PNG, PDF and SVG. Solid lines are training; dashed lines are reused source selection. '
        'Individual alignment/retention plots are raw terms; their weighted contributions have separate files. '
        'The semantic loss uses a logarithmic vertical axis. No additional training or t-SNE fitting was performed during export.','',
        '| File | Full name |','|---|---|']
    lines += [f'| [{r["file"]}]({r["file"]}) | {r["full_name"]} |' for r in inventory]
    lines += ['','[Numeric loss history](loss_history.csv) · [Saved t-SNE coordinates](tsne_coordinates.json) · [Provenance](index.json)','',
        't-SNE includes all80 identity embeddings, with class/split markers; separate fits have incomparable axis positions. '
        'A low loss or separated t-SNE is not evidence that attention identifies liquid regions.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(directory=str(out),plots=len(inventory),formats=['png','pdf','svg'])),flush=True)


if __name__=='__main__':main()
