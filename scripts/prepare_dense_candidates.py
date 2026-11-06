#!/usr/bin/env python3
"""Reuse measured frozen-reference maps; no refitting or validation tuning."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from vision_model.training.dense_spill import dense_consensus_candidates,candidate_coverage
from vision_model.models.dense_encoder import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    d=torch.load(a.reference/'candidates.pt',weights_only=False,map_location='cpu')
    prior=json.loads((a.reference/'report.json').read_text())
    targets=dense_consensus_candidates(d['maps']['raw'],d['maps']['residual'],d['valid'],
        d['content'].flatten(1),d['labels'],prior['calibration'])
    report=dict(metadata={**prior['metadata'],'code_sha256':file_sha256(__file__),
        'parent_reference_report_sha256':file_sha256(a.reference/'report.json'),
        'policy':'Both geometries and both views exceed training-clean q99; 3-MAD spatial excess in at least one geometry; both q95 normal-compatible for BG; no forced area'},
        calibration=prior['calibration'],coverage=candidate_coverage(targets,d['labels'],d['train_indices']),
        interpretation='Unverified automatic target coverage; not spill localization accuracy',sources=[])
    for i,name in enumerate(d['ids']):
        t=targets[i,0]
        report['sources'].append(dict(id=name,index=i,label=int(d['labels'][i]),
            split='train' if i in d['train_indices'] else 'validation_selection',
            foreground=int((t==1).sum()),background=int((t==0).sum()),
            same_frame_fg_bg=int(((t==1).any(-1)&(t==0).any(-1)).sum())))
    a.output.mkdir(parents=True,exist_ok=False)
    d['targets']=targets;torch.save(d,a.output/'candidates.pt')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['coverage']),flush=True)


if __name__=='__main__':main()
