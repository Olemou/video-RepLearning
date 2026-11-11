#!/usr/bin/env python3
"""Bounded, matched automatic-candidate pilot in an existing research directory."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    import torch
    if torch.cuda.device_count()!=2 or any('H100' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('This bounded protocol requires the two allocated H100s')
    weight=json.loads((a.root/'automatic_weight_calibration.json').read_text())['fg_weight']
    distributed=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    common=['--stage','joint','--pseudo-cache',str(a.root/'normal_reference/maps.pt'),'--pseudo-case','block5_raw',
        '--probe-checkpoint',str(a.root/'automatic_probe/checkpoint.pt'),'--fg-weight',str(weight),'--updates','100']
    stages=[('automatic_control',distributed+['scripts/train_spill_localization.py',*common,'--trajectory-only','--output',str(a.root/'automatic_control')]),
        ('automatic_joint',distributed+['scripts/train_spill_localization.py',*common,'--output',str(a.root/'automatic_joint')]),
        ('automatic_maps',distributed+['scripts/evaluate_automatic_spill_maps.py','--root',str(a.root),'--output',str(a.root/'automatic_maps')])]
    status=[]
    for name,cmd in stages:
        destination=a.root/name
        if destination.exists():raise FileExistsError(destination)
        item=dict(stage=name,command=cmd,state='running');status.append(item)
        (a.root/'pilot_status.json').write_text(json.dumps(status,indent=2)+'\n')
        with (a.root/(name+'.log')).open('x') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        item.update(state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode)
        (a.root/'pilot_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if result.returncode:raise SystemExit(result.returncode)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
