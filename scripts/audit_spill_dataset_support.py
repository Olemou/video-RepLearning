#!/usr/bin/env python3
"""Whole-manifest source hashes, label consistency, and acquisition overlap."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vision_model.data.labels import label_for


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for b in iter(lambda:handle.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
    hashes=collections.defaultdict(list);resolutions=collections.defaultdict(collections.Counter)
    for row in rows:
        if row['label']!=label_for(row['category']):raise ValueError('Authoritative label mismatch')
        path=a.manifest.parent/row['video_path'];row['sha256']=sha(path);row['bytes']=path.stat().st_size
        hashes[row['sha256']].append(row);resolutions[(row['width'],row['height'])][row['category']]+=1
    duplicates=[dict(sha256=h,sources=[r['video_id'] for r in group],labels=sorted(set(r['label'] for r in group))) for h,group in hashes.items() if len(group)>1]
    result=dict(manifest=str(a.manifest),manifest_sha256=sha(a.manifest),records=len(rows),
        unique_byte_sources=len(hashes),duplicate_groups=duplicates,
        cross_label_duplicate_groups=[r for r in duplicates if len(r['labels'])>1],
        acquisition_support=[dict(width=w,height=h,category_counts=dict(counts)) for (w,h),counts in resolutions.items()],
        limitation='Resolution does not establish camera/session identity. File hashes do not identify re-encoded duplicates or overlapping physical recordings.',
        source_rows=rows,training_updates=0,manual_annotations=0)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_rows'}),flush=True)


if __name__=='__main__':main()
