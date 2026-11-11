#!/usr/bin/env bash
# Finish spatial diagnostics after the paired training checkpoints are complete.
set -euo pipefail
cd /home/olemou/video-RepLearning
RUN_ROOT="$(realpath "${1:-reports/experiments/experiment_81_execution_20261106}")"
RUN_PYTHON=/home/olemou/miniconda3/envs/torchgpu/bin/python
export LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib
export OMP_NUM_THREADS=4
export PYTHONPATH=src:scripts
GPU_COUNT="$($RUN_PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if [ "$GPU_COUNT" -lt 1 ]; then exit 1; fi
SNAPSHOT="$RUN_ROOT/spatial_execution_source_snapshot"
if [ ! -d "$SNAPSHOT" ]; then
    "$RUN_PYTHON" - "$SNAPSHOT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path.cwd(); out=Path(sys.argv[1]); out.mkdir(); rows=[]
for prefix in ('src','scripts'):
    for p in sorted((root/prefix).rglob('*')):
        if p.suffix not in ('.py','.sh'): continue
        target=out/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        rows.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
fi
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT/scripts"
for MODE in frozen full; do
    while [ ! -f "$RUN_ROOT/$MODE/completed.json" ]; do sleep 10; done
    if [ ! -f "$RUN_ROOT/spatial/$MODE/completed.json" ]; then
        RESUME=()
        if [ -d "$RUN_ROOT/spatial/$MODE" ]; then RESUME=(--resume); fi
        "$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
            "$SNAPSHOT/scripts/attribute_experiment81.py" --root "$RUN_ROOT" --case "$MODE" \
            --output "$RUN_ROOT/spatial/$MODE" "${RESUME[@]}" \
            > "$RUN_ROOT/spatial_${MODE}_execution.log" 2>&1
    fi
    if [ ! -f "$RUN_ROOT/input_controls/$MODE/results.json" ]; then
        "$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
            "$SNAPSHOT/scripts/control_experiment81.py" --root "$RUN_ROOT" --case "$MODE" \
            --maps "$RUN_ROOT/spatial/$MODE" --output "$RUN_ROOT/input_controls/$MODE" \
            > "$RUN_ROOT/controls_${MODE}.log" 2>&1
    fi
done
while [ ! -f "$RUN_ROOT/report.md" ]; do sleep 5; done
"$RUN_PYTHON" "$SNAPSHOT/scripts/report_experiment81_spatial.py" --root "$RUN_ROOT" \
    > "$RUN_ROOT/spatial_gallery.log" 2>&1
echo 'Spatial diagnostics complete; inspect actual maps before a localization claim.'
