#!/usr/bin/env bash
# Separate, resumable experiment; never overwrites the original pipeline.
set -euo pipefail
cd /home/olemou/video-RepLearning
RUN_ROOT="${1:-reports/experiments/experiment_81_execution_20261106}"
RUN_ROOT="$(realpath "$RUN_ROOT")"
RUN_PYTHON=/home/olemou/miniconda3/envs/torchgpu/bin/python
export LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib
export OMP_NUM_THREADS=4
export PYTHONPATH=src:scripts
GPU_COUNT="$($RUN_PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if [ "$GPU_COUNT" -lt 1 ]; then
    echo 'No allocated CUDA GPU is visible' >&2
    exit 1
fi
if [ ! -f "$RUN_ROOT/prepared/prepared.pt" ]; then
    "$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
        scripts/prepare_experiment81.py --output "$RUN_ROOT/prepared"
fi
SNAPSHOT="$RUN_ROOT/training_source_snapshot"
if [ ! -d "$SNAPSHOT" ]; then
    "$RUN_PYTHON" - "$SNAPSHOT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path.cwd(); out=Path(sys.argv[1]); out.mkdir()
rows=[]
for prefix in ('src','scripts'):
    for p in sorted((root/prefix).rglob('*')):
        if p.suffix not in ('.py','.sh'): continue
        target=out/p.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        rows.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for name in ('AGENTS.md','README.md','pyproject.toml'):
    shutil.copy2(root/name,out/name)
(out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
fi
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT/scripts"
for MODE in frozen full; do
    if [ -f "$RUN_ROOT/$MODE/completed.json" ]; then
        echo "$MODE already complete; preserving its outputs"
        continue
    fi
    RESUME=()
    if [ -f "$RUN_ROOT/$MODE/checkpoint.pt" ]; then RESUME=(--resume); fi
    "$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
        "$SNAPSHOT/scripts/train_experiment81.py" \
        --prepared "$RUN_ROOT/prepared/prepared.pt" --output "$RUN_ROOT/$MODE" \
        --mode "$MODE" --updates 100 --chunk-sources 2 "${RESUME[@]}" \
        > "$RUN_ROOT/${MODE}_training.log" 2>&1
done
"$RUN_PYTHON" "$SNAPSHOT/scripts/report_experiment81.py" --root "$RUN_ROOT" \
    > "$RUN_ROOT/report_generation.log" 2>&1
echo 'Experiment81 training and measured plots complete; spatial validation is separate.'
