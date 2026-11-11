#!/usr/bin/env bash
set -euo pipefail
cd /home/olemou/video-RepLearning
RUN_ROOT="$(realpath "${1:-reports/experiments/experiment_81_execution_20261106}")"
RUN_PYTHON=/home/olemou/miniconda3/envs/torchgpu/bin/python
export LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib
export OMP_NUM_THREADS=4
SNAPSHOT="$RUN_ROOT/analysis_execution_source_snapshot"
if [ ! -d "$SNAPSHOT" ]; then
    "$RUN_PYTHON" - "$SNAPSHOT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path.cwd();out=Path(sys.argv[1]);out.mkdir();rows=[]
for prefix in ('src','scripts'):
    for p in sorted((root/prefix).rglob('*.py')):
        target=out/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        rows.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
fi
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT/scripts"
for MODE in frozen full; do
    while [ ! -f "$RUN_ROOT/$MODE/completed.json" ]; do sleep 10; done
    "$RUN_PYTHON" "$SNAPSHOT/scripts/diagnose_experiment81_shortcuts.py" --root "$RUN_ROOT" --case "$MODE" \
        --output "$RUN_ROOT/shortcut_controls/$MODE" --resume > "$RUN_ROOT/shortcuts_${MODE}_final.log" 2>&1
    "$RUN_PYTHON" "$SNAPSHOT/scripts/occlude_experiment81.py" --root "$RUN_ROOT" --case "$MODE" \
        --output "$RUN_ROOT/occlusion/$MODE" --resume > "$RUN_ROOT/occlusion_${MODE}.log" 2>&1
done
"$RUN_PYTHON" "$SNAPSHOT/scripts/audit_experiment81_gradients.py" --root "$RUN_ROOT" \
    --output "$RUN_ROOT/final_gradient_audit.json" > "$RUN_ROOT/final_gradient_audit.log" 2>&1
"$RUN_PYTHON" "$SNAPSHOT/scripts/report_experiment81_analysis.py" --root "$RUN_ROOT" \
    > "$RUN_ROOT/deep_analysis_report.log" 2>&1
echo 'Deep fixed-model controls complete; no additional training updates were taken.'
