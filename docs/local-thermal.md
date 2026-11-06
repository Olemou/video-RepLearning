# Finite-field local thermal experiments

The supported recognition/temporal path remains
`L_ViCo + 0.1 L_align + L_predict + 0.1 L_keep` with its original frozen V-JEPA
teacher and checkpoint. The optional local path below is a **separate output**.
It does not turn frozen last-layer attention into a segmentation mask.

Its input is the full 384x288 content frame, converted to grayscale. This is
separate from, and does not modify, official V-JEPA preprocessing. Four local
radii (4,8,16,32 pixels) give standardized signed contrast and neighboring-rank
channels. No global CDF, whole-image normalization, coordinates, absolute clock,
source ID or final contextual V-JEPA token enters the local head. Eight channels
are averaged onto an 8-pixel grid. FP16 storage is part of the execution path;
fresh input controls reproduce this boundary before the FP32 readout.

The pointwise head has a maximum radius of 39 pixels. An optional two-layer 3x3
convolutional head adds 16 pixels, giving 55 pixels. The latter never upsamples
and has no global pooling, batch normalization, MIL or segmentation decoder.
The locality tests change distant pixels and check the selected local output.

Automatic pairs come from connected components and local rings within each
frame. Raw-intensity components are retained as a failed control. Components
of relative contrast reduce broad dark-surface selection but still include
noise, clothing and other dark objects. Targets are **unverified**; no region
was drawn from the user's example. No video is assumed clean at its start.

The local objective is

`L_local = L_absence(clean) + lambda L_pair(candidate, local_ring)`.

Each source has equal weight. Pair ranking compares mean native-cell scores
within one candidate and its ring. A uniform scene offset cancels; unknown
sources abstain. Lambda uses training-only initial gradient norms, target
ratio 0.5, limits [0.01,10]. The absence-only control has lambda 0. This objective
is an explicitly proposed adaptation, not ViCo-LWIR Eq. 10. Temporal consistency
is disabled because registered/occlusion-checked correspondences have not been
validated. The original trajectory losses are not replaced or retrained.

From the repository root, use a fresh run directory and two allocated GPUs:

```bash
export LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=4
RUN_PYTHON=/home/olemou/miniconda3/envs/torchgpu/bin/python
RUN_DIR=reports/experiments/my_local_run
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/prepare_local_thermal.py --output "$RUN_DIR/prepared"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/prepare_local_pairs.py --root "$RUN_DIR"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/train_local_thermal.py --cache "$RUN_DIR/prepared/cache.pt" \
 --pairs "$RUN_DIR/relative_pairs.pt" --output "$RUN_DIR/control" --control
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/train_local_thermal.py --cache "$RUN_DIR/prepared/cache.pt" \
 --pairs "$RUN_DIR/relative_pairs.pt" --output "$RUN_DIR/joint"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/control_local_thermal.py --root "$RUN_DIR"
"$RUN_PYTHON" scripts/report_local_thermal.py --root "$RUN_DIR" --relative-pairs
```

Default budget 100 full-batch updates, seed 239, AdamW 1e-3, zero decay,
epsilon 1e-12, clipping 5, state reset before 21. `--updates 1000` is a separate
long-budget experiment. `--local-conv` selects the spatial readout ablation;
use it in both matched control/joint commands. Each source retains both fixed
views. Every GPU owns distinct sources; global source counts normalize losses,
evaluation rows are gathered without duplicates, and rank 0 writes outputs.

Overlays display excess above each model's **training-clean q99.5**, using a
common training-only magnitude scale across the compared models and all frames.
There is no per-frame scaling or blur. Raw scores remain in `maps.pt`.
The threshold is a normal false-alarm policy, not a liquid probability or proof
that blue pixels contain no liquid. Input controls recompute features from the
changed grayscale inputs; they do not reuse unperturbed descriptors.

The [80-source implementation report](../reports/experiments/spill_local_relative_80_20261022/report.md)
records successful checks and failed localization cases. Locality, pseudo-pair
fit, low false alarms and good original t-SNE are separate from correct liquid
identity. This path remains experimental unless actual liquid focus is verified.
