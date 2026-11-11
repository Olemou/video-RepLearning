# Spatial pilots with the causal trajectory model

## Current annotation-free path

Manual spatial annotation is not required for the current task. The
[completed two-H100 study](../reports/experiments/spill_annotation_free_80_20261021/report.md)
compares normal-reference novelty, normal-only correspondence training and
automatic-candidate shared-suffix pilots. All reference fitting uses training
clean sources only; cold patches are not uniformly labelled foreground.

`scripts/benchmark_normal_spill.py --output NEW_REFERENCE_DIRECTORY` writes the
normal-reference maps and detached automatic candidates. Optionally select
`--modes selected_residual` for the separately tested channel-selection control.
Candidates require absolute novelty, exceptional within-frame novelty and
identity/mirror agreement; ambiguous cells are ignored. Clean video labels
provide an absence-of-liquid assumption. Candidates in cold videos remain
**unverified**, including when the model matches them perfectly.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_spill_localization.py \
  --stage probe --pseudo-cache /path/to/reference/maps.pt \
  --pseudo-case block5_raw --output /path/to/NEW_probe --updates 100

torchrun --standalone --nproc_per_node=2 scripts/train_spill_localization.py \
  --stage joint --pseudo-cache /path/to/reference/maps.pt \
  --pseudo-case block5_raw --probe-checkpoint /path/to/probe/checkpoint.pt \
  --fg-weight WEIGHT --output /path/to/NEW_joint --updates 100
```

Set `WEIGHT` from training-only calibration. Add `--trajectory-only` for the
matched zero-local-loss joint control; it keeps the learned probe head frozen.
Checkpoint metadata distinguishes `automatic_unverified` from `verified_points`,
records supervision hashes and keeps pseudo-candidate agreement separate from
verified localization. `--annotations` and `--pseudo-cache` are mutually exclusive.

The completed pilot's calibrated coefficient was clipped to 10; this is not a
recommended universal default. It did not produce demonstrated liquid focus.
The local auxiliary can fit automatic seeds without making the episode classifier
depend on liquid. Use actual-decision input controls and keep that negative result.
These are custom adaptations inspired by normal-modeling and weak-supervision
research, not exact reproductions of UniNet, ToCo or FBR.

## Optional verified-point path (earlier implementation)

This opt-in path implements the approved dual-loss audit proposal. The default
trajectory trainer and its four coefficients remain unchanged. The new joint
objective is

`L = L_ViCo(episode) + 0.1 L_align + L_predict + 0.1 L_keep + lambda_FGBG L_FGBG`.

`L_FGBG` is class-balanced partial binary cross entropy at **verified liquid and
non-liquid points**. Unlabelled cells are ignored. The current all-patch video
label attraction and extra spatial retention-1 term are absent. This objective
is a custom adaptation, not an equation attributed to ViCo-LWIR or another
paper. Lambda must be explicitly selected using training-only information.

## Data requirement

No verified spatial point file was available during implementation. The code
therefore cannot train a real spill localizer yet. It never infers patch labels
from `cold`, `warm`, `clean`, pixel intensity, the existing failed heatmap or
an assumed floor region. Passing an empty/unverified template reports not-ready
and stops before creating a training output directory.

Generate the annotation template without overwriting an existing file:

```bash
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python \
  scripts/train_spill_localization.py \
  --write-template reports/experiments/spill_fg_bg_implementation_20261021/points_to_review.json
```

The template contains source ID/hash, source dimensions and the exact sampled
tubelet endpoint frame index. Every record starts with `verified: false`, an
empty reviewer and an empty `points` list. It contains **no generated labels**.

Root fields are `schema: spill_points_v1`,
`coordinate_system: source_pixel_centres`, `temporal_scope: tubelet_endpoint`,
and `records`. Each reviewed record requires `verified: true` and nonempty
`verified_by`. Its points have zero-indexed original-image pixel-center `x,y`
and `label: 1` for liquid or `label: 0` for non-liquid. Label `-1` explicitly
ignores an ambiguous point. Use clear centers away from mixed patch boundaries.
Do not label a whole patch foreground merely because a small corner is liquid.

Coordinates map through the official rounded full-frame letterbox, including
explicit source reflection before resizing for the mirror view. The loader
rejects wrong source hashes, incorrect dimensions, unsampled/padded frame
indices, padding-dominated cells, contradictory labels in a cell and liquid
labels on a clean source. Duplicate same-label points in one cell count once.
These provenance checks cannot independently establish that a human label is
correct. Review annotations against source frames.

```bash
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python \
  scripts/train_spill_localization.py --validate-only \
  --annotations /path/to/verified_points.json
```

Exit 2 means verified foreground or background is missing from the training
split. Validation labels never satisfy that training requirement. A valid
annotation file alone does not establish independent-scene evaluation.

## Stage 1: frozen-feature local probe

```bash
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python \
  scripts/train_spill_localization.py --stage probe --layer 23 \
  --annotations /path/to/verified_points.json \
  --output reports/experiments/spill_points_probe_NEW --updates 100
```

Layer 23 uses the FP32 frozen final-block reference from the verified pre23
cache. Layers 5, 11 and 17 use the accepted intermediate cache; select each in
a separate new output directory with the same verified training points. No
trajectory input, trajectory weight or encoder weight changes in probe mode.
The head learns two normalized prototypes from pointwise projected features.
Its sigmoid is across liquid/background classes, not across spatial positions,
so clean frames are not forced to assign attention mass to some region.

Evaluate verified held-out local points before joint adaptation. Point accuracy,
AUROC and AP are discrimination metrics over annotated cells, not mask IoU or
proof that the episode classifier uses liquid. Mirrored points are derived
views; reported point metrics use identity cells only.

## Stage 2: connect the local and temporal paths

After a useful final-layer probe and training-only coefficient selection:

```bash
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python \
  scripts/train_spill_localization.py --stage joint \
  --annotations /path/to/verified_points.json \
  --probe-checkpoint /path/to/probe/checkpoint.pt \
  --fg-weight FG_WEIGHT \
  --output reports/experiments/spill_points_joint_NEW --updates 100
```

Replace `FG_WEIGHT` with the positive numerical coefficient selected from
training-only calibration/inner validation. There is no automatic coefficient
search or claimed paper-prescribed coefficient. Start with a short smoke run.

The path is

`complete causal pre23 prefix → final-block attention → shared patch tokens`

with two consumers: the pointwise local head and the **existing spatial mean
into the original TrajectoryModel**. Teacher means remain immutable future/
retention targets. `student_features` explicitly bypass the training input
detach; targets do not. The temporal architecture, timestamps, masks, event
pooling, horizons 1/2/4 and loss coefficients are retained. Original trajectory
parameters, final-block MLP/norms and earlier blocks remain frozen in this pilot;
only final Q/K/V/output and the local head train. Changing student inputs can
still change recognition and prediction, so no no-degradation claim is automatic.

All six valid prefixes and both views are evaluated each full-batch update.
This has more prefix exposure per update than the old one-prefix spatial-only
run; compare budgets using the saved exposure tensor, not the update count
alone. Seed 239, AdamW epsilon 1e-12, zero decay, clipping 5 and the reset before
update 21 are recorded. Default head LR is 1e-3; attention LR is 3e-6. Both are
explicit CLI settings and are not claimed optimal. No mirror loss, MIL,
decoder, crop, erase or fresh per-epoch augmentation was added.

Use `torchrun --standalone --nproc_per_node=2` with the same script/arguments
on the allocated GPU node. Every rank owns a distinct subset of each split.
Global trajectory losses use differentiable output gathering; partial BCE
uses global foreground/background counts, including when a rank has only one
annotated class. Replay streams exact first-order cotangents through every
prefix and synchronizes DDP once per update. Only global rank zero writes.

## Outputs and interpretation

New directories contain `checkpoint.pt`, optimizer/RNG state, exact update and
source/view/time exposures, `history.csv`, fixed-final-curriculum
`evaluation_history.json`, `representations.pt` and `report.json`. Reports
include train/selection recognition/retrieval, point metrics, original and
matched-initial state drift, component losses, horizon-wise prediction versus
persistence, source/cache/annotation/code hashes and frozen-parameter checks.
There is no best-validation-checkpoint selection.

The initial joint evaluation records FP32 suffix replay drift from the original
BF16 teacher path. Compare it with the matched initial student as well as the
original model; do not attribute a precision change to spatial supervision.

For a real trained joint checkpoint, visualize a source with

```bash
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python \
  scripts/visualize_spill_localization.py --checkpoint /path/to/joint/checkpoint.pt \
  --source-id SOURCE_ID --output /path/to/new_visualization
```

The comparison distinguishes **supervised local scores**, fixed to [0,1], from
**conditional final-block gradients of the actual joint trajectory decision**.
The latter uses training-only episode prototypes from that checkpoint. Provide
`--decision-scale` when comparing multiple sources on one scale; otherwise the
single-source maximum is recorded. Raw tensors, frame/timestamp metadata and
checkpoint prediction replay checks are saved. No per-frame rescaling, chosen
head, ROI or smoothing is used. Cached pre23 states limit this explanation to
the final-block conditional path; it is not end-to-end pixel attribution.

## Acceptance criteria

Require independently verified spatial improvement and no recognition/prediction
regression. Retain video recognition and prediction below the persistence
baseline; report all horizons and representation drift. Use masks for IoU/pixel
AP or suitable independent point evaluation when only points exist. Test actual
episode-margin sensitivity to annotated liquid removal against equal-area
background/random controls with fresh full-prefix image encodes. An improved
auxiliary map alone is insufficient. Hold out scene/session groups from fitting
and prototype construction when those groups become available. The current
16-video split is a selection set, not an independent scene test.
