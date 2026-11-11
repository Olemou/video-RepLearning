# Experiment 81: separate online JEPA trajectory pipeline

The authoritative cached-feature trajectory default is unchanged. This opt-in
experiment compares a freshly initialized trajectory over frozen JEPA with the
same trajectory initialization and **all 24 encoder blocks, patch embedding and
final norm trainable**. Both keep

`L = L_ViCo(z_episode) + 0.1 L_align + L_predict + 0.1 L_keep`.

There is no manual spatial annotation, MIL, segmentation decoder, extra mirror
loss or per-patch inheritance of a positive video label. The objective still
does not specify liquid locations; classification success must be assessed
separately from spatial evidence.

Read the [proposal/audit](../reports/experiments/experiment_81_full_jepa_trajectory_20261106/report.md).
The [execution directory](../reports/experiments/experiment_81_execution_20261106)
contains the completed 100-update arms and their logs/checkpoints. Its
[diagnosis](../reports/experiments/experiment_81_execution_20261106/diagnosis.md)
records improved temporal fit and persistent qualitative localization failures.
It uses the same 80 sources,
64 training / 16 repeatedly used source-selection split, seed 239 and 100
full-batch updates per arm. This is not an independent scene test.

## Online student and immutable teacher

`scripts/prepare_experiment81.py` verifies source/checkpoint/manifest hashes,
reconstructs exact sampled frames and the existing identity/mirror/eight-view
finite augmentation bank, and preserves official full-frame preprocessing.
It stores letterboxed uint8 inputs and original decoded frames. Teacher targets
are recomputed once from the original immutable pretrained encoder using FP32
weights and BF16 forwards. Numerical differences from historical BF16-weight
targets are measured in `prepared/metadata.json`; historical curves are not
silently relabeled as the new baseline.

`scripts/train_experiment81.py` calls the original four-loss implementation via
its differentiable `student_features` interface. The full student is recomputed
online for each separately observed prefix. Representation-gradient replay
bounds activation memory while retaining the complete global contrastive loss.
The replay must reproduce pooled features before its gradients are accepted.
Prediction/retention targets remain detached; a full-video encode cannot replace
the independent prefix forwards.

FP32 master parameters/optimizer states are kept; encoder forwards use BF16,
and ViCo remains float64. Trajectory LR is 1e-3, encoder LR 1e-5, AdamW weight
decay 0, epsilon 1e-12, clipping 5. Both arms reset optimizer state before
update 21. Two distinct bank views/source/update are selected deterministically.
Fixed identity/mirror evaluation uses final-curriculum weights at every update.

## Execution

Run from the repository on an **allocated GPU node**, using a new run directory:

```bash
bash scripts/run_experiment81.sh reports/experiments/my_experiment81
bash scripts/run_experiment81_spatial.sh reports/experiments/my_experiment81
```

The first command prepares inputs if needed, snapshots executable sources, runs
both arms and generates loss/t-SNE figures. It can resume saved optimizer/RNG
states with the same settings. The second command runs all-source spatial
diagnostics, selection-source input controls plus cold56, and exports the gallery.
Both commands use every GPU visible to the allocation through torchrun. The
requested 6 November run on rg32101 has **one H100 80 GB**, allocation 25817192;
two GPUs must not be claimed for that run.

Ranks own distinct sources. Global semantic gradients are gathered
differentiably; DDP averaging cancels the replicated-loss cotangent factor.
Tests compare all parameter gradients with a serial reference, including uneven
source shards. Evaluation avoids duplicate sampler padding; only global rank 0
writes consolidated outputs. Do not add a second world-size division.

Existing outputs are protected. Completed arms are reused; changed settings need
a new directory. A partial spatial stage can resume only matching checkpoint and
method hashes. Snapshot directories identify executed code, rather than relying
on the dirty working tree's git HEAD.

## Outputs and interpretation

Loss figures include raw/weighted terms and one total-loss figure, plus CSV and
unmodified histories. t-SNE includes all 80 identity embeddings with class and
split markers; classification/retrieval are evaluated in the original space.

The spatial path exposes differentiable probabilities used by an explicit
attention forward, explaining a fixed training-prototype spill-minus-clean
margin. It publishes all-head final attention-times-gradient, all-layer positive
gradients, selected intermediate layers, and class-agnostic rollout with a
last-tubelet query readout rather than an invented CLS token. This is a
**LeGrad-style adaptation**, not original LeGrad, segmentation or pixel probability.
Instrumented-versus-production discrepancies are recorded; its own replay must
be exact within the stated tolerance. Maps remain at tubelet resolution.

Scales are fixed using training sources pooled across arms; no per-frame
normalization, chosen head, manual ROI, targeted recoloring or display smoothing
is used. Blue denotes low displayed sensitivity, not verified clean pixels.
Input controls recompute the video-to-embedding path after equal-area high-map
and random removals under mean and blur fills, with fixed prototypes. Their
unperturbed features/margins must match saved production results first.

Inspect cold56, cold10, cold13 and all-source outputs before asserting better
liquid focus. Input sensitivity and concentrated maps can still identify scene
shortcuts. No real IoU, pixel AP or pointing accuracy exists without ground truth.
The supplied reference image is an acceptance example, never a training target.

## Verification

Run focused tests and the full suite in the environment documented in AGENTS.md:

```bash
LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib \
PYTHONPATH=src:scripts OMP_NUM_THREADS=1 \
/home/olemou/miniconda3/envs/torchgpu/bin/python -m pytest -q
```

`tests/test_online_trajectory.py` covers true causal prefixes, invalid-tail
exclusion, four-loss gradient replay, activation checkpointing, two-rank gradient
equivalence, connected attention capture and non-CLS rollout. Actual GPU replay,
all-block gradients and block parameter changes are recorded during the run.

## All-block and independent input-gradient audit

`scripts/report_experiment81_all_blocks.py --root RUN_DIRECTORY` reuses saved
all-head gradients for every block, source and tubelet. The 6 November audit
contains 23,040 measurements and 190 montages, with one training-only scale
pooled across all 24 blocks and both arms. Markers indicate automatic native
maxima, not manually supplied liquid coordinates.

`scripts/audit_experiment81_all_blocks_gpu.py --root RUN_DIRECTORY --output NEW_DIRECTORY`
is a bounded single-GPU diagnostic for cold56, cold10, cold13 and the first clean
selection source. It preserves all-block last-prefix patch tokens, patch cosine
matrices, raw patch attention and actual production input gradients. It adds
zero training updates. Raw attention is class-agnostic and has its own fixed
probability scale; it is not labeled as class-specific localization.

The separate input-gradient helper replays prefix cotangents onto input pixels
and sums later-prefix contributions onto earlier observed frames. Tests compare
this gradient against direct end-to-end autograd and check padded-frame exclusion.
The additional cold56 straight-path integrated-gradient diagnostic uses both
mean and blur references, 32 trapezoidal intervals, and at most 64 if completeness
fails. Its acceptance tolerance is max(0.005, 5% of absolute score change).
Failure is reported rather than treated as evidence of a faithful region map.

## Further opt-in controls

`scripts/audit_attention_competition.py` and
`scripts/control_attention_competition.py` compare the original positive
attention-probability derivative with the exact softmax-logit derivative
`A * (g - sum_key A*g)`. This removes sensitivity normal to the probability
simplex, without changing the network. It is an attribution diagnostic, not a
published localization method or training loss. The
[80-source comparison](../reports/experiments/spill_attention_competition_80_20261106/report.md)
records exact reproduction of original gradients, mixed removal-control results,
and persistent qualitative localization failures.

The experimental trainer now also accepts `--clock-policy relative_prefix`.
`RelativeClockTrajectoryModel` subtracts the first observed timestamp inside
the adapter while the four-loss predictor still receives actual time differences.
The default `absolute` policy and original cached trainer remain intact.
Checkpoint loading follows the saved policy; resuming with another policy fails.
This is a separately trained robustness ablation, not a silent inference change
to an absolute-clock checkpoint. The
[matched100-update result](../reports/experiments/spill_relative_clock_80_20261106/report.md)
and its [named curves](../reports/experiments/spill_relative_clock_80_20261106/tsne_curves/README.md)
show faster fit and clock-origin invariance, without solving liquid localization.

## Model-locality and broader source-excluded evaluation

The [context/locality report](../reports/experiments/spill_context_locality_20261106/report.md)
records further inference-only tests. All use explicit Slurm job steps on the
verified allocation and preserve the original checkpoints and objective:

- `audit_spill_dataset_support.py` audits all manifest hashes and acquisition overlap.
- `audit_spill_context_locality.py` preserves systematic core pixels while changing
  distant context, measuring patch embedding, all24 blocks and final norm.
- `evaluate_spill_source_excluded.py` evaluates all76 unique sources outside the
  original80 IDs/hashes, with fixed original training prototypes. Scene exclusion
  and historical test-set independence are not claimed.
- `audit_constant_prefix_features.py` tests encoder variation under synthetic
  constant inputs; it is not a padded training clip or physical-motion benchmark.
- `audit_spill_padding_readout.py` compares all256, content192 and padding64 means
  at fixed heads/prototypes for all80 sources.
- `audit_content_only_attribution.py` checks the content-only inference readout
  through production input gradients and exact feature replay.

Padding-only recognition remains16/16 in both models, while content-only pooling
still misses liquid. This demonstrates distributed contextual class information;
it does not show that the classifier never uses liquid. A regional correction
still needs local features and an independently supported liquid cue. The prior
automatic contrast candidates are not promoted to verified targets.
