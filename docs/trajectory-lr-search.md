# I-JEPA-inspired trajectory learning-rate search

## Source protocol and adaptation

The [I-JEPA paper](https://arxiv.org/abs/2301.08243) trains an image encoder
and predictor at ImageNet scale. Its [official 300-epoch configuration](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml)
sets per-process batch 16 (the log path names global batch 2048), peak learning
rate `0.001`, start rate `0.0002`, final rate `0.000001`, 40 warmup epochs,
and weight decay increasing from `0.04` to `0.4`. The [official optimizer
code](https://github.com/facebookresearch/ijepa/blob/main/src/helper.py) uses
AdamW, excludes bias and one-dimensional parameters from weight decay, and
advances both [warmup/cosine LR and cosine weight-decay
schedules](https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py)
by update. It does not prescribe a universally optimal LR scaling formula.

This project's frozen V-JEPA 2 encoder feeds a much smaller trainable temporal
adapter. The tuner preserves AdamW, bias/norm exclusion when weight decay is
nonzero, update-wise warmup and cosine decay, and fixed compute budgets. It
keeps the adapter's existing weight decay default at zero and its epsilon at
`1e-12`; copying the large-backbone `0.04 -> 0.4` decay would change a
different optimizer protocol. `--weight-decay` makes a separate experiment.
AdamW betas are `(0.9, 0.999)`, matching the PyTorch defaults used by the
project and by I-JEPA's unmodified AdamW constructor.

The [linear scaling rule with warmup](https://arxiv.org/abs/1706.02677) was
developed for synchronous large-batch SGD. [SDE analysis of
Adam](https://arxiv.org/abs/2205.10287) and [random-matrix
analysis](https://jmlr.org/papers/volume23/20-1258/20-1258.pdf) motivate a
square-root alternative under particular assumptions; the SDE rule also
changes other optimizer settings, so testing only its LR exponent is an
empirical candidate, not a claim of SDE invariance. Constant LR is the
reference policy. None is assumed to transfer universally. The tuner tests
all three at larger batches, then refines the LR numerically. At the reference
batch of 32, all three formulas produce the same rate and cannot be
distinguished.

Global batch means unique **source videos per optimizer update**, with two
views each. It equals `per_rank_microbatch × world_size × accumulation_steps`.
An entire effective batch enters one global ViCo and temporal loss, then DDP
synchronizes gradients. Two `DistributedSampler`s preserve class balance,
each receives the true global rank and world size, and `set_epoch` is called
before each pass. Their padded repeats are excluded. All ranks audit each
update for unique source IDs and class counts. Only rank 0 writes outputs.

The fixed source-held-out validation set is used only for selecting hyperparameters.
It is not a final independent test, and source names alone do not prove scene
independence. The same fixed identity and mirror views and final curriculum
weights are used for every validation evaluation. Coarse and full stages use
equal source-exposure budgets within their respective stages; the number of
optimizer steps and warmup steps changes with batch size. Divergent validation
loss or nonfinite/excessive gradient norm prunes a trial.

## Measured two-H100 exploratory run

Artifacts: [comparison report](../reports/experiments/trajectory_lr_search_real12_20261021/tuning_v5/comparison.md)
and [machine-readable report](../reports/experiments/trajectory_lr_search_real12_20261021/tuning_v5/report.json).
The data are 156 unique videos (78 per class) in a 12-frame causal feature
cache with identity and whole-clip mirror views, extracted on two H100 GPUs.
Training used 124 sources; the selection set used 32 sources (16 per class).
The 19-trial budget was 512 source exposures for coarse candidates and 1024
for full trials. Candidate batches were 32, 64, and 120; gradient accumulation
was 2. The complete per-step curves, norms, and padding counts are in
`trials.json` beside the report.

| Global source batch | Peak LR | Full-budget validation loss |
|---:|---:|---:|
| 32 | 0.0005 | 0.04231, 0.04540, 0.04179, 0.03822 |
| 32 | 0.001 | 0.05259, 0.04013, 0.04111, 0.07244 |
| 64 | 0.0005 / 0.001 / 0.002 | 0.11972 / 0.14004 / 0.10905 (one seed each) |
| 120 | 0.001 / 0.00194 / 0.00375 | pruned in coarse stage for rising validation loss |

The lowest observed four-seed mean validation loss is **0.04193** at global
source batch **32** and peak LR **0.0005**. Its four-seed standard deviation is
about 0.003. Batch 32 is the reference point, so the evidence does not select
constant, square-root, or linear scaling as a law. The selected checkpoint is
the best individual seed among that configuration's four full-budget runs.
This is a finite search result for the 12-frame cache, not a mathematical
global optimum or a choice validated for the standard 64-frame eight-view run.

The selected configuration's mean temporal prediction loss was approximately
**0.03905**, versus the frozen persistence baseline **0.03800**. It therefore
does **not** establish useful future prediction on this selection set, despite
its low total and semantic losses. No spatial-localization or unseen-scene
claim follows from the result.
