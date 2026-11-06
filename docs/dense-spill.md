# Decision-connected dense spill experiments

The default pipeline remains the frozen V-JEPA 2 causal trajectory with
`L_ViCo + 0.1 L_align + L_predict + 0.1 L_keep`. Its original checkpoints and
teachers are preserved. The opt-in path below adds a **separate spatial decision
readout**, with no manual spatial annotations, MIL, learned spatial selection or
segmentation decoder. A dense map is not automatically a verified liquid mask.

The strict V-JEPA 2.1 loader consumes all 302 EMA encoder tensors, including the
image tokenizer and all hierarchical norms. Its video path uses corrected RoPE
frequency pairs and spatial interpolation. The implementation matches the
official FP32 encoder exactly on the recorded two-H100 parity checks. The
original V-JEPA 2 encoder/rotation is unchanged. Official weights:
`checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt`, SHA-256
`7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`.

Run from the repository root on a two-GPU node, using a new output directory.
Set `RUN_PYTHON` to the environment's Python. In the current environment use
`/home/olemou/miniconda3/envs/torchgpu/bin/python` and include its `lib` directory
in `LD_LIBRARY_PATH`. The scripts require `PYTHONPATH=src:scripts`.

```bash
RUN_DIR=reports/experiments/my_dense_spill_run
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/extract_dense_spill_features.py --model vjepa21 --output "$RUN_DIR/vjepa21_384.pt"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/extract_dense_spill_features.py --model vjepa2 --output "$RUN_DIR/vjepa2_384.pt"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" --output "$RUN_DIR/vjepa21_reference"
"$RUN_PYTHON" scripts/prepare_dense_candidates.py \
  --reference "$RUN_DIR/vjepa21_reference" --output "$RUN_DIR/vjepa21_consensus"
```

Extraction preserves the entire field of view, uses a 24x24 patch grid and
encodes every observed prefix independently. Start/end reference windows come
only from training-clean sources; they do not move selection sources into
fitting. Normal-reference calibration excludes a queried clean source from its
nearest matches. The original 64/16 source split is retained.

The strict two-geometry/three-MAD seed intersection is retained as an ablation.
The consensus policy requires absolute q99 novelty in both geometries/views,
with robust spatial excess in at least one geometry. It never forces top-k
area. This policy was selected from training target-coverage diagnostics, not
validated as true liquid segmentation.

Training requires same-frame FG and BG candidates in at least half the positive
training sources. Sources lacking both abstain from the local term. Increasing
the weight or inventing labels to bypass this check is not a correction. The
historical automatic trainer now also rejects insufficient within-spill support.

```bash
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" \
  --candidates "$RUN_DIR/vjepa21_consensus/candidates.pt" \
  --output "$RUN_DIR/dense_control" --control
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" \
  --candidates "$RUN_DIR/vjepa21_consensus/candidates.pt" \
  --output "$RUN_DIR/dense_joint"
```

Both pilots use 100 full-batch updates, seed 239, identical initialization,
AdamW LR 1e-3/epsilon 1e-12/zero decay, clipping 5 and state reset before update
21. The **new dense readout** objective is

`L_dense = L_ViCo(z_dense_episode) + lambda L_source-balanced partial BCE`.

The original four-loss trajectory is an immutable control, not silently
retrained or included in this dense head's optimization. The local term gives
equal weight to clean-BG, cold-BG and cold-FG strata; within each stratum sources
are equally weighted. Only cold frames containing both candidates contribute.
Distributed global counts normalize the loss without duplicated samples.
Lambda is measured from training-only initial gradients (target local/semantic
norm ratio 0.5, bounds 0.01–10); the control uses zero.

One shared pointwise embedding supplies both patch scores and the normalized
fixed-mean episode embedding. Training-class prototypes are detached statistics
refreshed after each update; selection videos cannot affect them. Signed patch
contributions average exactly to the same episode margin. Changing selection
features leaves trained weights/prototypes unchanged in the integration test.

An additional cold-only candidate teacher is available:

```bash
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/prepare_thermal_candidates.py --root "$RUN_DIR" --output "$RUN_DIR/thermal_candidates"
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" \
  --candidates "$RUN_DIR/thermal_candidates/candidates.pt" --output "$RUN_DIR/dense_thermal"
```

This teacher assumes visible cold regions are apparently darker than local
surroundings. It measures 31/63-pixel local contrast, compensates grayscale
contrast gain using an inter-decile range and requires support in both actual
tubelet frames. Foreground exceeds training-clean q99.9; background must have
low cold contrast and a normal-compatible dense-feature match. It uses no
location/area prior and never changes encoder normalization. **Dark objects,
clothes and shoes remain confounders; these targets are unverified.** Warm and
thermally neutral spills are unsupported by this teacher.

The additional `--cold-pixel-input` readout appends one local contrast value to
each frozen 1024-D patch feature. This gives the decision an explicit pixel cue
when a contextual token alone does not preserve the liquid's dark appearance.
It requires candidates prepared with `--encoder-aligned`; those descriptors use
the exact CPU resize/normalization boundary of the feature extraction pipeline.
Scale is fixed from training-clean q99.9. Input-removal evaluation recomputes
the descriptor from the changed encoder pixels, never reusing an unperturbed
descriptor. This is a cold-contrast-assisted model, not pure V-JEPA attention.

```bash
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/prepare_thermal_candidates.py --root "$RUN_DIR" \
  --output "$RUN_DIR/thermal_encoder_input_candidates" --encoder-aligned
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" \
  --candidates "$RUN_DIR/thermal_encoder_input_candidates/candidates.pt" \
  --output "$RUN_DIR/pixel_control" --cold-pixel-input --control
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_dense_spill.py --cache "$RUN_DIR/vjepa21_384.pt" \
  --candidates "$RUN_DIR/thermal_encoder_input_candidates/candidates.pt" \
  --output "$RUN_DIR/pixel_joint" --cold-pixel-input
```

```bash
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/control_dense_spill_inputs.py --root "$RUN_DIR" --output "$RUN_DIR/input_controls"
"$RUN_PYTHON" scripts/report_dense_spill.py --root "$RUN_DIR"
```

Input controls decode videos afresh and re-encode every prefix. About 13.5% of
content cells are removed with mean/blur fills, comparing high/low evidence to
three seeded equal-area random masks. These are offline evaluation interventions,
not training erasers. `--cases dense_thermal` evaluates the additional readout.

For its gallery, use `report_dense_spill.py --joint-case dense_thermal
--candidate-dir thermal_candidates --output-name figures_thermal`.
The renderer preserves native cells and uses one training-only scale across
sources, frames and compared arms. Positive/negative contributions are shown
separately. They are not raw self-attention, probabilities or masks. Clean frames
need not highlight a spill. Source-selection performance is not independent-scene
generalization, and lower local loss or sharper maps do not prove liquid accuracy.

For the cold-input comparison use `--control-case pixel_control --joint-case
pixel_joint --candidate-dir thermal_encoder_input_candidates --output-name
figures_pixel` in the reporter, and `--cases pixel_control pixel_joint` in the
input-control script. Both allocated GPUs own distinct sources and contribute;
only global rank zero writes consolidated artifacts.

The final controlled ablation uses `--unnormalized-patches` together with
`--cold-pixel-input` in both control and joint training commands, writing
`amplitude_control` and `amplitude_joint`. The pointwise projector's amplitudes
are then preserved; the episode mean is still normalized. Pooling remains a
fixed mean, with no additional instance-selection module. For this comparison,
pass those case names to the input-control and reporting scripts and write a
new gallery directory such as `figures_amplitude`.

This tests a specific conflict: for unit patch vectors, prototype difference
norm `D`, background projections at most `-gamma`, and positive fraction `a`,
the raw mean margin is at most `a*D-(1-a)*gamma`. A positive decision requires
`a > gamma/(D+gamma)`. Sparse foreground and confident background can therefore
conflict with this readout. Preserving amplitudes removes that particular
bound, but supplies no new evidence of liquid identity. The completed final
run improves selection classification to 15/16, while wrong-region activations
remain. None of these experiments is adopted as a verified liquid localizer.

The [implementation report](../reports/experiments/spill_dense_fix_80_20261022/report.md)
contains all seven results, shared-scale galleries, loss curves, t-SNE,
input-level controls and limitations. Keep the original four-loss trajectory
as the supported recognition/temporal output until a local branch passes the
liquid-focus requirement independently.
