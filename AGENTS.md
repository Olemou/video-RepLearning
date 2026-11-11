# Repository instructions for spill video learning

These instructions apply to the entire repository. The current authoritative
pipeline is the causal trajectory pipeline implemented by
`scripts/extract_causal_video_features.py`, `scripts/train_trajectory.py`,
`src/vision_model/training/trajectory.py`, and
`src/vision_model/training/vico.py`.

`CLAUDE.md` is the Claude entry point; this file is the canonical repository
guidance. Experimental local readouts do not replace the pipeline above.

## Current evidence and working constraints

- The 22 October 2026 local experiment is implemented, but **spill localization
  acceptance failed**. Inspected cold56, cold10 and cold13 outputs still miss
  visible liquid and respond to unrelated structures. Do not report this issue
  as solved because tests pass, losses decrease or t-SNE separates classes.
- The current requested approach uses no manual spatial annotations, MIL or
  segmentation decoder. Do not make verified points a prerequisite for running
  this annotation-free research path, or silently introduce those methods.
  This constraint does not remove the need for independent localization evidence.
- User-supplied images and descriptions define qualitative expectations, not
  training annotations. Do not convert them into hand-selected ROIs, pseudo
  ground truth or a desired heatmap template.
- Keep the original four-loss trajectory, frozen teacher and saved checkpoints
  intact. New local objectives must be explicit opt-in experiments, with their
  own outputs, controls and reports.
- Read the [scene-confounding audit](reports/experiments/spill_scene_confound_audit_20261022/report.md)
  and [local experiment report](reports/experiments/spill_local_relative_80_20261022/report.md)
  before proposing another attention correction. Mean brightness alone separated
  all 64 training and 16 selection sources in the audit; rank-normalized coarse
  layout also separated all 16 selection sources. These demonstrate available
  shortcuts, not exclusive causal dependence of the neural model on brightness.
- Read the [6 November model-locality audit](reports/experiments/spill_context_locality_20261106/report.md)
  before another correction. Reliable localization still fails after full
  fine-tuning, clock, attribution and content-only pooling controls. Preserve
  these negative results; do not repeat an expensive run solely to improve the
  semantic curve or the appearance of its heatmap.

## Latest controlled evidence: 6 November 2026

- [Experiment 81 diagnosis](reports/experiments/experiment_81_execution_20261106/diagnosis.md):
  matched frozen/full arms completed 100 updates each. The full arm updated all
  24 encoder blocks, patch embedding, final norm and trajectory model. Both
  classify 16/16 reused selection sources correctly; neither establishes
  reliable liquid focus. Freezing alone is not a sufficient explanation of
  these failures under the tested protocol.
- [All-block inspection](reports/experiments/experiment_81_execution_20261106/all_blocks_report.md):
  some individual blocks have liquid-adjacent peaks, but no reliable common
  layer or complete liquid extent was demonstrated. Do not select a favorable
  block from cold56/cold10/cold13 and call localization solved. These examples
  remain qualitative diagnostics, not training annotations.
- [Normalized-attention comparison](reports/experiments/spill_attention_competition_80_20261106/report.md):
  exact softmax-logit sensitivity is `A * (g - sum_key(A*g))`, with `g=dS/dA`.
  This answers a different question from independent probability gradients.
  Its replay passed, but localization still failed and removal-control gains
  depended on fill choice. It is not original LeGrad, AttnLRP or a new loss.
- [Elapsed-clock ablation](reports/experiments/spill_relative_clock_80_20261106/report.md):
  the opt-in `--clock-policy relative_prefix` subtracts the first observed
  timestamp inside the adapter; real prediction deltas and the four losses
  remain intact. It improves tested clock-origin invariance, but does not fix
  liquid focus or the additional clean acquisition counterexample. The default
  remains `absolute`; loaders must honor the recorded policy and resume must
  reject policy changes. Do not silently reinterpret an old checkpoint.
- [Model-locality tests](reports/experiments/spill_context_locality_20261106/report.md):
  unchanged probe pixels and patch embeddings can have substantially changed
  late tokens after distant-context interventions. Padding-only contextual
  tokens classify 16/16 selection sources in both frozen/full arms. Their
  pre-attention embeddings are identical across sources. This demonstrates
  distributed class information, not absence of every liquid cue. Content-only
  pooling preserves recognition but still misses liquid; it is not an accepted
  standalone focus correction. High token norms alone do not explain the
  inspected wrong peaks.
- Constant-input prefix features also vary as encoded context grows. Prediction
  below persistence measures the evaluated representation trajectory, not
  necessarily physical motion or spill dynamics. The synthetic test is not
  dataset padding, a supported training augmentation or a replacement target;
  its variation ratios are not an additive motion/context decomposition.
- Prioritize a genuinely local feature path plus independently supported liquid
  evidence before another regional objective. The earlier automatic thermal
  candidates have already failed liquid identity/coverage. Same-camera,
  exposure-consistent dry/spill recordings or calibrated thermal cues are
  potential annotation-free evidence, not currently verified resources. Require
  registration, occlusion and capture-status checks before treating them as
  targets; never assume an existing first frame is dry.
- [Named loss curves and t-SNE](reports/experiments/spill_relative_clock_80_20261106/tsne_curves/README.md)
  compare the frozen-backbone absolute/elapsed runs, not full-backbone training.
  Preserve their labels, raw/weighted term distinction, source/split markers
  and saved numeric histories. New probes added no training updates.

## Scientific objective

Learn a video representation that separates class 0 (`clean`) from class 1
(`cold` or `warm` spill), preserves temporal changes, and can eventually support
spatial spill localization. Treat these as separate claims:

1. a low semantic loss demonstrates fit of the supplied class labels;
2. temporal prediction below the persistence baseline demonstrates prediction
   of the evaluated representation trajectory; physical dynamics need separate
   evidence because prefix length/context can change features without motion;
3. spill localization requires spatial evidence and localization validation;
4. generalization requires videos or scene groups excluded from every training
   operation, including representation fitting and prototype construction.

Never infer claim 3 or 4 from claim 1 alone.

## Data and labels

- The authoritative binary mapping is `clean -> 0`, `cold/warm -> 1` in
  `src/vision_model/data/labels.py`.
- Reject byte-identical source videos before training. A view, frame, or
  augmentation derived from one source must remain in the same split as that
  source.
- The historical 14-video run is an all-data optimization study. Source-excluded
  prototypes do not make it a held-out test when the source trained the model.
- The current 80-source study has 40 clean and 40 cold videos: 64 training sources
  and 16 repeatedly used source-selection videos. There are no warm examples in
  this study. Keep this selection set distinct from an independent test set.
- The [whole-manifest audit](reports/experiments/spill_context_locality_20261106/dataset/results.json)
  finds 173 entries but 156 byte-unique sources, balanced 78 clean / 78 cold.
  Seventeen redundant clean entries occur in 13 duplicate groups, with no
  cross-label byte duplicates. The current 80-source split remains byte-unique.
  Hashes do not detect every re-encoded duplicate or shared physical recording.
- Of the unique sources, 77 clean videos are 384x288, one clean video
  (`clean_folderImage81`) is 640x480, and all 78 cold videos are 640x480.
  Resolution is an acquisition-cohort indicator, not a verified scene ID or
  proof that resolution alone causes the trained decisions.
- The [76-source evaluation](reports/experiments/spill_context_locality_20261106/source_excluded/results.json)
  excludes all original 80 IDs and byte hashes and holds training prototypes
  fixed: frozen absolute 76/76, full absolute 75/76, frozen elapsed 75/76;
  all three AUROCs are 1.0. These are 38 clean and 38 cold sources. Scene
  exclusion and historical test-set independence remain unverified. Do not
  replace recognition metrics with a claim of spatial generalization.
- Trusted scene/session groups and spill masks are unavailable. Do not invent
  scene IDs, assert scene-held-out generalization or report mask-based metrics
  from unverified automatic candidates. Record source hashes and split provenance.
- Do not assume the first frame or temporal median is clean: wall liquid is
  already visible at frame zero in cold56. Clean videos from different scenes
  are not automatically valid pixelwise normal references.
- Preserve `temporal_context_policy=past_only_prefix_last_tubelet_v1` for future
  prediction. Features from a full video are not valid causal-prefix targets.
- Repeated tail frames are padding and must not contribute to attention,
  pooling, alignment, or prediction losses.

## Pretrained V-JEPA 2 model

- Use checkpoint `facebook/vjepa2-vitl-fpc64-256` unless an experiment explicitly
  records another model ID.
- Load with `load_pretrained_encoder()`. It validates the architecture, consumes
  every Hugging Face encoder key, maps Q/K/V explicitly, and uses strict state
  loading. Do not silently accept missing or unexpected weights.
- The authoritative default is parameter-efficient fine-tuning: V-JEPA 2 is frozen;
  the causal trajectory adapter and its heads are trained. Do not describe it as
  end-to-end backbone fine-tuning.
- The encoder accepts `[B,3,T,H,W]`, uses tubelets of two frames and 16x16
  patches, and returns 1024-dimensional tokens. At 256x256 there are 256 spatial
  tokens per tubelet.
- Prefix feature extraction must encode each observed prefix separately and
  average only the final tubelet's 256 spatial tokens. Reusing tokens from a
  full-video encode would leak future context.
- The trainable temporal adapter receives `[video,time,1024]`. Spatial tokens
  have already been averaged, so its attention can select times but cannot learn
  a patch-level spill mask.
- Token indexing surviving the encoder does not imply a local receptive field.
  Final tokens over both content and letterbox cells can carry global class
  information. Do not treat their feature similarity or class score as a
  pixel-level material label. Preserve the default extraction; alternate
  content-only pooling and spatial masks require explicitly recorded experiments.

## Spill-preserving preprocessing and augmentation

- Use `OfficialVJEPA2Transform(preserve_field_of_view=True)`.
- Resize the complete frame into the 256x256 checkpoint grid and letterbox with
  channel-mean pixels. Do not center-crop a possible spill.
- Keep checkpoint normalization unless a new experiment explicitly validates
  and records a replacement: mean `(0.485,0.456,0.406)`, standard deviation
  `(0.229,0.224,0.225)`, after one `1/255` rescaling.
- The supported augmentation is `ThermalAugmentor` in
  `src/vision_model/data/transforms.py`. One parameter set is shared by the
  entire clip:

  `x_aug = round(b * (c*x + (1-c)*mu))`,

  where `b ~ U(0.9,1.0)`, `c ~ U(0.85,1.0)`, `mu` is the per-frame/channel
  spatial mean, and whole-clip horizontal flip has probability 0.5.
- Do not introduce crop, erase, cutout, elastic deformation, temporal reversal,
  frame-local random geometry, or an assumed protected center without verified
  spill masks. Any of these can delete the target or change its dynamics.
- Sampling parameters must be reproducible with a local `torch.Generator` and
  stored with the feature cache.
- The current eight augmented views are a finite pre-encoded bank. Do not call
  them fresh augmentation at every epoch.

## Authoritative combined loss

The training objective is

`L = L_ViCo + 0.1 L_align + 1.0 L_predict + 0.1 L_keep`.

`L_ViCo` is ViCo-LWIR Eq. 10 as implemented in
`src/vision_model/training/vico.py`. For normalized whole-video embeddings
`z_i`, cosine `s_ik=z_i^T z_k`, temperature `tau=0.1`, and `epsilon=1e-6`, each
positive pair has its own positive-versus-negative competition:

`ell_ij = softplus(log(sum_{k:y_k!=y_i} alpha_ik exp(s_ik/tau) + epsilon)
                   - (s_ij/tau + log(w_ij)))`.

Average `ell_ij` over positives for each anchor and then over anchors. This is
not shared-denominator SupCon and has no `log(number_of_positives)` floor. Do
not subtract an entropy constant or relabel another objective as Eq. 10.

- `alpha_ik` follows Eq. 7, including its square-root negative-count factor.
- `w_ij` follows the uncertainty-rank curriculum in Eq. 8. Mirrored or augmented
  views of the same source retain weight one.
- Mining weights are detached selection statistics. Do not optimize through
  ranks or hard-pair weights.
- Compute the small semantic loss in float64 for numerical stability.
- The 20-epoch curriculum changes pair weights. Evaluation curves use fixed
  final-curriculum weights at every recorded epoch so points are comparable.

`L_align` is Smooth L1 between matched-view dynamics states at valid times.
`L_predict` predicts normalized frozen future teacher features at horizons
1, 2, and 4 tubelets using the real timestamp delta. Normalize its summed error
by the number of valid scalar targets. `L_keep` is Smooth L1 between adapted
states and normalized frozen teacher states. Targets remain detached.

## Training protocol

- Default run: 100 epochs, where one epoch is one full-batch optimizer update.
- Seed 239, AdamW, learning rate `1e-3`, weight decay 0, epsilon `1e-12`, and
  gradient norm clipping at 5.
- Reset AdamW state once before update 21, immediately after the mining
  curriculum. Record this reset in reports.
- With an augmentation bank, select two distinct, deterministically seeded
  views per source per epoch. Evaluate only the fixed identity and whole-clip
  mirror views.
- Save the cache metadata, model state, optimizer state, settings, history,
  representations, view exposures, and exact number of optimizer updates.
- Never call a fixed 100-epoch run fully converged only because its semantic term
  is near zero. Report the semantic and temporal terms separately and state
  whether their slopes have flattened.

## Evaluation and attention maps

- Report minimum same-class cosine, maximum cross-class cosine, minimum
  per-anchor separation, worst positive-pair loss, and prototype accuracy.
- Compare future prediction with the frozen persistence baseline.
- Keep training-fit metrics separate from source-held-out and scene-held-out
  metrics.
- Raw V-JEPA block-23 attention is frozen in the authoritative default and cannot
  have learned spill focus from its class labels. The separate full-fine-tuning
  arm has updated weights; identify the actual arm/checkpoint for each figure.
- The final-block diagnostic is one-block attention-times-gradient, not a
  segmentation map, Grad-CAM, full relevance propagation, or pixel probability.
  Blue means below the visualization threshold, not `no spill`.
- Use all heads, a shared color scale, no per-frame renormalization, no chosen
  favorable head, and no manual spill ROI when producing audit figures.
- Validate a localization claim with real masks and metrics such as IoU or
  pointing accuracy on unseen scenes, plus input-level removal controls. Generic
  attention and a visually pleasing heatmap are insufficient.
- To make attention learn spill location, retain spatial tokens in a trainable
  local branch and train it with verified spatial supervision or a validated
  localization objective. Unfreezing the final backbone block alone does not
  rule out scene shortcuts.
- Cached or detached features have no end-to-end gradient path back to V-JEPA.
  Any separate differentiable replay must record its target, preprocessing,
  checkpoint, layer, tensor shapes and frame/timestamp alignment.
- Instrumented explicit-attention BF16 forwards can differ from production
  SDPA. Publish both the production discrepancy and self-replay error; matching
  predictions alone does not establish identical gradients. The incomplete
  integrated-gradient checks in Experiment 81 are preserved as failed numerical
  checks, not accepted localization evidence.
- Padding-only/content-only readout substitutions are internal interventions,
  not real-input removal. Distant-context probes preserve systematic core cells
  but alter large image areas; report that scope. Automatic peak crosses and
  probe tiles are predictions/experimental geometry, never manual spill labels.
- A new local detector's output is not an explanation of the original trajectory
  decision. Label temporal attention, spatial attention, signed contributions,
  gradients and local detector scores according to what was actually computed.
- Inspect native maps and representative videos, including the known failures,
  before declaring improved focus. Keep no-spill false activations separate from
  spill localization; a clean video need not highlight any foreground region.
- Input-removal controls must recompute the complete affected feature path and
  reproduce the unperturbed saved output first, including cache precision and
  batching. Compare matched random removals and multiple fill choices; feature
  masking alone does not establish input-level faithfulness.

## Optional local thermal research branch

See [local thermal documentation](docs/local-thermal.md) for commands and the
implementation in `src/vision_model/training/local_thermal.py` and
`scripts/{prepare_local_thermal,prepare_local_pairs,train_local_thermal,control_local_thermal,report_local_thermal}.py`.

- This separate path uses full-content 384x288 grayscale frames, eight local
  measurement channels, and an FP16 cache shaped `[80,2,12,8,36,48]`. Its
  pointwise and local-convolution heads have maximum radii 39 and 55 pixels.
  Official V-JEPA preprocessing and the original temporal teacher are unchanged.
- The experimental objective is
  `L_local = L_absence(clean) + lambda L_pair(candidate, local_ring)`.
  Pair ranking is `softplus(1 - (mean_region a - mean_ring a))`, with
  source-balanced reduction and detached automatic selection statistics.
  This proposed objective is **not ViCo-LWIR Eq. 10** and is not a replacement
  name for the authoritative four-loss objective.
- Automatic components and rings are unverified. Report per-source support,
  abstentions and actual local-loss gradients before training. Complete pair
  coverage does not prove liquid coverage; candidate noise and incomplete
  extent remain observed problems. Do not label every positive-video patch spill.
- Temporal consistency is disabled until registration and occlusion handling
  are validated. Do not present unimplemented correspondence checks as passed.
- Distant-context checks and baseline replay passed in the latest run, while
  real photometric invariance and reliable liquid identity did not. More updates
  reduced the pair loss without solving localization. Mean-region ranking does
  not constrain every liquid cell, and its score offset is not a calibrated
  spill probability.
- Use each model's own training-clean q99.5 threshold and one training-only
  magnitude scale across compared arms, sources and times. Keep raw scores;
  do not blur or normalize each frame to manufacture contrast. This threshold
  is a false-alarm policy, not verified liquid/background segmentation.
- The latest gallery is `figures_conv/` under the local report directory;
  `figures_calibrated/` and `figures_1000/` are valid earlier comparisons.
  The initial `figures/` used an inappropriate shared threshold and is superseded.
  Accepted removal controls are the three `input_controls*_replay/` directories.
  Preserve superseded artifacts and identify them rather than overwriting them.
- Prioritize evidence that distinguishes liquid from local appearance: inspect
  target validity and, where available, matched-camera clean clips, raw thermal
  calibration and dark non-liquid counterexamples before expensive retraining.
  None of these is currently a validated localization solution.

## Distributed execution and reproducibility

- Inspect existing caches, target support and gradient connectivity before
  launching expensive GPU training. Use an actual allocated GPU job for H100
  work and record the allocation, GPU types, seed and exact optimizer updates.
- Recheck the live Slurm allocation rather than reusing an old job ID. Prefer
  explicit `srun` steps and record the actual job/step, host, visible GPU count
  and timezone. The latest locality probes used one H100 on rg32101, allocation
  25829624; this is historical provenance, not permission to assume it is still
  allocated. Do not manually spoof `SLURM_JOB_ID` to satisfy a runtime guard.
- Give ranks distinct sources. Reduce losses and metrics using global valid
  counts, including ranks with no positive pairs; gather evaluation without
  duplicated sampler padding. Only global rank 0 writes consolidated artifacts.
- Fit thresholds, prototypes and selection rules on training sources only.
  Preserve identical source splits, views, seeds and budgets in matched controls.
- Save source snapshots, settings, hashes and executed commands. A dirty
  workspace's HEAD alone does not identify the code used for a figure. Record
  any missing historical snapshots as a provenance limitation.

## Verification and documentation

- Run focused tests for edited modules, followed by the full pytest suite for
  changes to model, loss, transforms, caching or attribution. Prefer
  `.venv/bin/python -m pytest -q` when that environment exists. It was absent in
  the latest run; the verified environment was:

  ```bash
  LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib \
  PYTHONPATH=src:scripts OMP_NUM_THREADS=4 \
  /home/olemou/miniconda3/envs/torchgpu/bin/python -m pytest -q
  ```

- The recorded local run passed 236 tests with 4 skips; see its
  [verification record](reports/experiments/spill_local_relative_80_20261022/verification.json).
  This is historical evidence, not a substitute for running checks after code
  changes. Explain skips and distinguish CPU tests from actual GPU execution.
- The latest [model-locality verification](reports/experiments/spill_context_locality_20261106/verification.json)
  records 257 passed, 4 skipped, no failures. One skip lacked CUDA on the login
  test node; three required an unavailable generated dataset. Actual H100 probes
  ran separately in verified Slurm steps. This is historical software evidence,
  not a localization acceptance result. Keep new unit tests independent of
  network downloads; use processor fixtures and verify real preprocessing in
  the allocated diagnostic path.
- Documentation-only changes need link/consistency checks and `git diff --check`,
  not a new training run.
- Preserve existing experiment directories. Write a new output directory for a
  changed dataset, loss, augmentation, seed, or trainable parameter set.
- A report must distinguish measured results, mathematical implications, and
  unresolved scientific claims.
- Keep the root `README.md` synchronized with current executable defaults.
  Historical objectives belong in their report directories and must not be
  presented as the current loss.

