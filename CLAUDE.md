# Claude guidance for spill video learning

Read and follow [AGENTS.md](AGENTS.md), the canonical repository instructions,
before editing code or launching experiments. This entry point adds no separate
loss definition or training default.

## Scientific status

The goal is to distinguish clean videos from spills, preserve causal temporal
information and ultimately localize the actual liquid. These are separate
capabilities. Good t-SNE, low semantic loss and passing software tests do not
establish spill localization or unseen-scene generalization.

The latest 80-video study contains 40 clean and 40 cold sources, split into 64
training and 16 repeatedly used source-selection videos. Trusted scene groups
and spill masks are unavailable. Do not call that selection set an independent
test set or calculate ground-truth localization metrics from automatic targets.

**Reliable spill focus remains unsolved.** The latest local branch still misses
visible liquid in cold56, cold10 and cold13 and highlights unrelated structures.
The full-backbone, elapsed-clock, attention-logit and content-only pooling
controls also failed to establish reliable localization. Do not repeat an
expensive global training run solely because class separation is good.
Read these records before proposing another correction:

- [Scene-confounding audit](reports/experiments/spill_scene_confound_audit_20261022/report.md):
  simple brightness/layout shortcuts separate source labels; automatic targets
  can be missing or incorrect.
- [Local experiment report](reports/experiments/spill_local_relative_80_20261022/report.md):
  locality and reproducible replay passed, but liquid identity and strong real
  photometric invariance did not. Longer training did not solve localization.
- [Local branch commands and objective](docs/local-thermal.md).
- [Experiment 81 diagnosis and all-block findings](reports/experiments/experiment_81_execution_20261106/diagnosis.md):
  both frozen/full arms completed 100 updates; updating all 24 encoder blocks
  did not solve the known misses. Some individual blocks respond near liquid,
  without a reliable common layer or full liquid extent.
- [Normalized-attention audit](reports/experiments/spill_attention_competition_80_20261106/report.md):
  probability and softmax-logit derivatives answer different questions; neither
  is a segmentation map. The new derivative did not solve focus.
- [Elapsed-clock ablation](reports/experiments/spill_relative_clock_80_20261106/report.md):
  removing file-origin dependence improved tested invariance, not localization.
- [Latest model-locality investigation](reports/experiments/spill_context_locality_20261106/report.md):
  distant context changes late features despite identical local pixels;
  padding-only contextual tokens classify 16/16 reused selection sources in
  both frozen/full arms. Content-only pooling still misses liquid. These tests
  demonstrate distributed class information, not that every liquid cue is absent.

The whole manifest has 173 entries but 156 byte-unique sources; the original
80-source split remains byte-unique. On 76 additional sources excluded by ID
and byte hash, frozen absolute recognition is 76/76, full absolute and frozen
elapsed recognition are 75/76, with AUROC 1.0 for all three. This remains source
evaluation with unverified scene exclusion and historical test independence.
Nearly all clean/cold sources belong to different native-resolution cohorts;
resolution is not a verified scene label or proof of exclusive model dependence.

Constant-input features change as prefixes grow. Prediction below persistence
therefore does not by itself demonstrate liquid motion. Synthetic constant-input
tests are not dataset padding or revised training targets. A norm outlier,
strong recognition, or successful gradient replay is also not liquid identity.

## Pipeline and scope

The authoritative path is `scripts/extract_causal_video_features.py`,
`scripts/train_trajectory.py`, `src/vision_model/training/trajectory.py` and
`src/vision_model/training/vico.py`. Keep its objective:

`L_trajectory = L_ViCo(z_episode) + 0.1 L_align + L_predict + 0.1 L_keep`.

The default encoder is frozen `facebook/vjepa2-vitl-fpc64-256`, loaded strictly
through `load_pretrained_encoder()`. Preserve official full-field preprocessing,
past-only prefix extraction and valid-frame masks. The temporal adapter receives
`[B,T,1024]` after spatial averaging; its attention cannot directly select image
patches. Cached/detached features do not provide a classifier-to-backbone gradient
path. Raw frozen backbone attention cannot have learned focus from these labels.

Separate Experiment 81 paths support full fine-tuning and the opt-in
`--clock-policy relative_prefix`. The authoritative frozen default and absolute
clock policy remain intact. Load checkpoints using their recorded policy and
reject resume-policy mismatches. The original future predictor retains real
time deltas; do not silently change its teacher or causal-prefix policy.

Final patch tokens are contextual, including positions over artificial
letterboxing. Removing padding from pooling did not fix the observed focus.
Explicit-attention BF16 outputs can differ from production SDPA; disclose that
discrepancy separately from self-replay. Failed integrated-gradient completeness
checks are not accepted as faithful localization maps.

The opt-in local thermal branch has its own objective,
`L_local = L_absence(clean) + lambda L_pair(candidate, local_ring)`.
It is a separate detector, not original trajectory attribution or ViCo Eq. 10.
Its automatic targets are unverified; never equate target fit with liquid fit.

The user requests **no manual spatial annotations, MIL or segmentation decoder**.
Do not require manual points to continue annotation-free research. Do not turn
example heatmaps or descriptions into training ROIs. Do not assume a video's
first frame is clean. Preserve original teachers, checkpoints and figures; put
changed experiments in new directories.

The next regional correction needs local features and independently supported
liquid evidence. Earlier automatic thermal components/rings failed identity and
coverage; do not relabel them as verified targets. Same-camera/exposure dry/spill
recordings or calibrated thermal cues are possible annotation-free evidence,
not established resources. Validate capture status, registration and occlusion
before using them, and preserve static-liquid cases. No code-only focus fix has
been validated on the current data.

## Working and verification rules

- Diagnose representation loss, target support, gradient paths and scene
  shortcuts before expensive retraining. Preserve matched splits, seeds, views
  and budgets for controls.
- On DDP, assign distinct sources, aggregate global valid counts, avoid duplicate
  evaluation samples and restrict consolidated writes to global rank 0.
- Verify a live GPU allocation before execution; use explicit Slurm job steps
  and record the actual job/step and timezone. Historical allocation IDs are
  not evidence of current availability; do not spoof `SLURM_JOB_ID`.
- Name the plotted quantity precisely. Inspect native maps and known failure
  cases. Use all heads for the specified attention diagnostic, fixed training-only
  calibration and shared magnitude scales; never manufacture focus through
  per-frame scaling, selective heads or recoloring.
- Recompute perturbed inputs through the complete path. Match unperturbed replay
  first, including cache dtype and batch execution. Report random-removal and
  multiple-fill controls separately from localization evidence.
- Keep reports explicit about measured results, mathematical implications,
  qualitative failures and unresolved claims. Save exact code/settings/checkpoint
  provenance for every new visualization. Keep README executable defaults current.
- Run focused tests and the full suite after model, loss, transform, cache or
  attribution edits. The latest [verification](reports/experiments/spill_context_locality_20261106/verification.json)
  records 257 passed, 4 skipped: one CUDA test unavailable on the login node and
  three generated-dataset tests. Real H100 probes ran separately. Passing these
  checks did not validate localization; rerun relevant checks after code changes.

The [tsne_curves catalog](reports/experiments/spill_relative_clock_80_20261106/tsne_curves/README.md)
contains named PNG/PDF/SVG plots and numeric histories for the frozen-backbone
absolute/elapsed comparison. Keep it distinct from full-backbone curves and
from inference-only locality measurements.

Use `.venv/bin/python` if available. The latest verified environment, when that
virtualenv was absent, was:

```bash
LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib \
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 \
/home/olemou/miniconda3/envs/torchgpu/bin/python -m pytest -q
```

For documentation-only edits, check local links, consistency and
`git diff --check`; do not rerun training. Preserve unrelated workspace edits.
