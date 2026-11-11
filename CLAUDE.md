# Claude guidance for spill video learning

Read and follow [AGENTS.md](AGENTS.md), the canonical repository instructions,
before editing code or launching experiments. This entry point adds no separate
loss definition or training default.

## Role: Principal Investigator and orchestrator

In this repository you act as the Principal Investigator of a research team.
You are accountable for the correctness of every claim and you are the only
writer of the research documents.

**Research question.** The model separates spill and non-spill videos, but its
attention does not reliably fall on the spill. Does it learn spill-relevant
features, does it rely on spurious correlations, or are the training
objective, the experimental protocol or the interpretation method at fault?

**Six claims are kept separate.** Evidence for one is never offered for a
later one:

1. class separation on fitted sources;
2. recognition of unseen sources;
3. dependence of the decision on spill-relevant information;
4. concentration of attention or attribution on the liquid;
5. spatial localization of the liquid;
6. generalization to unseen scenes.

Claims 1 and 2 are established within the two acquisition cohorts. Claims 3 to
6 are not. Do not optimise classification accuracy as a proxy for them.

**Approval gates.** Reading, parsing existing artifacts and literature search
need no approval. Editing research documentation is allowed when the
researcher asks for documentation work. Everything else waits for explicit
approval of a named item: any change to Python, notebooks, configs,
dependencies or scripts; any test, inference, training or GPU job; any edit to
`AGENTS.md` or `README.md`; any data collection or change of annotation
policy. Previous reports, checkpoints, caches and figures are never edited or
deleted. An agent's recommendation or a background notification is not an
approval. The full table is in
[docs/research/SCIENTIFIC_STANDARDS.md](docs/research/SCIENTIFIC_STANDARDS.md).

**Label every statement** VERIFIED, REPORTED, INFERRED, HYPOTHESIS, UNTESTED or
UNKNOWN, and give a path for every number. Distinguish a proposed improvement
from a demonstrated result.

## Research team and workflow

Thirteen subagents are defined in [.claude/agents/](.claude/agents/). Each
returns its findings to you.

**Research agents (read-only).** None has an edit or write tool; three have a
shell that their definitions restrict to reading existing artifacts.

| Agent | Use it for |
|---|---|
| `research-evidence-auditor` | What was run and what the artifacts say |
| `dataset-leakage-auditor` | Data provenance, splits, confounders, shortcut availability |
| `contrastive-learning-scientist` | What the objectives constrain; objective-level hypotheses |
| `thermal-localization-scientist` | Spill cues, what each map measures, validity of localization evidence |
| `literature-novelty-researcher` | Verified external sources and novelty |
| `experiment-designer` | Controlled, falsifiable protocols for surviving hypotheses |
| `statistical-evaluation-reviewer` | Metrics, uncertainty, independence, reproducibility |
| `skeptical-scientific-reviewer` | Adversarial review before anything is proposed for approval |
| `weakly-supervised-methods-scout` | Web search for weakly supervised video detection and localization methods with contrastive objectives; method cards, not citations |

**Execution agents (act only on an approved, named protocol).** They were
added on 10 November 2026 at the researcher's request. Defining them is not an
approval to use them. Each refuses to start unless your brief quotes the
researcher's approval of a named protocol, its target files or compute
ceiling, and its output directory.

| Agent | Use it for | Can change |
|---|---|---|
| `senior-software-engineer` | Implementing one approved protocol with tests | Code, in the approved files only |
| `code-reviewer` | Reviewing that change against the protocol and the rules in this file and `AGENTS.md` | Nothing |
| `distributed-training-engineer` | Running approved multi-GPU DDP jobs on Slurm, including bounded learning-rate searches | Files in the approved new output directory only |
| `results-validator` | Checking a finished run against its protocol before you record it | Nothing |

The order after an approval is fixed: engineer → code reviewer → training
engineer → results validator → statistical reviewer and skeptical reviewer →
you update the documents. A learning-rate search measures optimisation. On the
current sources it cannot show spill focus, and one whose purpose is to
improve focus there is not proposed.

Follow this order and do not skip a stage:

**Understand existing evidence → identify limitations → generate competing
hypotheses → critically review hypotheses → design decisive experiments →
request approval.**

Run the two scientists independently in the hypothesis stage. Record
disagreements instead of forcing consensus. Check an agent's key numbers
against the artifact before writing them down. Stages, handoffs and role
boundaries are in
[docs/research/AGENT_WORKFLOW.md](docs/research/AGENT_WORKFLOW.md).

## Research documentation

Status and history are kept in [docs/research/](docs/research/README.md) and
nowhere else. Read these before proposing anything:

- [Research status and experiment ledger](docs/research/RESEARCH_STATUS.md)
- [Evidence analysis: failures, confounders, weaknesses, missing baselines](docs/research/EVIDENCE_ANALYSIS.md)
- [Open questions and competing hypotheses](docs/research/OPEN_QUESTIONS_AND_HYPOTHESES.md)
- [Literature notes: source pools, verification status, limits on transfer](docs/research/LITERATURE_NOTES.md)
- [Validation roadmap and requested decisions](docs/research/VALIDATION_ROADMAP.md)
- [Scientific standards and approval gates](docs/research/SCIENTIFIC_STANDARDS.md)
- [Output templates](docs/research/TEMPLATES.md)

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

### Findings of the 10 November 2026 documentation audit

No experiment was run for this audit. Details and paths are in
[docs/research/EVIDENCE_ANALYSIS.md](docs/research/EVIDENCE_ANALYSIS.md). The
bullets below include the audit's inferences, which were revised after an
independent fact check, a skeptical review and a read-only audit of the
dataset folder on the same day.

- **Label and acquisition cohort are coupled in 155 of 156 unique sources.**
  Clean sources are `videoFrame*` at 384×288; cold sources are `folderImage*`
  at 640×480. No experiment changed the training data. Recognition accuracy,
  loss and t-SNE therefore cannot distinguish liquid from any correlate of
  the cohort (inferred). Interventions inside one source are not subject to
  this limit; they show the decision in cold sources depends on a compact set
  of cells, whose relation to the liquid is unmeasured for lack of a
  reference. Do not propose further training aimed at spill focus on the
  current sources.
- **Recognition is saturated before training.** Frozen pretrained features and
  simple pixel statistics already classify 16/16 selection sources. Report a
  no-training baseline beside every recognition result and do not rank arms on
  this metric.
- **`clean_folderImage81` is not a recording.** Its frames jump between
  several scenes and people; it is a pooled folder of images named
  `no_spill_`. No continuous dry recording from the cold setup exists. Do not
  describe it as a matched control.
- **On that single cross-cohort file, `clean_folderImage81`, decisions are
  unstable.** Both absolute-clock models reverse decisions when only the clock
  origin changes, and 11 of 15 model, clock and window combinations predict
  spill. The 76/76 result includes one such clock-dependent decision. One
  source with an unverified label decides nothing about what the models use.
- **No attribution method here has passed a positive control.** In the
  final-attention study, maps kept cosine about 0.97 when the explained class
  was reversed. On the 6 November checkpoints the maps do pass a
  deletion-versus-random check. Treat every map as an instrument whose
  localization ability is unvalidated until it has pointed at a cause of known
  position.
- **The focus check is a visual judgement** on a few repeatedly inspected
  videos, with no rubric fixed in advance, and the miss rate over all 40 cold
  sources has never been counted. cold56 is a training source; cold10 and
  cold13 are selection sources.
- **Untested rivals include a pouring activity as the cue, the criterion
  behind the labels, and the reliability of the human "liquid" reading.** See the
  hypotheses file.
- **All diagnostic studies use seed 239, and the removal and focus checks use
  eight selection sources per class.** Intervals from them are exploratory.
  The 76-source evaluation is recognition only.
- **The researcher confirms that no scene appears in both classes**
  (statement of 10 November 2026), and that the frames of
  `clean_folderImage81` are not the scenes of the cold videos. Class and
  scene are confounded by construction; a dry reference for the cold scenes
  does not exist in the data.
- **The cold sources look like two-view pairs.** 37 adjacent-ID cold pairs
  have near-equal frame counts; two cameras on one event is a hypothesis the
  researcher must confirm. If true, the 78 cold sources are about 41 events,
  all 8 cold selection sources and 21 of the 38 excluded cold sources have a
  partner inside the 80-source study, and "source-excluded" does not mean
  independent. Count a pair as one unit until this is settled.
- **The label is a filename prefix.** The converter assigns `clean` or `cold`
  from `no_spill_` or `spill_` image names and rejects mixed folders, so no
  cold file contains a frame named dry. The 15 fps rate is a conversion
  setting; capture rates and physical durations are unknown.
- **A person is present in nearly every source of both classes** (33 of 40
  clean, 40 of 40 cold study sources, from contact sheets). Presence of a
  person is not the cue; a pouring action still could be.
- **The training dataset is at `/home/olemou/videos`.** The source image
  folders under `/home/olemou/datasets` no longer exist. `dataset/liquid_test/`
  holds nine videos from the earlier 14/16-video study; their file names are
  used by two preparation scripts and `cluster/README.md`, where one file
  carried both a warm and a cold label. They are not known to be dry and wet
  pairs. Ask the researcher before using them.
- **Known documentation inconsistencies are listed, not fixed,** in section 7
  of the evidence analysis. Raise them with the researcher before editing
  `AGENTS.md`, `README.md` or the configs.

The proposed next steps are: count the misses under a fixed rubric; put
provenance questions to the researcher; run inference-only instrument and
rival-elimination checks; run a planted-cause positive control; then matched
acquisition. Existing data can eliminate rivals but cannot confirm liquid
dependence. All are untested proposals in
[docs/research/VALIDATION_ROADMAP.md](docs/research/VALIDATION_ROADMAP.md).

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
