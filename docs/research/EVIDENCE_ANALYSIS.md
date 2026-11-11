# Evidence-backed analysis of previous results and failures

Audit date: 2026-11-10. The experiment ledger is in
[RESEARCH_STATUS.md](RESEARCH_STATUS.md); ledger tags such as L09 refer to it.
Evidence key: **V** read from a primary artifact during this audit, **R** taken
from report prose. Statements marked **INFERRED** are this audit's reasoning
from those facts and have not been tested.

This file was corrected on the same day after an independent fact check and a
skeptical review. The fact check confirmed the numbers and found thirteen
errors of wording, tagging or scope, all applied here. It was extended later
that day with a read-only audit of the dataset folder (sections 4 and 5);
the details are in [RESEARCH_STATUS.md](RESEARCH_STATUS.md) section 3.1.

## 1. What is established

1. **The two source populations are separable, and were before training.**
   Frozen pretrained V-JEPA 2 mean features with training-only prototypes
   classify 16/16 selection sources (L06, L08). So does a logistic model on
   four raw intensity statistics (L06) and a single mean-brightness scalar
   (L16). The trained adapter starts at selection ViCo 0.85 and reaches 2e-7
   (V, L18), but its prototype accuracy cannot exceed a baseline that is
   already perfect.
2. **Recognition needs neither temporal order nor the learned temporal
   residual.** Reversing, shuffling, repeating one state, zeroing timestamps
   and removing the residual all keep 16/16 (L06).
3. **The trained model does need spatial structure.** On spatially uniform
   frames it falls to 10/16 (L06), 8/16 and 5/16 (L18). That structure is not
   shown to be liquid.
4. **Class information is spatially distributed in the final tokens.** Tokens
   over letterbox padding, whose input embeddings are identical across
   sources, classify 16/16 in both arms (L22). With 11/12 of the frame
   modified, block-23 tokens at the untouched positions change by a median 37%
   to 85% over four sources and twelve tiles (L22).
5. **Inside a cold source, the decision depends on a compact set of cells.**
   Removing the highest-map cells lowers the margin more than removing random
   cells, in both arms (L18). Several other readouts show the same for their
   own maps (L13, L15).
6. **The four global losses do not constrain where evidence comes from.** All
   four act on the spatial mean of the final tubelet. Any zero-sum
   rearrangement of patch outputs leaves them unchanged (R, L18 diagnosis).
   This is a mathematical property, not an experimental result.
7. **The class objective saturates.** At the final full-fine-tuning checkpoint
   the prediction term's feature-gradient norm is about 28,000 times the
   semantic term's (R, L18). In the full arm selection ViCo is 2e-6 by update
   10; in the frozen arm it is still 1.27 at update 10 (V, L18 loss history).
8. **Freezing the backbone is not a sufficient explanation for the L18 run.**
   Updating all 24 blocks changes parameters and lowers the losses without
   changing the focus outcome. This does not rule out longer or differently
   supervised fine-tuning.
9. **The engineering path is sound where it was checked.** Encoder parity with
   the reference implementation is exact in FP32; saved-feature and gradient
   replays are exact; the latest suite records 257 passed and 4 skipped (R).
   Passing these checks says nothing about localization.

## 2. What failed, and what each failure does and does not show

| Failure or result | Shows | Does not show |
|---|---|---|
| No method gives reliable focus or full liquid extent on cold56, cold10 and cold13 (L09–L22). Isolated blocks or definitions do peak near liquid: block 2 on cold10, blocks 2–5 on cold13, the block-23 positive gradient on cold56 (L19) | The measured sensitivity of the class margin is not concentrated on the liquid in those videos | That the model never uses liquid information. The L19 diagnosis says a blanket statement of that kind would be incorrect |
| Maps barely change when the explained class is reversed or the projector is reset (cosine about 0.97, L09, L10); target reversal leaves maps highly similar in L13 | Those positive-gradient maps are dominated by something other than the trained class decision | That every attribution method here is uninformative |
| High-attribution removal is no better than random (L08, L09), or flips sign with the fill (L17 blur) | Those maps do not rank decision-relevant cells reliably | Anything about liquid identity |
| High-map removal does beat random on cold selection sources in both L18 arms at 10% and 20% (frozen 0.041 and 0.153; full 0.183 and 0.982 mean drop, mean fill; intervals exclude zero; clean sources near zero). Also for block-5 novelty in L13 and for the amplitude arm in L15 | With the source, and so the cohort, held fixed, the decision is sensitive to a compact set of cells that these maps identify. A cue spread evenly over the frame would not give this | That those cells are liquid; in the videos viewed they are elsewhere. Eight sources, exploratory intervals |
| Automatic targets abstain or select non-liquid (L14, L16, L17) | Darkness, novelty and local contrast do not reliably identify liquid in this data; L17 candidates partly cover the wall liquid | That no automatic cue could |
| `clean_folderImage81`: the clock-invariant model predicts spill; absolute-clock models flip with the clock origin (L20, L21, L22; table in section 4) | On the one available cross-cohort file, decisions are unstable across models and clock policies, and mostly disagree with the manifest label | What the models use; a general rate. The file is a pooled set of frames from several scenes, so its three windows show different content and none is a normal recording |

## 3. What is inconclusive

- **Whether the representation encodes the liquid at all.** A few blocks peak
  near liquid in the three known cases (L19). There is no reference to score
  this against, and the examples were chosen and re-inspected many times.
- **Which cue the model uses.** Brightness is not supported as the only cue:
  mean-matched and gain-shifted inputs keep 16/16 (L18). One control that
  resized cold sources to 384×288 did not change decisions for the L09 models,
  so native size alone is not established as the shortcut. Scene content,
  camera texture, viewpoint, layout and the activity of the person remain
  untested as separate causes. The mere presence of a person does not
  separate the classes (section 4).
- **How often the maps miss.** The observation rests on three named videos.
- **Whether temporal prediction reflects liquid dynamics** (section 2 of the
  status file).
- **Whether the selected learning rate is better.** Seed ranges for LR 5e-4
  and 1e-3 overlap (L02, L04).

## 4. Confounder map

Every row is a variable that predicts the label without reference to liquid.
Counts are over the 156 byte-unique sources unless stated.

| Confounder | Evidence | Tested as a cause of the model's decision? |
|---|---|---|
| Native resolution: 77 clean at 384×288, 78 cold at 640×480, one clean at 640×480 | V, manifest aggregation | Partly. One resize control did not change decisions for the L09 models. Not repeated for the 6 November checkpoints |
| Source naming and export batch: clean `videoFrame*`, cold `folderImage*` | V, manifest aggregation (new in this audit) | No. It indicates two acquisition or export pipelines aligned with class |
| Mean brightness: non-overlapping ranges | R, L16 | Yes. Not supported as the only cue (L18 controls) |
| Coarse layout and rank-normalised layout | R, L16 | Availability only |
| Recording duration: clean median 14.7 s over 94 records, 17.9 s over the 77 unique `videoFrame` sources; cold median about 38 s | V, manifest aggregation and fact check | Partly. A clock-only probe reaches 62.5% (L18). The absolute-clock model keeps 16/16 under zero, common and random origins and a +30 s shift, and loses decisions at −30 s (12/16) and +120 s (8/16) (L18, L21) |
| Downscale factor into the 256 grid: 0.667 against 0.4 | INFERRED from code and resolution | No, apart from the L09 resize control |
| Presence of a person | Not a discriminator: a person appears in 33 of 40 clean and 40 of 40 cold study sources (R, dataset audit, contact sheets) | Not needed as stated |
| What the person does, and where | Occlusion effects sit around the person (L09); a "cold pouring room" recurs (L06). Clean subjects also squat, crawl and lie on the floor (R, dataset audit), so posture alone is unlikely to separate the classes; a pouring action might | No |
| Viewpoint and rendering | All 40 cold study sources are steep high-angle views of a sparse floor with dark rendering, from two or three camera positions; clean sources are oblique indoor views or one static outdoor view, with brighter, hazier rendering (R, dataset audit) | No. This is the visible content of "cohort" |
| Scene, room, camera, session | No identifiers. Recurring environments across the split (R, L06). Seven clean study sources share one outdoor view: four in training, three in selection (R; split V) | No |
| Two views of one event | About 37 adjacent-ID cold pairs with near-equal frame counts (V); interpretation as simultaneous cameras is a HYPOTHESIS. Pairs cross every split boundary (section 5, item 4) | No. Affects independence, not the label |
| File-level proxies | In-sample AUROC over 156 sources: bitrate 0.946, file size 0.902, duration 0.671 (V). Codec and encoder settings identical | The model sees decoded frames, not the container; bitrate reflects noise and texture, which the frames do carry |
| Label origin | The label is the filename prefix of the source images, `no_spill_` or `spill_`, and a folder must be all one or the other (V, `conversion.py`). So no cold file contains a frame that was named dry | Not a confounder; it removes any within-file dry contrast by construction |
| Label noise | `clean_folderImage81` is a pooled folder of images named `no_spill_`; whether each frame is dry is unverified | No |
| Floor traces | Bright traces around a barefoot person in some clean recordings (R, dataset audit). Polarity coupling with class UNKNOWN | No |
| Byte-identical duplicates | 17 clean entries removed (V) | Handled by hashing; re-encoded duplicates and second views of one event are not detected by hashing |

**The cross-cohort observation in detail.** `clean_folderImage81` is the only
clean source with the cold cohort's resolution and naming. It is also the
longest file in the manifest, 124 s against about 101 s for the next. **It is
not a continuous recording.** The rendered inputs of the three windows (V,
viewed in the same directory as the results) show sampled frames about seven
frames apart jumping between at least three viewpoints and different people
in the beginning window, a different scene in the centre window, and several
unrelated scenes in the end window, one frame nearly featureless. The file is
a pooled set of images named `no_spill_`. The beginning window resembles the
cold cohort's steep high-angle views; the centre and end windows are blurrier
side views that do not.
Spill-minus-clean margins for three predefined windows, positive meaning
"spill" (V,
`spill_attention_competition_80_20261106/acquisition_counterexample/results.json`;
elapsed row R, `spill_relative_clock_80_20261106/report.md`):

| Model | Clock | Beginning | Centre | End |
|---|---|---:|---:|---:|
| Frozen, absolute clock | Actual file time | +1.38 | −1.19 | −1.53 |
| Frozen, absolute clock | Origin set to zero | +1.41 | +1.19 | +0.69 |
| Full, absolute clock | Actual file time | +1.48 | +1.45 | +1.75 |
| Full, absolute clock | Origin set to zero | +1.48 | −0.80 | −1.29 |
| Frozen, elapsed clock | Invariant to origin | +1.83 | +1.68 | +1.47 |

With identical pixels, both absolute-clock models reverse their centre and end
decisions when only the clock origin changes, and they reverse in opposite
directions. Across the 15 combinations of model, clock policy and window, 11
predict spill and 4 predict clean. No model gives the manifest label in all
three windows under any clock policy.

What follows from this, and what does not:

- On the 76 source-excluded videos, this is the only source any model gets
  wrong under either clock policy. With a zero origin the frozen model goes
  from 76/76 to 75/76 and the full model from 75/76 to 76/76 (V,
  `spill_context_locality_20261106/source_excluded/results.json`). The other
  75 are always correct.
- The centre and end windows sit at about 61 s and 121 s. The end lies beyond
  every training recording, so the absolute-clock flips there are
  extrapolation of the clock input.
- The three windows are different scenes, not one scene at three times. The
  table's columns therefore differ in content as well as in clock value, and
  each window is a sequence with scene cuts that no training recording has.
- The beginning window, the one that looks like the cold cohort, is called
  spill by all five rows. This is what H1 predicts for a dry scene seen from
  the cold setup, but it is one window, the frames' dryness is unverified, and
  the cuts are out of distribution. HYPOTHESIS, not a result.
- The file is a duration outlier as well as a cohort exception. It shows that
  recognition does not carry over reliably to the one cross-cohort case. It
  does not establish what the models use, and it is not a matched dry
  recording. The number of continuous dry recordings from the cold setup is
  zero.

**The files in `dataset/liquid_test/`** (V, container metadata, hashes, and
the fact check's reading of the scripts). Nine MP4 files of 45 to 50 frames at
10 fps, 160×120 to 800×800. The directory path is not referenced anywhere, but
the file names belong to the earlier 14/16-video study:
`scripts/prepare_unique_experiment.py`, `scripts/prepare_balanced_experiment.py`
and `cluster/README.md` use `clean_frames_20260227_181941`, `warm_scene_1_hot`,
`cold_scene3` and `cold_scene_4`, and `cluster/README.md` records that
`scene_4` is an exact duplicate of `scene_1_hot`, so the same bytes carried a
warm and a cold label. `sceneN` and `sceneN_warm` are therefore more likely
cold and warm spill recordings than dry and wet pairs, and they differ in
resolution. The labels of `scene1`, `scene2` and `mixte-scene` are UNKNOWN.
The frames were not viewed.

## 5. Methodological weaknesses

1. **Across sources, the data cannot separate "spill" from "cohort B".** The
   label and the resolution and naming cohort coincide in 155 of 156 sources,
   so recognition accuracy, loss and t-SNE cannot distinguish liquid from any
   correlate of the cohort. No experiment changed the training data, so every
   training run from L07 to L18 kept this coupling. Interventions inside one
   source are not subject to this limit, because the cohort is held fixed.
   They exist (L09, L13, L15, L18) and show that the decision in cold sources
   depends on a compact set of cells. What stopped them from testing C3 and
   C4 is the absence of any reference for where the liquid is, together with
   instruments that have not passed a positive control. INFERRED. (Wording
   revised after the skeptical review, which found the original claim, that no
   experiment could have shown liquid dependence "whatever the model did",
   overstated.)
2. **The recognition metric is saturated.** It is 16/16 for frozen-feature
   prototypes and for every trajectory arm. The only departures are the
   untrained projector (13/16, L08) and the L15 joint arms (14/16 and 15/16).
   It cannot rank methods or detect improvement.
3. **The selection set is not a test set.** The same 16 sources were used to
   choose learning rate and batch size, the best seed and several post-hoc
   arms, informed a block preference in L13, and were inspected visually many
   times. Loss weights were fixed from training-only gradient norms and were
   not chosen on it.
4. **Small n and unknown independence.** Lower limits of two-sided 95%
   Clopper–Pearson intervals, computed in this audit assuming independent
   sources: 16/16 → 0.794; 8/8 → 0.631; 76/76 → 0.953; 75/76 → 0.929. Sources
   that share a scene are not independent, so the true uncertainty is larger
   by an unknown amount. The dataset audit makes this concrete. If adjacent
   cold IDs are two views of one event (HYPOTHESIS; frame counts V), the 78
   cold sources are about 41 events, and the pairs cross every boundary: 3
   join training to selection, 16 join training to the excluded pool and 5
   join selection to the excluded pool. All 8 cold selection sources and 21
   of the 38 excluded cold sources then have a partner inside the 80-source
   study; only 17 excluded cold sources have none. On the clean side, three
   of the 8 selection sources share a static outdoor view with four training
   sources (R). The reports' "source-excluded" label is correct for files and
   byte hashes and is not evidence of independent events.
5. **One seed.** Every `seed` field in the settings, metadata and report JSON
   files is 239 (63 occurrences, V). Only the learning-rate searches used four
   seeds (239, 10239, 20239, 30239).
6. **The focus criterion is not a measurement.** It is a visual judgement by
   the report author on a handful of named videos: cold10 and cold13
   (selection), cold56 (training), clean121 and clean124. No rubric, blinding
   or second rater is documented. The negative verdicts are consistent across
   the ledger, which makes "unreliable" well supported, but "how unreliable"
   has no number, and the miss rate over all 40 cold sources is unknown.
7. **Bootstrap intervals are exploratory.** They resample eight cold sources
   and are unadjusted across dozens of variants, fills and area fractions.
8. **Several arms were added after seeing results** on the same selection set
   (L08 retention arms, L07 warm-up, L09 higher learning rate, L15 amplitude
   arm). The reports label them exploratory.
9. **Removal fills are out of distribution.** Mean, zero and blur fills give
   different signs in L17 and different magnitudes elsewhere. Uniform frames
   carry a class bias of their own: 0/8 cold and 8/8 clean in L09.
10. **Provenance gaps.** Historical code for the base checkpoint was not
    snapshotted; several runs record a dirty working tree; `reports/` is not
    under version control.
11. **Physical time is undocumented.** The 15 fps rate is a conversion
    argument, not a capture property, and the command used is not recorded;
    the capture rate is UNKNOWN (V, `conversion.py`). Every duration in
    seconds, the "4 fps" sampling and the "3 s" window are therefore in
    conversion time. The model sees a 12-frame centred window. Liquid is already visible in the earliest frames of the
    five cold recordings inspected over time (L09, L16), so there is no dry
    interval to compare against in those.

## 6. Missing baselines and controls

| Missing | Why it matters |
|---|---|
| A cross-cohort evaluation set | The only between-source test that separates liquid from cohort. Currently one file, and that file is a pooled image set, so the number of usable matched recordings is zero |
| An evaluation set grouped by event | The selection and excluded sets are disjoint from training by file, not by event; cold pairs and the shared clean outdoor view cross the boundaries |
| A positive control for the attribution pipeline | No method here has been shown to localise a cause whose location is known. Without it, "the map misses the liquid" cannot be separated from "the map cannot localise" |
| Randomisation and target-reversal checks on the 6 November checkpoints | Done for L09, L10 and L13 only |
| A count of misses over all 40 cold sources | The observation rests on three named videos; its frequency is unknown |
| Per-source removal effects | The L18 means over eight sources could be driven by one or two |
| A test of the person or activity as the cue | Never isolated |
| A sham-pour or dry-distractor recording | Nothing separates "liquid" from "someone pouring" or "a dark object" |
| A no-training baseline in every comparison table | Frozen-prototype and pixel-statistic baselines exist in L06 and L16 but are not carried into later reports |
| Multiple seeds and splits for any claimed difference | All diagnostic comparisons are single-seed |
| An independent liquid reference that needs no manual mask | Same-camera dry and spill recordings, registered, are named in L17, L18 and L22 as the missing resource and have not been obtained |
| A pre-registered focus criterion | Acceptance is decided after looking |

## 7. Documentation and protocol inconsistencies found

These are recorded, not corrected. Changing them needs approval because they
touch canonical documents or historical records.

1. The base checkpoint (L05) used batch 32, LR 5e-4 with warm-up and cosine
   decay, and no optimizer reset. [AGENTS.md](../../AGENTS.md) documents the
   default as full batch, LR 1e-3, with a reset before update 21. The L06
   diagnosis and the L18 proposal flag this; the canonical documents do not.
2. [AGENTS.md](../../AGENTS.md) and `README.md` require real masks and IoU for
   a localization claim, while the project excludes manual annotation.
   AGENTS.md names same-camera dry and spill recordings as potential
   annotation-free evidence, but neither file defines an accepted replacement
   metric.
3. `configs/data.yaml` declares thermal normalisation and 10 fps; the
   authoritative path uses checkpoint normalisation and the manifest is 15 fps.
   `configs/train.yaml` still targets the 14-video study.
4. `README.md` sections 3, 8 and 9 describe 64-frame sampling and the 14-video
   result, and link a report directory that does not exist.
5. `cluster/README.md` presents mask annotation and segmentation as the
   workflow. Annotation, MIL and decoder code remains in `scripts/` and `src/`
   without a legacy marker.
6. The known failures are listed together, but cold56 is a training source and
   cold10 and cold13 are selection sources.
7. The final-block diagnostic is called attention-times-gradient in
   [AGENTS.md](../../AGENTS.md); the L05 implementation is a positive attention
   gradient without the attention factor (R, L06).
8. Status bullets are repeated in `CLAUDE.md`, `AGENTS.md`, `vico.md` and
   `README.md`. This directory is now the single place for status and history.
9. Reports and `AGENTS.md` describe the 76 additional videos as
   source-excluded "by ID and byte hash". That is accurate, but 21 of the 38
   cold ones have an adjacent-ID partner with a near-equal frame count inside
   the 80-source study (V), which the reports do not mention.
10. Reports treat `clean_folderImage81` as a recording with a beginning,
    centre and end. Its rendered frames show a pooled image set.
11. The conversion log points to `/home/olemou/datasets`, which no longer
    exists; the dataset is at `/home/olemou/videos`.

## 8. Literature already cited in the reports

The reports cite the following sources. They were not re-opened in this audit,
so each is **REPORTED**. The Literature and Novelty Researcher must verify
them before any is used to justify an experiment.

| Topic | Cited in | Sources named |
|---|---|---|
| Context confounding in weak localization | L16 | CONTA (NeurIPS 2020); Evaluating WSOL Methods Right (CVPR 2020) |
| Anomaly detection under shift, style mixing | L16 | Anomaly Detection under Distribution Shift (ICCV 2023); MixStyle (ICLR 2021) |
| Token over-smoothing and dense features | L16, L18, L22 | Token Contrast (CVPR 2023); V-JEPA 2; V-JEPA 2.1; DINOv3; Vision Transformers Need More Than Registers |
| Attribution | L18, L20 | LeGrad; attention rollout; Integrated Gradients; AttnLRP; TokenCut |
| Thermal spill detection | L15, L16; L18, L22 | Warm Liquid Spill Detection and Tracking (CIVEMSA 2022); the local ViCo-LWIR PDF in `loss_doc/` |
| Region proposals | L16 | MSER (BMVC 2002) |

A second pool, the researcher's focused review of weakly supervised
contrastive video learning, is tracked in
[LITERATURE_NOTES.md](LITERATURE_NOTES.md) with its verification status and
its limits for this project.

The L18 proposal records that the ViCo-LWIR paper reports clean acquisition
under identical settings, challenging negatives and scene-level splits (R).
This dataset has none of the three.
