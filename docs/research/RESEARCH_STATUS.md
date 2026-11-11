# Research status and experimental history

Audit date: 2026-11-10. Scope: every directory under `reports/experiments/`,
the root and `docs/` Markdown, and the code paths named in
[AGENTS.md](../../AGENTS.md). This file records what was done and what was
measured. Interpretation is in [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md);
open questions are in
[OPEN_QUESTIONS_AND_HYPOTHESES.md](OPEN_QUESTIONS_AND_HYPOTHESES.md).

Evidence key used in the tables:

- **V** — the number was read from a primary artifact (JSON, CSV, manifest)
  during this audit.
- **R** — the number is taken from a report's prose and was not re-read from
  its primary artifact by the author of this file.

An independent fact check on 2026-11-10 re-traced the numbers in this file to
their reports or artifacts and its corrections are applied. A read-only audit
of the dataset folder followed on the same day; its findings are in section 3
and its key counts were recomputed from the manifest by the Principal
Investigator. No experiment, training, inference or GPU job was run for this
audit.

## 1. Status in one paragraph

The pipeline separates the `clean` and `cold` source labels on every split
evaluated: 16/16 on the reused selection set, and 76/76, 75/76 and 75/76 on 76
source-excluded videos. Frozen pretrained features already reach 16/16 on the
selection set before any training. No experiment has shown that the decision
depends on the liquid. Fourteen levers on the model, objective, spatial
readout, backbone trainability, clock input and attribution method (section 5)
have each failed the qualitative spill-focus check. The dataset couples the
class label with the acquisition cohort almost perfectly. On the one source
that breaks the coupling, decisions are unstable: the two absolute-clock
models change their answer when only the clock origin changes, and 11 of 15
model, clock and window combinations contradict the manifest label. That
source is not a continuous recording but a pooled set of frames from several
scenes, so no matched dry recording exists at all. The cold sources also look
like about 37 two-view pairs of the same events, which lowers the number of
independent cold samples and links the 76 "source-excluded" videos to the
training set (section 3.1). Removal
tests inside cold sources show that the decision depends on a compact set of
cells, but no independent reference for liquid location exists, so whether
those cells are liquid has never been measured, only inspected by eye on a few
videos.

## 2. Claim ladder

The six claims are separate. A higher row is not evidence for a lower row.

| # | Claim | Status on 2026-11-10 | Strongest evidence |
|---|---|---|---|
| C1 | Class separation on fitted sources | Established | 64/64 training prototype accuracy; training ViCo 9.3e-7 (frozen) and 5.0e-8 (full) at update 100 (V, `experiment_81_execution_20261106/figures/loss_history.csv`) |
| C2 | Recognition of unseen sources from the same cohorts | Established for unseen files within cohorts; independence of those files from the training events is now in doubt | 16/16 on the reused selection set (V); 76/76, 75/76, 75/76 on 76 source-excluded videos (V, `spill_context_locality_20261106/source_excluded/results.json`). Every error is the single cross-cohort source. If adjacent cold IDs are two views of one event (HYPOTHESIS, section 3.1), all 8 cold selection sources and 21 of the 38 excluded cold sources have a partner inside the 80-source study |
| C3 | The decision uses spill-relevant information | Not established; never testable for lack of a liquid reference | Removing high-map cells beats random removal in cold sources in both arms (R, `experiment_81_execution_20261106/report.md`), so the decision uses a compact region; in the videos viewed it is not the liquid. On the only clean source from the cold cohort, 11 of 15 model, clock and window combinations predict spill (V, `spill_attention_competition_80_20261106/acquisition_counterexample/results.json`; R, `spill_relative_clock_80_20261106/report.md`). One source, unverified label |
| C4 | Attention or attribution concentrates on the liquid | Failed qualitative acceptance in every study | See section 4. Isolated blocks or definitions peak near liquid; none does so reliably |
| C5 | Spatial localization of the liquid | Never measured | No masks, no independent reference; every report states IoU and pointing accuracy are unavailable |
| C6 | Generalization to unseen scenes | Untested | No scene or session identifiers in the manifest |

Temporal prediction is a seventh, separate claim. Prediction falls below the
persistence baseline (0.0244 frozen and 0.0142 full against 0.0391, V), but on
four sources a constant-image control produces prefix-to-prefix feature
variation 1.07 to 2.10 times the real-video variation (R,
`spill_context_locality_20261106/report.md` §6). Prediction of the
representation trajectory is established; prediction of physical liquid
dynamics is not.

## 3. Data used

| Fact | Value | Evidence |
|---|---|---|
| Dataset root | `/home/olemou/videos` with `clean/`, `cold/`, `manifest.jsonl`, `conversion_errors.jsonl`; outside the repository | V |
| Manifest | 173 records | V, `spill_context_locality_20261106/dataset/results.json` |
| Byte-unique sources | 156: 78 clean, 78 cold; 17 redundant clean entries in 13 duplicate groups; no cross-label duplicates | V |
| Main study subset | 80 sources: 40 clean, 40 cold; 64 training, 16 selection (8 per class); seed 239 | R, V |
| Additional evaluation | 76 source-excluded videos: 38 clean, 38 cold | V, `dataset/available_source_excluded_pool.json` |
| Cohort A | 94 clean records (77 unique sources) named `videoFrame*`, all 384×288, 15 fps; median duration 14.7 s over records | V, aggregated from `source_rows` |
| Cohort B | 78 cold records named `folderImage*` (three spelled `folderimage*`), all 640×480, 15 fps; median duration about 38 s | V |
| Cross-cohort source | One: `clean_folderImage81`, 640×480, 1,864 frames (124 s at the assigned rate), the longest file in the manifest. It is not a continuous recording (section 3.1) | V |
| Warm class | Mapped to label 1 in code; no warm source in the current manifest. Warm sources existed in the earlier 14/16-video study | R |
| Input window | 12 frames at about 4 fps, centred in the file: about 3 s of recordings lasting 4 to 124 s | R, `experiment_81_execution_20261106/deep_analysis.md` |
| Origin of the videos | Built from image folders by `src/vision_model/data/conversion.py` (V, code and `/home/olemou/videos/conversion_errors.jsonl`). The source folders, logged under `/home/olemou/datasets`, no longer exist on this machine (V), so the bit depth and palette of the original images are UNKNOWN | V, UNKNOWN |
| Frame rate | 15 fps in every record. The rate is a conversion argument, not a capture property: the converter writes whatever `--fps` it is given (script default 30), and the command used is not recorded. The capture rate, and so every duration in seconds, is UNKNOWN. `configs/data.yaml` states 10 fps | V, UNKNOWN |
| Origin of the labels | The converter labels a folder `clean` when every image name starts with `no_spill_` and `cold` when every name starts with `spill_`, and rejects mixed folders (V, `conversion.py` lines 144–180). Each cold file therefore contains only frames named `spill_*`. Who named the frames, and whether dry frames were removed first, is UNKNOWN | V, UNKNOWN |
| Scene, session, camera | Absent from the manifest keys; not found elsewhere in this audit | V |
| Spill masks or points | None verified; 480 unverified template records | R, `spill_fg_bg_implementation_20261021` |

### 3.1 Findings of the read-only dataset audit (10 November 2026)

Run by the Dataset and Leakage Auditor on `/home/olemou/videos`, the manifest
and already-rendered contact sheets. No video was decoded into a model. Counts
marked V were recomputed by the Principal Investigator from
`manifest.jsonl`, `spill_context_locality_20261106/dataset/results.json` and
the split recorded in `experiment_81_execution_20261106/figures/tsne_frozen_final.json`.
Visual statements come from the auditor's reading of existing contact sheets
of the 80 study sources and are REPORTED unless stated.

| Finding | Detail | Evidence |
|---|---|---|
| Folder content | `clean/` 95 MP4 files, `cold/` 78, a 173-line manifest and a 2-line error log. No README or sidecar metadata. All files were written in one conversion batch on 2026-10-21, so file dates carry no capture information | V; dates R |
| Source IDs | Cold: `folderImage2` to `79`. Clean: `videoFrame1`–`10` and `53`–`136` (11–52 absent), plus `folderImage81` | V |
| Meaning of the names | The researcher states (2026-11-10) that all sources are videos and that the `folderImage` and `videoFrame` names should be ignored. The names are used in these documents only as handles. The two cohorts are defined by what is in the files: native resolution (V) and the viewpoint and rendering seen in the frames (REPORTED) | Researcher statement |
| Cold sources look like two-view pairs | 29 of 77 adjacent-ID cold pairs differ by at most 2 frames, against 4 of 83 for clean outside `videoFrame1`–`10`. At a tolerance of 15 frames, 37 disjoint adjacent pairs cover 74 of the 78 cold sources; 42, 67, 76 and 77 are unpaired. The auditor's reading of the contact sheets is two cameras recording each event at once | Counts V; interpretation HYPOTHESIS |
| Where the pairs fall | Of the 37 pairs: 5 lie inside training; 3 straddle training and selection (4/5, 36/37, 38/39); 16 join a training source to an excluded source; 5 join a selection source to an excluded source; 8 lie inside the excluded pool. Every one of the 8 cold selection sources has its partner outside the selection set. The partners of cold10, cold13 and cold56 are cold11, cold12 and cold55, all in the excluded pool | V, given the pairing |
| Clean sub-cohorts | `videoFrame1`–`10` each have exactly 1,501 frames. The seven of them in the study (1, 5, 6, 8 training; 3, 4, 7 selection) show one static elevated outdoor view. The other 33 clean study sources are indoor oblique views with a person | Frame counts and split V; views REPORTED |
| Cold sub-cohorts | IDs 2–41 are encoded at 0.050–0.105 bits per pixel (40 sources), IDs 42–79 at 0.143–0.584 (38 sources); the ranges do not overlap. Encoder settings are constant, so this reflects image content or noise | V |
| What the classes look like | All 40 cold study sources: indoor, steep high-angle view of a sparse floor, dark rendering, a person in every one, two or three camera positions. Clean: brighter, hazier rendering, oblique views; a person in 33 of 40. Clean subjects also squat, crawl and lie on the floor | REPORTED |
| File-level proxies of the label | Over 156 unique sources, in-sample AUROC: width 0.994, bitrate 0.946, file size 0.902, duration 0.671. Codec, container and encoder settings are identical across classes | V |
| `clean_folderImage81` | The three rendered windows show consecutive sampled frames jumping between at least three viewpoints and different people within the first 41 frames, and unrelated scenes in the centre and end windows, one frame nearly featureless. It is a pooled folder of `no_spill_` images, not one scene over time. The beginning window resembles the cold cohort's high-angle views; the centre and end windows do not | V, viewed by the Principal Investigator in `spill_attention_competition_80_20261106/acquisition_counterexample/` |
| Matched subset | None. No cold source at 384×288, no `videoFrame` source labelled cold, and no continuous dry recording from the cold setup | V |
| Researcher's confirmation | The researcher states (2026-11-10) that the dataset contains no scene that appears in both classes, and that the frames of `clean_folderImage81` are not the same scenes as the cold videos. Class and scene are therefore confounded by construction, and `folderImage81` cannot serve as a dry reference for the cold scenes | Researcher statement |
| Possible floor-trace cue | Several indoor clean recordings show bright traces on the floor around a barefoot person (for example `videoFrame54`). Whether brightness polarity of floor marks is coupled with class is UNKNOWN | REPORTED |

## 4. Experiment ledger

Directory names are under `reports/experiments/`. "Focus" means the qualitative
check that map peaks follow the visible liquid in the inspected videos.

### Phase 0 — Establishing the trajectory pipeline (21 October 2026)

| Ledger | Directory | What was tested | Measured outcome | Verdict |
|---|---|---|---|---|
| L01 | `trajectory_smoke_h100_20261021` | Four videos, one update | Plumbing only | Software check |
| L02 | `trajectory_lr_search_real12_20261021` | 156 videos, 124/32 split, 19 trials, four seeds | Batch 32 and LR 5e-4 selected; prediction about 0.0391 above persistence 0.0380 (R) | Temporal claim not supported at this budget |
| L03 | `trajectory_20_sources_12f_100ep_20261021`, `…_lr5e4_20261021` | 20 videos, no held-out data | Training fit only | Not evidence of generalization |
| L04 | `trajectory_80_sources_12f_lrsearch_20261021` | 80 videos, 64/16 split, 16 trials | Batch 32 and LR 5e-4 selected on the 16 selection sources; prediction 0.0423 above persistence 0.0391 at 32 updates (R) | Selection set consumed for tuning |
| L05 | `trajectory_80_sources_real_ddp_100updates_20261021`, `…_all_losses_20261021` | Base checkpoint, 100 updates, seed 239 | 64/64 and 16/16; prediction 0.0269 below persistence 0.0391 (R) | C1 and C2 on a reused set |
| L06 | `…_all_losses_20261021/research_diagnosis` | Temporal ablations, simple baselines, patch readouts, latent MIL, rank-16 adapter | Every temporal ablation keeps 16/16; frozen teacher prototypes 16/16; a separate classifier on four raw intensity statistics 16/16; the trained model on spatially uniform frames 10/16; MIL scores correlate with token norm on two cold videos (Spearman 0.54 to 0.73) (R) | Recognition is achievable without the trained adapter or time order, and by a pixel-statistic classifier. The trained model itself needs spatial structure, which is not shown to be liquid |

### Phase 1 — Spatial readouts on frozen tokens (21 October 2026)

| Ledger | Directory | What was tested | Measured outcome | Verdict |
|---|---|---|---|---|
| L07 | `spatial_semantic_mirror_*` (four directories) | Class-query spatial attention with mirror consistency | Attention collapses to uniform: entropy 1.0, maximum probability 0.00521 against uniform 0.00521; clean and spill query maps have cosine 1.0 (R) | Focus failed |
| L08 | `all_patch_vico_*` (four directories) | ViCo applied to every content patch with the video label | Within-image patch cosine rises from 0.367 (raw) to 0.952; high-minus-random removal 2.0e-6 with interval including zero (R). Frozen raw features already score 16/16; the untrained projector scores 13/16 (V, `initial_metrics.json`) | Homogenization; focus failed |
| L09 | `final_attention_80_sources_20261021` | Training block-23 attention at LR 3e-6 and 1e-4 | Maps nearly unchanged by training (cosine 0.99998 and 0.99939), by reversing the explained class (about 0.97) and by resetting the projector (about 0.97); removal advantage about 0.005 with intervals including zero; strongest single-cell occlusion effects sit around the person (R). Resizing cold sources to 384×288 keeps 16/16 with mean margin change 0.0007 (V, `acquisition_controls.json`). Liquid is visible in the earliest frames of four inspected cold recordings (R) | Focus failed; maps not decision-specific |
| L10 | `test_80`, `test_80_fixed_20261021` | Galleries and requirement review | Map entropy 0.997; reversed-target cosine 0.97 (R) | Focus failed |
| L11 | `spill_attention_dual_loss_audit_20261021` | Gradient audit of patch ViCo plus retention | The two terms oppose each other: gradient cosine −0.9987 (R) | Objective conflict documented |
| L12 | `spill_fg_bg_implementation_20261021`, `spill80_run_preflight_20261022_020721` | Verified-point supervision path | Zero verified records; never trained; preflight exited without a GPU job (R) | Dormant; excluded by the no-annotation constraint |

### Phase 2 — Annotation-free spatial targets (21–22 October 2026)

| Ledger | Directory | What was tested | Measured outcome | Verdict |
|---|---|---|---|---|
| L13 | `spill_annotation_free_80_20261021` | Normal-reference novelty candidates as automatic targets | Joint and control attention parameters differ by relative L2 2.0e-5; no gain from the local loss; target reversal leaves maps highly similar. Block-5 raw novelty beats random removal on cold sources, interval [0.012, 0.078] (R) | Focus failed |
| L14 | `spill_focus_deep_audit_20261022` | Why the automatic branch still classifies scenes | Within the 32 training cold sources, 178 foreground cells against 1 background cell; a spatially constant field scores better on the candidate loss than the real field; letterbox-only pooling keeps 16/16 (R) | Automatic targets invalid as a selector |
| L15 | `spill_dense_fix_80_20261022` | Dense readout on frozen V-JEPA 2.1 at 384 with automatic targets | Joint arms drop to 14/16 or 15/16; the amplitude arm's own high-evidence cells beat random removal (0.63, interval [0.52, 0.74]); cold10 and cold13 still wrong (R) | Focus failed; recognition degraded |
| L16 | `spill_scene_confound_audit_20261022` | CPU shortcut-availability probes; no model changed | One scalar (mean grey level) gives 64/64 and 16/16; class ranges do not overlap (clean 108–133, cold 18–81); rank-normalised 8×6 layout gives 16/16; cold56 receives zero foreground targets (R) | Shortcuts available; target support missing |
| L17 | `spill_local_relative_80_20261022` | Separate bounded-receptive-field thermal detector with candidate-versus-ring ranking, 100 and 1,000 updates | Locality check passes (difference 0.0); removal result changes sign with fill (+0.039 mean, −0.103 blur); photometric stress not passed; candidates partly cover wall liquid; cold56, cold10, cold13 still wrong (R) | Locality verified; liquid identity failed |

### Phase 3 — Backbone, clock and attribution controls (6 November 2026)

| Ledger | Directory | What was tested | Measured outcome | Verdict |
|---|---|---|---|---|
| L18 | `experiment_81_full_jepa_trajectory_20261106`, `experiment_81_execution_20261106` | Matched frozen versus fully trainable encoder, four losses, 100 updates each | Both 16/16; prediction 0.0244 and 0.0142 against 0.0391 (V). Semantic feature-gradient norm 1.7e-9 against prediction 4.8e-5 at the final full checkpoint; late-token patch cosine rises 0.360 to 0.442. High-map removal beats random on cold selection sources in both arms. Patch shuffle gives 13/16 (frozen) and 8/16 (full) with AUROC 1.0; mean-only frames 8/16 and 5/16; zero, common and random clock origins keep 16/16 (R) | Freezing alone rejected as a sufficient explanation for this run; focus failed |
| L19 | `experiment_81_execution_20261106` (all-block and input-gradient audits) | All 24 blocks, production input gradients, integrated gradients | Three attribution definitions peak in three different places on cold56. Block 2 on cold10, blocks 2–5 on cold13 and the block-23 positive gradient on cold56 peak near liquid. Integrated-gradient completeness fails (0.579 against tolerance 0.145) (R) | No reliable layer; attribution definitions disagree |
| L20 | `spill_attention_competition_80_20261106` | Exact softmax-logit attention derivative | Replay exact; focus still fails; removal gains depend on fill (R) | Attribution change did not help |
| L21 | `spill_relative_clock_80_20261106` | Elapsed-clock adapter | Clock shifts on the absolute-clock model: +30 s keeps 16/16, −30 s gives 12/16, +120 s gives 8/16; the elapsed model stays 16/16. The elapsed model predicts `clean_folderImage81` as spill in all three windows (R) | Clock invariance improved; focus and counterexample not improved |
| L22 | `spill_context_locality_20261106` | Distant-context probes, padding-only readout, content-only pooling, constant-image prefixes, 76 source-excluded videos | With 11/12 of the frame modified, block-23 tokens change by a median 37–40% (gain) and 83–85% (blur) over four sources and twelve tiles; padding-only tokens classify 16/16 in both arms; content-only pooling still peaks away from the liquid (R). With a zero clock origin the frozen model goes from 76/76 to 75/76 and the full model from 75/76 to 76/76; the source that changes is always `clean_folderImage81` (V) | Class information is spatially distributed |

A separate dataset check, `balanced_78_clean_78_spill_20261021`, validates the
156-source manifest and records one clip shorter than 64 frames.

## 5. Levers already pulled for spill focus

Each row is a completed experiment whose focus check failed. Repeating one
needs a new reason.

| Lever | Ledger | Why it did not establish focus |
|---|---|---|
| Train longer | L17 (100 → 1,000 updates) | Loss falls; maps do not improve. Tested on the local detector only; the global pipeline was never run beyond 100 updates |
| Unfreeze the last attention block | L09 | Maps unchanged |
| Unfreeze all 24 blocks | L18 | Semantic gradient already saturated; focus unchanged |
| Keep the patch grid in the readout | L06, L07, L08 | Recognition stays perfect; maps homogeneous or uniform |
| Patch-level ViCo with inherited labels | L08 | Every patch of a cold video is trained as spill |
| Mirror consistency | L07 | Trivial uniform solution |
| Latent MIL and a small adapter | L06 | Scores follow token norm; excluded by the current constraints |
| Intermediate blocks | L06, L13, L19 | No common informative layer |
| Automatic novelty or darkness targets | L13, L14, L15, L17 | Targets missing or wrong; a constant field wins (L14) |
| Different pretrained backbone | L15 | V-JEPA 2.1 dense features do not change the outcome |
| Bounded local detector without V-JEPA | L17 | Finds dark components, not reliably liquid |
| Clock correction | L21 | Invariance improves; focus does not |
| Different attribution definition | L19, L20 | Definitions disagree; none follows the liquid reliably |
| Remove padding from pooling | L22 | Same or still off-liquid peaks |

## 6. Assets that exist

- Three fixed checkpoints from 6 November: frozen absolute, full absolute,
  frozen elapsed. Hashes are in
  `spill_context_locality_20261106/source_excluded/results.json`.
- The base checkpoint from 21 October, trained with batch 32 and LR 5e-4, which
  differs from the documented default protocol (see
  [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) section 7).
- Named loss curves and t-SNE figures:
  `spill_relative_clock_80_20261106/tsne_curves/README.md`.
- `dataset/liquid_test/`: nine short MP4 files from the earlier 14/16-video
  study. The directory path is not referenced, but the file names are used by
  `scripts/prepare_unique_experiment.py`, `scripts/prepare_balanced_experiment.py`
  and `cluster/README.md`. See
  [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) section 4.
- `reports/`, `dataset/` and `checkpoints/` are excluded by `.gitignore`. The
  experimental record exists only on this filesystem.
