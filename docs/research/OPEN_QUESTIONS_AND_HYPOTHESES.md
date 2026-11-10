# Unresolved scientific questions and competing hypotheses

Audit date: 2026-11-10. Revised the same day after a first skeptical review
(section 6) and again after the read-only dataset audit
([RESEARCH_STATUS.md](RESEARCH_STATUS.md) section 3.1). **Every hypothesis below is UNTESTED as a causal claim.** The
evidence cited for and against each one comes from experiments designed for
other purposes; see [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md). Proposed
tests are in [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md). Ledger tags such
as L18 refer to [RESEARCH_STATUS.md](RESEARCH_STATUS.md).

The observation to explain: the model separates spill from non-spill sources,
but its attention and attribution maps do not reliably fall on the liquid.

**The observation itself is under-measured.** It rests on a few named videos
(cold56, cold10, cold13) read by one unblinded viewer. How many of the 40 cold
sources show a miss is UNKNOWN. Fixing that denominator comes before testing
any explanation (roadmap stage O).

## 1. Unresolved questions

| ID | Question | Why it is open |
|---|---|---|
| Q0 | In how many cold sources do the maps miss the liquid, under a rubric fixed in advance? | Never counted |
| Q1 | Does the class decision depend on the liquid at all? | No reference for liquid location; label and acquisition cohort coincide in 155 of 156 sources |
| Q2 | Which correlate of the label carries the decision: global acquisition signature, scene layout, a person or activity, or the liquid? | Brightness is not supported as the only cue, and one resize control changed nothing; no other cue was isolated |
| Q3 | Can the attribution methods used here localise a cause whose location is known? | No positive control was ever run |
| Q4 | Do the features encode liquid distinctly from other dark, cold or textured regions? | No reference to score against |
| Q5 | Would the same objective produce liquid-focused evidence on data without the cohort coupling? | Such data has not been evaluated |
| Q6 | Is "focus on the spill" well-posed for a static cold spill in a 3 s window with no physical time base? | Not analysed |
| Q7 | What annotation-free measurement can stand in for a localization metric? | Matched dry and spill recordings are proposed in three reports and not available |
| Q8 | How many independent scenes and events does the dataset contain? | No scene or session identifiers. Frame counts suggest the 78 cold sources are about 41 events seen from two cameras; two or three cold camera positions and one shared clean outdoor view are reported from contact sheets |
| Q9 | What labels do `scene1`, `scene2` and `mixte-scene` in `dataset/liquid_test/` carry, and is any pair a same-scene dry and spill pair? | The files belong to the earlier 14/16-video study; scripts label some as warm, cold or clean, and one file carried two labels |
| Q10 | Do the non-liquid parts of a spill video resemble clean videos closely enough for instance-selection methods to work? | Existing evidence says no for the current sources (L22). The dataset audit agrees for the 80 sources viewed: the classes differ in viewpoint and rendering, and no matched subset exists. Untested on matched data |
| Q11 | What do the labels `clean` and `cold` denote, who assigned them, and is `clean_folderImage81` really dry? | Partly answered: the label is the filename prefix of the source images, `no_spill_` or `spill_`. Who named the images, by what criterion, and whether each pooled frame of `folderImage81` is dry, is not documented |
| Q12 | How was "visible liquid" established in the videos used for the focus check? | The user's description for cold56; the report author's reading elsewhere |
| Q13 | Are adjacent cold IDs two cameras recording one event? | Frame counts fit (37 pairs); never confirmed. Decides the number of independent cold samples and which evaluation sources are clean of training events |
| Q14 | Were dry frames removed from the cold folders before conversion, and do the original image folders still exist? | The converter rejects folders that mix `spill_` and `no_spill_` names; `/home/olemou/datasets` is gone from this machine |
| Q15 | Is the brightness polarity of floor marks coupled with class? | Bright floor traces reported in some clean recordings; not measured |

## 2. Hypothesis cards

### H1 — The decision follows a correlate of the label other than the liquid

- **Status:** UNTESTED as a causal claim. Shortcut availability is measured.
- **Three versions, which make different predictions:**
  - **H1a, global acquisition signature.** Camera, export or resampling
    texture present everywhere in the frame. Predicts that spatial shuffling
    leaves the decision intact and that removing high-map cells is no worse
    than removing random cells.
  - **H1b, scene layout.** Fixed rooms and viewpoints that differ by class.
    Predicts that map peaks sit at the same grid positions across sources.
  - **H1c, activity.** A pouring action, or a posture tied to it, appears in
    spill recordings. The presence of a person cannot be the cue by itself: a
    person appears in 33 of 40 clean and all 40 cold study sources, and clean
    subjects also squat, crawl and lie on the floor (dataset audit,
    REPORTED). Predicts that the margin follows the action inside a
    recording.
- **Evidence for some version:** one scalar classifies 16/16 (L16); frozen
  features classify 16/16 before training (L06, L08); padding-position tokens
  classify 16/16 (L22); rooms and subjects recur across the split, including a
  "cold pouring room" (L06); occlusion effects in cold10 and cold13 sit around
  the person (L09). The dataset audit adds a visible form for H1b: every cold
  study source is a steep high-angle view of a sparse floor from two or three
  camera positions, and no clean study source is (REPORTED).
- **Evidence bearing on H1a:** mixed. Against it: removing high-map cells
  lowers the margin more than removing random cells in cold sources, in both
  arms (L18), and patch shuffling lowers accuracy to 13/16 and 8/16 (L18). For
  it: under the same shuffle the ranking of sources stays perfect, AUROC 1.0
  in both arms (L18). Mean-matched inputs keep 16/16 (L18), so brightness is
  not supported as the only cue. One control that resized cold sources to
  384×288 did not change decisions for the L09 models.
- **Cross-cohort source:** decisions on `clean_folderImage81` vary with the
  model and the clock policy, 11 "spill" and 4 "clean" across 15 combinations.
  The file is a pooled set of frames from several scenes. Its beginning
  window, which looks like the cold setup, is called spill by all five model
  and clock rows; its other windows show different scenes. One window with
  scene cuts and unverified dryness is not decisive either way.
- **Rival:** the decision uses the liquid, and the maps or the human reading
  are at fault (H3, R3).
- **Falsified if:** on a crossed set with sham-pour recordings, all
  checkpoints under a clock policy declared in advance follow the label and
  not the cohort or the actor.

### H2 — Final-token attribution cannot recover a local cause after pooling

- **Status:** UNTESTED. The mathematical fact that all four losses act on a
  spatial mean is established (L18) and is not itself a hypothesis.
- **Mechanism:** tokens are contextual and the objective rewards any
  class-separating direction of their mean. A cause that is local at the input
  need not be local in the final tokens that the maps read.
- **Evidence for:** every readout that kept the patch grid gave perfect
  recognition without liquid-aligned maps (L06–L08); distant pixels change
  late tokens strongly (L22).
- **Evidence against:** removal results in L18 show that the maps do pick out
  cells that matter to the decision, so the evidence is compact in at least
  some sources, although it is not on the liquid in the videos viewed.
- **Rival:** H1. The maps are right and the cause is not the liquid.
- **Prediction:** for a planted cause that is local by construction, token
  maps fail to point at it while occlusion at the planted location removes
  the decision.
- **Falsified if:** token maps point at the planted cause in the frozen and
  full arms.
- **Limit of the available test:** an adapter-only run never shapes spatial
  features, so it tests the instrument and pooling, not the objective. A
  statement about the objective needs the full arm.

### H3 — The attribution methods are not valid readouts of the decision

- **Status:** partly tested. On the 6 November checkpoints the
  attention-gradient maps pass a deletion-versus-random check on cold
  selection sources in both arms (L18). Class specificity and localization of
  a known cause are untested on those checkpoints.
- **Mechanism:** attention-gradient maps on contextual tokens, input gradients
  and attention probabilities each measure something other than "where the
  class evidence is".
- **Evidence for:** in the final-attention models, maps kept cosine about 0.97
  when the explained class was reversed or the projector reset (L09, L10);
  three definitions peak in three places on cold56 (L19); integrated gradients
  fail completeness (L19); removal results depend on the fill (L17, L20).
- **Evidence against:** the L18 deletion result above; production input
  gradients with exact replay also miss the liquid (L19).
- **Rival:** the maps are faithful and the decision is simply elsewhere (H1),
  or the human reading of "liquid" is wrong (R3).
- **Rejection rule, per method:** a method is rejected as a readout if its
  maps are no more similar to the trained model's maps than maps of different
  sources are to each other after the adapter is randomised, or if it fails to
  point at a planted cause that occlusion confirms.

### H4 — The features do not represent the liquid distinctly

- **Status:** UNTESTED.
- **Mechanism:** the encoder was pretrained on natural RGB video, and the
  content area is 256×192 with 16-pixel patches, so a small spill covers few
  tokens.
- **Evidence for:** a second backbone gave the same outcome (L15).
- **Evidence against:** block 2 on cold10, blocks 2 to 5 on cold13 and the
  block-23 positive gradient on cold56 peak near the liquid (L19);
  qualitative and post hoc.
- **Rival:** liquid is encoded but lies outside the class direction being
  explained (R5), or nothing in training asks for it (H2).
- **Prediction:** at blocks declared in advance, a linear probe cannot
  separate liquid patches from distractor patches defined by the same
  procedure, on held-out scenes, above a shuffled-label control.
- **Falsified if:** it can. Needs matched recordings.

### H5 — Semantic-gradient saturation (corollary of H1)

- **Status:** the saturation is measured (L18). It is a consequence of the
  classes being separable before training, not an independent explanation.
- **What is measured:** selection ViCo is 2e-6 by update 10 in the full arm,
  while in the frozen arm it is still 1.27 at update 10; the prediction
  term's feature gradient is about 28,000 times the semantic term's at the
  end of the full run (L18).
- **Why it still matters:** during full fine-tuning the encoder was moved
  mostly by the prediction and retention terms, so L18 was not a strong test
  of class supervision shaping the backbone.
- **Test, only on crossed data:** matched arms with and without a persistent
  semantic gradient, scored against the matched-recording reference.

### H6 — Local appearance does not distinguish liquid from distractors

- **Status:** UNTESTED.
- **Mechanism:** a static cold spill is a dark region. Without motion or
  calibrated temperature it may be indistinguishable, locally, from a cold
  object, a shadow or clothing. The window is 3 s with conversion-assigned
  timing.
- **Evidence for:** automatic cues based on darkness, novelty or local
  contrast abstained or selected non-liquid structures, with only partial
  coverage of the wall liquid in L17 (L13, L14, L15, L17); the liquid is
  already present in the earliest frames of five inspected cold recordings
  (L09, L16).
- **Evidence against:** a human identifies the liquid in the frames. Whether
  that judgement uses local appearance or context is not recorded (Q12).
- **Prediction:** a family of local descriptors fixed in advance does not
  separate liquid from distractors recorded dry, on held-out scenes.
- **Falsified if:** it does. Needs matched recordings.

### H7 — Positive-bag contamination in patch-level objectives

- **Status:** measured for the patch-level variants; UNTESTED as a route to a
  fix. Raised by the researcher's literature review (its H1); see
  [LITERATURE_NOTES.md](LITERATURE_NOTES.md). Not yet reviewed by the skeptic.
- **Mechanism:** when every patch of a spill video inherits the spill label,
  background patches are trained as positives.
- **Scope:** not the authoritative pipeline, which contrasts one pooled
  embedding per video. It describes the all-patch, final-attention and dense
  variants (L08, L09, L15).
- **Evidence for:** all-patch ViCo raised within-image patch cosine from 0.367
  to 0.952 (L08); such labels admit a class-constant solution (L11); a
  constant field beats the real one on automatic patch targets (L14).
- **Evidence against a simple fix:** latent MIL probes, which do select
  instances, kept 16/16, and their scores correlated with token norm on the
  two cold videos examined (Spearman 0.54 to 0.73, L06).
- **Rival:** H1. If every patch of a cold-cohort video already differs from
  every patch of a clean-cohort video, instance selection has nothing
  liquid-specific to select.
- **Falsified if:** on data without the coupling, inherited labels and
  selected instances localise equally.
- **Constraint:** the literature check found no reviewed method verified to
  be free of MIL-style pooling: three are explicit MIL and three pool top-k
  scores against the video label
  ([LITERATURE_NOTES.md](LITERATURE_NOTES.md) section 7.2). Testing them needs
  the researcher to lift the no-MIL rule.

## 3. Rival explanations added by the skeptical review

| ID | Rival | What would support it |
|---|---|---|
| R1 | Label semantics or noise: `cold` may not mean "liquid visible", or `clean_folderImage81` may contain liquid | A provenance answer (Q11). The label is known to be a filename prefix; its criterion is not. The cross-cohort observation is already weak for a separate reason: the file is a pooled image set |
| R2 | Actor or activity (same as H1c) | Margin follows motion energy within a recording; person-free static clips lose margin |
| R3 | The human "liquid" judgement is unreliable: dark regions read as liquid may be shadow, reflection or a cold surface | A blinded second viewer disagrees; the same regions appear in matched dry recordings |
| R4 | Failure-case selection: the three videos may be a minority | Rubric scoring of all 40 cold sources finds most maps acceptable |
| R5 | The wrong scalar is explained: every map explains the spill-minus-clean margin, which may be the cohort axis, while liquid lies in within-class directions | Attributions of contrasts inside the cold cohort land on compact, stable regions |
| R6 | A planted cause is unrepresentative: synthetic patches are easier than a faint, amorphous cue | Pointing accuracy falls to chance as the planted cue is made weaker |

A seventh rival, that the 3 s window misses the spill event, is largely
contradicted: liquid is present from the earliest frames in the five cold
recordings inspected over time (L09, L16). It is UNKNOWN for the others.

## 4. Where the hypotheses disagree

Only the first three rows can be obtained with existing data.

| Test outcome | H1a | H1b | H1c | H2 | H3 |
|---|---|---|---|---|---|
| Peak positions repeat across cold sources at fixed grid cells | Silent | Favoured | Silent | Silent | Silent |
| Margin follows motion inside a recording; static person-free clip loses margin | Disfavoured | Disfavoured | Favoured | Silent | Silent |
| Maps after adapter randomisation are as similar to the originals as maps of different sources are | Silent | Silent | Silent | Silent | Favoured |
| Planted local cause: token maps point at it | Silent | Silent | Silent | Refuted for that cue and arm | Refuted for that cue and method |
| Planted local cause: occlusion confirms it, token maps miss it | Silent | Silent | Silent | Favoured | Favoured; not separable from H2 by this row |
| Crossed set with sham pours: decisions follow the label | Refuted | Refuted | Refuted | Silent | Silent |

H4, H5, H6 and H7 are decided only on matched recordings.

## 5. Priority and rationale

0. **Fix the observation (Q0, R4).** Count the misses under a rubric fixed in
   advance. Needs no compute.
1. **H3, with H2.** Every focus verdict relies on instruments that have not
   passed a positive control. Use the randomisation check and the planted
   cause.
2. **H1a, H1b and H1c through interventions on existing data.** These can
   eliminate rivals. They cannot confirm liquid dependence, because the cold
   cohort has no liquid-absent contrast.
3. **H1 as a whole, H4, H6 and H7** on matched recordings.
4. **H5** as a corollary, on matched recordings.

The hypotheses are not mutually exclusive. The researcher's literature review
proposes three: its H1 is H7 here, its H2 is H2, and its H3 is H1. The mapping
is in [LITERATURE_NOTES.md](LITERATURE_NOTES.md) section 3.

## 6. Review status

| Step | Status |
|---|---|
| Hypotheses drafted by the Principal Investigator from the audit | Done, 2026-11-10 |
| Researcher's literature review mapped onto the hypotheses (H7 added) | Done, 2026-11-10 |
| First skeptical review of H1 to H6, the roadmap and the central synthesis | Done, 2026-11-10, by a general-purpose agent following the reviewer definition. Every item received "revise"; revisions applied in this file and the roadmap |
| Skeptical review of H7 and of the revised cards | Not yet run |
| Independent fact check of cited numbers against artifacts | Done, 2026-11-10. Numbers confirmed; thirteen errors of wording, tagging or scope and a set of overstatements corrected in the status, evidence and hypotheses files. Not re-checked after correction |
| Read-only dataset audit of `/home/olemou/videos` | Done, 2026-11-10, by a general-purpose agent following the auditor definition. Manifest counts recomputed by the Principal Investigator; the `folderImage81` frames viewed by the Principal Investigator; other visual readings not re-checked. H1c narrowed, Q8, Q10 and Q11 updated, Q13 to Q15 added |
| Independent first-pass hypotheses from the two scientist agents | Not yet run |
| Literature and novelty check | Done for the researcher's seven sources, 2026-11-10: three read in full text, four from metadata, abstract or code only. Three arXiv abstracts re-opened by the Principal Investigator. The sources cited in the reports and the leads in section 7 are not yet verified. See [LITERATURE_NOTES.md](LITERATURE_NOTES.md) section 7 |
| Statistical review of each proposed test | Not yet run |

**What the first skeptical review changed.** H1 was split into three
versions. H2 was narrowed from "the objective" to what the available test can
show. H3's status was corrected from untested to partly tested. H5 was demoted
to a corollary. A circular clause was removed from H6. Six rivals were added.
The priority order now starts with the observation and the instruments instead
of H1.

**One recorded disagreement.** The reviewer judges H1a already disfavoured by
the shuffle and removal results. The Principal Investigator reads the evidence
as mixed, because source ranking survives shuffling with AUROC 1.0. The
measurement that would settle it is the per-source removal table and a
transplant test (roadmap O2 and B2).

## 7. Literature leads for the Literature and Novelty Researcher

These titles are recalled by the Principal Investigator and were **not**
source-checked in this audit. Each is an **unverified lead**. None may be
cited as support until the researcher has confirmed title, authors, venue and
the specific claim.

- Shortcut learning and dataset-source confounding: "Shortcut Learning in Deep
  Neural Networks"; "AI for radiographic COVID-19 detection selects shortcuts
  over signal"; "Variable generalization performance of a deep learning model
  to detect pneumonia in chest radiographs".
- Simplicity bias and gradient starvation: "The Pitfalls of Simplicity Bias in
  Neural Networks"; "Gradient Starvation: A Learning Proclivity in Neural
  Networks"; "Can contrastive learning avoid shortcut solutions?".
- Supervised contrastive learning: "Supervised Contrastive Learning".
- Validity of saliency and attention: "Sanity Checks for Saliency Maps";
  "Attention is not Explanation"; "A Benchmark for Interpretability Methods in
  Deep Neural Networks"; "Transformer Interpretability Beyond Attention
  Visualization".
- Token artefacts in vision transformers: "Vision Transformers Need
  Registers".

Sources already cited inside the reports are listed in
[EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) section 8. The researcher's own
review is tracked in [LITERATURE_NOTES.md](LITERATURE_NOTES.md).
