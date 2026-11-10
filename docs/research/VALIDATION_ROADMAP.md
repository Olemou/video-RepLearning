# Future experimental validation roadmap

Audit date: 2026-11-10. Revised the same day after a first skeptical review
and after the read-only dataset audit
([RESEARCH_STATUS.md](RESEARCH_STATUS.md) section 3.1).
**Nothing in this file has been run, implemented or approved.** Every item is
a proposal with status UNTESTED. Each needs a full protocol from the
Experiment Designer, review by the Statistical Evaluation Reviewer and the
Skeptical Scientific Reviewer, and explicit approval from the researcher
before any code is written or any job is launched. Gates are defined in
[SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md).

Hypotheses H1 to H7 and rivals R1 to R6 are defined in
[OPEN_QUESTIONS_AND_HYPOTHESES.md](OPEN_QUESTIONS_AND_HYPOTHESES.md). How the
researcher's literature review bears on this roadmap is in
[LITERATURE_NOTES.md](LITERATURE_NOTES.md).

## Principles

1. **No further training aimed at producing spill focus on the current
   sources.** Fourteen levers of that kind have been tried. Small diagnostic
   training whose answer is known by construction (stage C) is a different
   thing and is proposed.
2. **Measure the observation before explaining it.** The miss rate over all
   cold sources has never been counted.
3. **Validate the instruments before reading them.**
4. **Existing data can eliminate rivals; it cannot confirm liquid
   dependence.** The cold setup has no continuous dry recording: its one
   clean file is a pooled set of frames from several scenes. Every cold file
   contains only frames named `spill_*`, and the liquid is present from the
   earliest frames in the recordings inspected, so no liquid-absent contrast
   exists inside the cold cohort.
5. **Group by event before counting.** Until the two-view pairing of cold
   sources is confirmed or refuted, treat each adjacent-ID pair as one unit
   in any split, bootstrap or count, and use only the 17 excluded cold
   sources without a partner in the 80-source study as "unseen events".
6. **Prefer the cheapest test that can reject a hypothesis, and state the
   decision rule before looking.**
7. **Standing constraints:** no manual spatial annotation, MIL or
   segmentation decoder; originals stay intact; the known failure videos are
   not used to choose anything.

## Stage O — Fix the observation (existing artifacts, no model run)

| ID | Analysis | Answers | Limit |
|---|---|---|---|
| O1 | Score the existing 80-source galleries for all 40 cold sources against a rubric written before viewing | Q0, R4: how often do the maps miss? | The viewer is the reference; this gives a denominator, not a localization metric |
| O2 | Tabulate the L18 removal effect per source from the saved results, instead of a mean over eight | Whether the decision-relevant cells are compact in most cold sources or in one or two | Says nothing about liquid |
| O3 | Compare native peak positions across cold sources from the saved peak files | H1b: fixed grid positions point to a layout cue; source-specific positions to an object-like cue | Not decisive for liquid |
| O4 | View the existing contact sheets of the 37 presumed cold pairs side by side, under a written criterion, and record for each whether the two files show the same event from two positions | Q13: the number of independent cold events, and which evaluation sources are clean of training events | A visual judgement; the researcher's answer to A11 supersedes it |
| O5 | For the pairs confirmed by O4, compare the two views' saved peak positions (from the same peak files as O3) | Whether a map follows the same physical object in both views or a fixed image position | Exploratory; only 8 pairs have both members among the 80 sources with saved peaks. Says where the peak is, not whether it is liquid |

These read saved artifacts only. They fall under gate G0, but O1 is a new
qualitative judgement, so the rubric is shown to the researcher first.

## Stage A — Provenance (questions for the researcher)

| ID | Question | Decides |
|---|---|---|
| A1 | How were the `videoFrame*` (384×288) and `folderImage*` (640×480) recordings captured: cameras, rooms, dates, export tools, and capture frame rate? | Whether H1 can be split by camera, scene and export. The filesystem gives no capture date and the 15 fps is a conversion setting |
| A2 | Who named the source images `spill_` or `no_spill_`, by what criterion, and is liquid present throughout each cold recording? | R1; how to read every result. Known: the label is that filename prefix |
| A3 | What is `clean_folderImage81`: is it a pooled folder of dry frames from several scenes, and is every frame dry? Do continuous dry recordings from the 640×480 setup exist, or spill recordings from the 384×288 setup? | Size of any cross-cohort set. Known: none exists in `/home/olemou/videos` |
| A4 | Do spill recordings always contain a pouring action, and do clean recordings ever contain one without liquid? | H1c. Known from contact sheets: a person appears in 33 of 40 clean and 40 of 40 cold study sources, so presence alone is not the cue |
| A5 | Were dry frames removed from the cold folders before conversion? Do the original image folders still exist anywhere? The five recordings inspected show liquid from the earliest frames (L09, L16) | Whether any within-recording dry contrast can be recovered. `/home/olemou/datasets` is gone from this machine |
| A6 | Which recordings share a room or session? | Scene-grouped splits and honest sample sizes |
| A11 | Were the cold recordings made with two cameras at once, so that adjacent IDs (2/3, 4/5, …) are two views of one event? | Q13; the independence of the selection and excluded sets |
| A7 | The nine files in `dataset/liquid_test/` come from the earlier 14/16-video study. What labels do `scene1`, `scene2` and `mixte-scene` carry, and is any pair the same scene dry and with a spill? Scripts treat `scene3` as cold and `scene_1_hot` as warm, and `scene_4` is a byte-identical copy of `scene_1_hot` labelled cold | Whether any matched pair exists; currently none is known |
| A8 | Was automatic gain control or 8-bit export used? Is raw radiometric data available? | H6; whether dry and wet frames can be differenced. The source images are no longer on this machine, so their bit depth cannot be read |
| A9 | How was "visible liquid" established for the videos used in the focus check? | R3 |
| A10 | Can new recordings be made? | Stage D |

## Stage B — Inference-only diagnostics on the three fixed checkpoints

No optimizer update. Each needs a new diagnostic script, so each needs
approval. Outputs go to a new report directory. Where possible these run on
excluded sources, not on the 16 selection sources, which include cold10 and
cold13. Among the 38 excluded cold sources, only 17 have no presumed partner
in the 80-source study; results are reported for that subset separately, and
paired sources count as one unit.

### B1 — Attribution specificity check (tests H3)

- **Design:** recompute each method's maps after (a) randomising the
  trajectory adapter and prototypes, (b) replacing the explained scalar by a
  random direction in embedding space, (c) swapping the explained class.
- **Reference levels:** replays are exact, so the ceiling is 1. The floor is
  the similarity between maps of different sources under the trained model.
- **Decision rule:** a method is rejected as a readout of the trained
  decision if, after (a) or (b), its maps remain closer to the original than
  the cross-source floor.
- **What it cannot show:** localization. A method that passes is
  decision-specific, nothing more.
- **Prior:** (a) and (c) were run for the final-attention models only (L09).

### B2 — Tile transplant (tests H1a against local cues)

- **Design:** take a tile from source X and place it in the surround of a
  source Y of the opposite class, for all 12 predefined tiles of the locality
  study (L22). Record the margin, not the accuracy.
- **Why transplant and not fill:** uniform fills carry a class bias of their
  own. Uniform frames score 0/8 cold and 8/8 clean in one study (L09) and
  5/16 with AUROC 0.28 in another (L18), so a mostly filled frame returns the
  fill's verdict.
- **Baselines:** the unmodified pair; a tile from another source of the same
  class as Y; a keep-k curve over the number of transplanted tiles.
- **Decision rule that needs no liquid location:** if each of the 12 disjoint
  tiles, transplanted alone, moves the margin toward X's class by a similar
  amount, no single region is necessary and the cue is global (H1a). If a few
  tiles carry the effect, the cue is local, and O3 says whether it is fixed in
  position.
- **Constraint:** tiles are systematic geometry. Nobody states where the
  liquid is.

### B3 — Cross-cohort evaluation (placeholder)

Already run on one file (L20, L21), which turned out to be a pooled image set
and not a recording. Do not repeat it on that file. Run only if stage A finds
continuous cross-cohort recordings, on all three checkpoints under clock
policies declared in advance.

**Candidate variant on existing data (withdrawn 2026-11-10).** The researcher
states that the frames of `clean_folderImage81` are not the same scenes as
the cold videos and that no scene appears in both classes, so the variant
below cannot give a matched dry reference. It is kept for the record. The
researcher confirmed on 2026-11-10 that `/home/olemou/videos` holds all the
videos. The only dry material from the cold setup is then the pooled frames
of `clean_folderImage81`. Its frames could be regrouped by scene, with a rule
fixed in advance, into sets of dry stills from cold-like viewpoints, and each
checkpoint asked whether it calls them spill. Limits: the researcher must
first confirm the frames are dry and where they came from; the frames are
stills or very short runs, not recordings, so a clip built from them differs
from training clips in more than the liquid; the number of distinct scenes is
unknown and may be three or fewer; and a "spill" answer would support a
setup cue (H1) without showing what the cue is.

### B4 — Actor test inside recordings (tests H1c)

- **Scope after the dataset audit:** a person is present in nearly all
  sources of both classes, so this test is about what the person does, not
  whether one is there.
- **Design:** slide the window across each full recording. Regress the margin
  on an automatic per-window motion-energy scalar. Cohort, scene and the
  static liquid are constant within a recording.
- **Companion:** encode a person-free static clip made by repeating a
  per-recording temporal-median image. This image is not treated as dry.
- **Reading:** margin that follows motion, or that collapses for the static
  clip, supports H1c. A flat margin eliminates it.
- **Unit:** windows are repeated measures of one source.

### B5 — Resolution equalisation replication (low priority)

Repeat the L09 control on the 6 November checkpoints, in both directions,
with a matched low-pass filter without resize as a control, reporting margin
shifts. L09 found a mean margin change of 0.0007, so neither outcome is
expected to change the next step.

## Stage C — Planted-cause positive control (tests H3 and H2)

Diagnostic training on sources from a single cohort. Needs approval for code,
feature extraction and a small budget.

- **Design:** insert a local pattern at a random recorded location into half
  of the clean sources of one cohort, and train on the resulting
  pseudo-labels. Assign pseudo-classes so that they do not align with
  recurring scenes.
- **Dose ladder:** from a high-contrast synthetic patch down to a faint one,
  and a natural patch transplanted from another recording.
- **Evaluation:** each held-out source is evaluated with and without the
  insertion, as a pair.
- **Reference instrument:** occlusion at the planted location. It confirms
  that the planted pattern, and not something else, drives the decision.
- **Outcome measure:** pointing accuracy of each token-level method against
  the known location, compared with a random-location baseline, at each step
  of the ladder.
- **Controls:** a global pseudo-cue (uniform offset) and a random-label arm.
- **Two arms, two questions.** The adapter-only arm tests the instruments and
  the effect of pooling. Only a full-backbone arm can say anything about the
  objective, and it is proposed only if the adapter-only arm is informative.
- **Reading the result:**
  - Occlusion confirms the cause and token maps point at it: the instruments
    can localise a local cause of that strength. The real-data misses then
    point away from the instruments.
  - Occlusion confirms the cause and token maps miss it: H3 or H2 binds.
    Replace the instruments before anything else.
  - The pseudo-label is not learned: the cue is diluted by pooling over 192
    content cells. This says nothing about liquid.
- **Limit:** a planted pattern is not liquid (R6). The ladder shows where the
  instruments stop working; it does not validate them for real spills.

## Stage D — Matched acquisition (researcher decision)

The step that would make dependence on liquid and localization measurable
without manual masks. A data-collection request, not a code change.

- **Crossed design:** dry and spill recordings from the same camera, exposure
  and room, for each camera in use.
- **Controls the skeptical review requires:** a sham pour (same person and
  motion, no liquid); liquid at ambient temperature; randomised recording
  order to break thermal drift; gain locked or logged.
- **Distractors:** dark or cold non-liquid objects, people walking through.
- **Reference without manual masks:** fixed-camera before-and-after pairs,
  differenced after registration. The difference also contains people, moved
  objects and warming liquid, so it must be validated before use.
- **Recorded at capture:** scene, session, camera, settings, onset time.
- **Splits:** by scene. Sample size set from the number of independent
  scenes.
- **First analysis on the new data:** the no-training baselines. Recognition
  may no longer saturate, or may fail.

## Stage E — Model experiments (only after stage D)

Not proposed for approval now.

| Candidate | Precondition |
|---|---|
| Re-run the unchanged four-loss pipeline on crossed data | Stage D data |
| Cohort-balanced or group-aware supervised contrastive sampling | Stage D data with group labels |
| Patch-preserving local head scored against the difference reference | Stage D reference validated |
| Matched arms with and without a persistent semantic gradient (H5) | Stage D data |
| Instance selection inside positive videos, or a debiased contrastive term, as in the researcher's literature review (H7) | Stage D data; the source papers verified; and, for any MIL-based form, the researcher lifting the no-MIL rule |

Instance-selection and debiased-contrastive methods are placed here and not
earlier because, on the current sources, the non-liquid parts of a cold video
already differ from clean videos, so these methods have nothing
liquid-specific to select (see [LITERATURE_NOTES.md](LITERATURE_NOTES.md)
section 4). The recognition metrics they are usually judged on are also
saturated here.

## Requested decisions

1. Answers to A1 to A11. A11 (two cameras) and A3 (what `folderImage81` is)
   change the most.
2. Go-ahead for stage O, which reads saved artifacts only. The O1 rubric and
   the O4 pairing criterion would be shown to you before anything is scored.
3. Approval, or not, to have the team draft full protocols for B1, B2, B4
   and C. Drafting is read-only.
4. Whether new matched recordings (stage D) are feasible.
5. Whether the no-manual-annotation rule also covers a small evaluation-only
   reference that is never used for training. Your literature review
   recommends commissioning such a set. The standing instruction is read as
   excluding it, so the roadmap assumes none until you say otherwise.
   - **Principal Investigator's recommendation: allow it, for evaluation
     only.** Every localization claim in the literature that could be read is
     scored against annotated test data
     ([LITERATURE_NOTES.md](LITERATURE_NOTES.md) section 7.2); without any
     reference, claims C4 and C5 stay a matter of looking. A small set of
     boxes on cold sources, drawn by someone who has not seen the maps, never
     used to train, tune or select, would turn stage O1 into a pointing
     score with a random baseline.
   - **What it would not do:** show that the decision depends on the liquid
     (C3). The cohort coupling is untouched, and the boxes rest on the human
     reading of "liquid" (R3). Matched recordings (stage D) remain the only
     annotation-free route and the only route to C3.
   - The rule is unchanged until you say otherwise.
6. Whether the no-MIL rule stands. The literature check found no reviewed
   method verified to be free of MIL-style pooling.
   - **Principal Investigator's recommendation: keep the rule for now.** On
     the current sources, instance selection has nothing liquid-specific to
     select: padding tokens of a cold video already carry the class (L22), a
     constant field beats the real one on automatic patch targets (L14), the
     one MIL probe run here followed token norm (L06), and an external study
     reports that adding patch-level localization barely reduced source bias
     (REPORTED, literature notes section 7.3). Lifting the rule now would
     permit experiments that cannot decide anything. Revisit it at stage E,
     on matched data.
   - **One clarification needed:** whether pooling the top-k instance scores
     against the video label counts as MIL under your rule. The roadmap
     assumes it does.
7. Literature verification of the seven sources in the review was run on
   2026-11-10. Still open: full text of the four paywalled papers, if you
   have access, and verification of the sources cited inside the reports.
8. The execution agents (`senior-software-engineer`, `code-reviewer`,
   `distributed-training-engineer`, `results-validator`) are defined and
   idle. Each needs your approval of a named protocol, with target files or a
   compute ceiling, before it does anything. No protocol has been drafted
   yet; decision 3 comes first.

## Explicitly not recommended

- Another training run on the current sources to improve loss, t-SNE or map
  appearance.
- A learning-rate or batch search on the current sources whose purpose is
  better attention or localization. Two searches were already run (L02, L04);
  recognition is saturated before training, and no optimiser setting changes
  what the data can show. A search is in scope only inside an approved
  protocol that needs stable optimisation, such as stage C or stage E.
- Choosing a layer, head, threshold or colour scale from cold56, cold10 or
  cold13.
- Reusing the earlier automatic thermal candidates as targets.
- Reporting any localization number against automatic candidates.
- Treating the 16 selection sources or the 76 source-excluded sources as a
  test of scene generalization, or as independent of the training events.
- Reading `clean_folderImage81` as a matched dry recording.
- Region tests that fill most of the frame with a constant.
