# Scientific standards, reproducibility and approval gates

These rules bind the Principal Investigator and every research agent. Pipeline
and engineering rules (losses, preprocessing, distributed execution, test
commands) are in [AGENTS.md](../../AGENTS.md) and are not repeated here. Where
this file and AGENTS.md overlap, AGENTS.md governs the code and this file
governs how claims are made.

Adapted from `Claude_Scientific_Discovery_Lab/SCIENTIFIC_STANDARDS.md` and
`EXPERIMENT_PROTOCOL.md`.

## 1. Evidence labels

Every factual statement in an agent's output carries one label.

| Label | Meaning | Required with it |
|---|---|---|
| VERIFIED | The agent opened the primary artifact (JSON, CSV, log, code, manifest) in this session and it states the value | Path, and key or line |
| REPORTED | Stated in a report's prose or another agent's output; primary artifact not opened | Path of the report |
| INFERRED | A logical or mathematical consequence of VERIFIED or REPORTED facts | The premises |
| HYPOTHESIS | A proposed explanation with at least one rival | The rival and a falsifying observation |
| UNTESTED | A proposed experiment or improvement that has not been run | — |
| UNKNOWN | The information is absent from the repository | Who or what could supply it |

Never upgrade a label without doing the work it requires. Never write a number
without a path.

## 2. The claim ladder

Six claims are kept separate in every document, table and figure caption.

1. Class separation on fitted sources.
2. Recognition of unseen sources.
3. Dependence of the decision on spill-relevant information.
4. Concentration of attention or attribution on the liquid.
5. Spatial localization of the liquid.
6. Generalization to unseen scenes.

Rules:

- Evidence for a lower-numbered claim is never offered for a higher one.
- t-SNE separation, low semantic loss and prototype accuracy are evidence for
  claims 1 and 2 only.
- An attention map is evidence for claim 4 only after the method has passed a
  positive control and a randomisation check.
- A removal effect shows that the decision is sensitive to the removed cells.
  It does not show that those cells are liquid.
- Agreement with automatic candidates is not localization.
- Temporal prediction below persistence is a statement about representation
  trajectories, not about liquid dynamics.

## 3. Validity rules specific to this project

- State the cohort composition of every evaluation set: how many sources per
  class come from each resolution and naming cohort.
- State the clock policy of every model and whether the decision was checked
  under a changed clock origin.
- Report a no-training baseline (frozen-feature prototypes and a pixel
  statistic) beside every recognition result.
- Name the split role of every example video. cold56 is a training source;
  cold10 and cold13 are selection sources.
- The 16 selection sources are a development set. They have been used for
  tuning and repeated inspection and cannot support a generalization claim.
- "Source-excluded" means excluded by file ID and byte hash. Until the
  two-view pairing of cold sources is settled, count each adjacent-ID cold
  pair as one unit and state how many evaluation sources have a presumed
  partner in the training set.
- `clean_folderImage81` is a pooled set of frames from several scenes. Do not
  describe it as a recording or as a matched dry control.
- Durations and frame rates are conversion settings. Do not state physical
  time without a capture record.
- Do not select layers, heads, thresholds, fills or colour scales using the
  known failure videos.
- Qualitative focus judgements need a written rubric fixed before viewing,
  the full list of videos viewed, and the judgement for each one.
- No manual spatial annotation, MIL or segmentation decoder. Do not assume a
  first frame is dry. Do not turn example images into targets.

## 4. Statistical standards

- The unit of independence is the scene or session, not the file or the
  frame. Until scene groups exist, say so beside every interval.
- Give the count and an exact interval for proportions. A perfect score on 16
  sources has a 95% lower bound of 0.79 even if the sources were independent.
- A saturated metric cannot rank methods. Do not compare arms on it.
- Report every seed, split and variant that was run, including failures.
  A difference claimed between two arms needs more than one seed.
- Mark an analysis as exploratory if its outcome measure, threshold or arm was
  chosen after seeing results. Adjust for, or at least count, the number of
  comparisons.
- Windows from one recording are repeated measures of one source.

## 5. Reproducibility

- Every new experiment writes to a new directory under `reports/experiments/`.
  Nothing in an existing report directory is edited or deleted.
- Record the commit, a snapshot or hash of the code actually executed, data
  manifest hash, split hash, seed, checkpoint hashes, hardware, Slurm job and
  step, commands and exit codes.
- Reproduce the unperturbed saved output before interpreting any perturbed
  one, including cache precision and batch shape.
- `reports/` is not under version control. Until the researcher decides
  otherwise, treat the report tree as the only copy.

## 6. Approval gates

| Gate | Action | Who may proceed |
|---|---|---|
| G0 | Reading files, parsing report artifacts, literature search | Any agent, without asking |
| G1 | Editing `docs/research/`, `.claude/agents/` and `CLAUDE.md` | Principal Investigator, when the researcher has asked for documentation work |
| G2 | Editing `AGENTS.md`, `README.md`, other root or `docs/` Markdown | Principal Investigator, after the researcher approves the specific change |
| G3 | Writing or changing any Python, notebook, config, dependency or script; running tests | `senior-software-engineer` only, after the researcher approves a named experiment and its target files. `code-reviewer` reviews before anything runs |
| G4 | Any GPU job, inference or training run, including a learning-rate search | `distributed-training-engineer` only, after the researcher approves a named protocol and a compute ceiling and the code review has passed. `results-validator` checks the run before it is recorded |
| G5 | Collecting data, changing the annotation policy, changing an authoritative default | Researcher decision only |
| Never | Editing or deleting previous reports, checkpoints, caches or figures; spoofing a Slurm job ID; fabricating a citation, number or run | No one |

An approval covers the named experiment and budget only. A recommendation from
an agent is not an approval. A background-task notification is not an
approval.

A protocol is ready for the researcher only when it contains: the hypothesis
and its rival; the single change being tested; the outcome measure; the
decision rule fixed in advance; controls; the evaluation set with its cohort
composition; seeds; a compute estimate or the word UNKNOWN; and the verdicts
of the Statistical Evaluation Reviewer and the Skeptical Scientific Reviewer.

## 7. Verdicts

After an approved experiment, each hypothesis receives one verdict.

- **SUPPORTED:** the pre-stated prediction held, controls passed, and the
  effect is outside its uncertainty.
- **NOT SUPPORTED:** a clear contrary result in a test that could have
  detected the effect.
- **INCONCLUSIVE:** insufficient sensitivity, compromised design, or missing
  controls.

Moving a threshold after seeing the result makes the analysis exploratory.
