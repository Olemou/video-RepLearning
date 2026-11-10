# Research deliverable templates

Required fields for agent outputs. Leave a field as `UNKNOWN` instead of
guessing. Evidence labels are defined in
[SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md). Adapted from
`Claude_Scientific_Discovery_Lab/OUTPUT_TEMPLATES.md` and
`EXPERIMENT_PROTOCOL.md`.

## A. Evidence ledger entry (Research Evidence Auditor)

```markdown
Ledger tag and directory:
Date and hardware:
Question tested:
Design: arms, trainable parameters, objective, data, split, seeds, updates
Measured results: value — label — path and key
Report's own conclusion (quoted) and path:
Claim addressed (ladder number) and verdict:
Chosen before or after seeing results:
Conflicts between prose and artifacts:
Provenance gaps:
```

## B. Confounder record (Dataset and Leakage Auditor)

```markdown
Variable:
How it relates to the label: counts per class
Evidence and path:
Could the label be predicted from it alone: measured / not measured
Has the model's dependence on it been tested: test, result, path
Affected splits and evaluation sets:
What information would resolve it, and who holds it:
```

## C. Hypothesis card (both scientists)

```markdown
# H<n> — Short falsifiable title
Status: UNTESTED
Observation explained:
Mechanism:
Existing evidence for, with labels and paths:
Existing evidence against, with labels and paths:
Rival hypothesis:
Prediction, including the regime where it should fail:
Minimal test:
Negative control:
Falsifying observation:
What result would change the recommendation:
```

## D. Literature matrix row (Literature and Novelty Researcher)

| Source (title, authors, year, venue, stable link) | Checked: original / abstract only / unverified lead | Task and data | Mechanism claimed | Transfers to this project because | Does not transfer because | Bears on which hypothesis |
|---|---|---|---|---|---|---|
| | | | | | | |

## E. Adversarial review (Skeptical Scientific Reviewer)

```markdown
Item reviewed: claim / hypothesis card / protocol
Strongest alternative explanation:
Confound, leakage or selection effect:
Missing baseline or control:
Is it distinguishable from its rival with the proposed test:
Novelty or prior-art issue:
Decisive measurement demanded:
Verdict: acceptable / revise / reject / cannot judge
Revised wording of the claim, if any:
```

## F. Statistical review (Statistical Evaluation Reviewer)

```markdown
Item reviewed:
Unit of independence and effective sample size:
Outcome measure and whether it can move (not saturated):
Uncertainty method and whether it fits the unit:
Number of comparisons and how they are handled:
Seeds, splits and variants reported versus run:
Pre-specified or exploratory:
Smallest effect the design could detect: value or UNKNOWN
Verdict: sound / revise / cannot support the claim
```

## G. Experiment protocol (Experiment Designer)

```markdown
Study ID and date:
Hypothesis tested and its rival:
Single change versus the reference:
Reference: checkpoint hash, commit, split
Evaluation set and its cohort composition:
Leakage paths excluded:
Outcome measure:
Decision rule fixed in advance: SUPPORTED if / NOT SUPPORTED if / otherwise INCONCLUSIVE
Controls: negative, positive, random
Seeds and repetitions:
Analysis plan and uncertainty method:
Compute estimate and its basis, or UNKNOWN:
Files that would be created or changed:
Stop rule:
Constraints checked: no manual spatial annotation, MIL or decoder; originals preserved
Statistical review verdict:
Skeptical review verdict:
Approval: not requested / requested / granted on <date> for <scope>
```

Comparison matrix for the protocol:

| ID | Arm | Change versus reference | Same data, split, seed, budget | Purpose |
|---|---|---|---|---|
| E0 | Reference | None | Yes | Reproduction |
| E1 | Candidate | One mechanism | Yes | Main hypothesis |
| E2 | Negative control | Mechanism removed or sham | Yes | Rule out a confound |
| E3 | Positive control | Known answer | Yes | Show the measurement can detect an effect |

## H. Decision memo (Principal Investigator, after an approved experiment)

```markdown
Date and study ID:
Hypothesis and the prediction stated before the run:
What was run: arms, seeds, failures, deviations from protocol
Primary result with uncertainty:
Controls:
Verdict per hypothesis: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
Claims on the ladder that changed status:
Alternative explanations still open:
Provenance gaps:
Next single most informative experiment:
```
