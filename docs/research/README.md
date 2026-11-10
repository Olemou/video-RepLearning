# Research documentation

This directory holds the research state of the thermal spill project: what was
done, what it shows, what is unresolved, and how the research team works. It
was created by the audit of 2026-11-10.

The research question: the model separates spill and non-spill videos, but its
attention does not reliably fall on the spill. Does it learn spill-relevant
features, does it rely on spurious correlations, or are the objective, the
protocol or the interpretation method at fault?

## Read in this order

| File | Contents |
|---|---|
| [RESEARCH_STATUS.md](RESEARCH_STATUS.md) | Claim ladder, data facts, the ledger of all completed experiments, levers already tried |
| [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) | What is established, what failed, confounders, methodological weaknesses, missing baselines, documentation inconsistencies |
| [OPEN_QUESTIONS_AND_HYPOTHESES.md](OPEN_QUESTIONS_AND_HYPOTHESES.md) | Unresolved questions and seven competing hypotheses, all untested |
| [LITERATURE_NOTES.md](LITERATURE_NOTES.md) | The three pools of external sources, their verification status, and what the researcher's literature review means for this project |
| [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) | Proposed experiments in stages, and the decisions requested from the researcher |
| [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) | The nine research agents and four execution agents, their boundaries, the workflow and handoffs |
| [SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md) | Evidence labels, claim rules, statistical and reproducibility standards, approval gates |
| [TEMPLATES.md](TEMPLATES.md) | Required formats for agent outputs and experiment protocols |

## Where other information lives

| Need | Location | Status |
|---|---|---|
| Principal Investigator instructions | [CLAUDE.md](../../CLAUDE.md) | Current |
| Canonical pipeline, loss and engineering rules | [AGENTS.md](../../AGENTS.md) | Current |
| Agent definitions | [.claude/agents/](../../.claude/agents/) | Current |
| Commands and executable defaults | [README.md](../../README.md) | Current, with some sections from the 14-video study |
| Opt-in branches | [experiment81.md](../experiment81.md), [local-thermal.md](../local-thermal.md), [dense-spill.md](../dense-spill.md) | Current |
| Learning-rate and batch searches | [trajectory-lr-search.md](../trajectory-lr-search.md), [trajectory-80-batch-search.md](../trajectory-80-batch-search.md) | Historical record |
| Plain-language pipeline tutorial | [trajectory-learning-explained.md](../trajectory-learning-explained.md) | Current |
| Earlier proposals and task briefs | [video_supervised_contrastive_framework.md](../video_supervised_contrastive_framework.md), `vico.md`, `info.md`, `audit.md` at the root | Historical |
| Earlier annotation path | [spill-localization.md](../spill-localization.md), `cluster/README.md` | Partly superseded by the no-annotation constraint |
| Individual experiment reports | `reports/experiments/<directory>/report.md` | Primary record; not under version control |
| Agent template this team was adapted from | `Claude_Scientific_Discovery_Lab/` | Unmodified |
| Researcher's literature review (weakly supervised contrastive video learning, 2026) | `Claude_Scientific_Discovery_Lab/Weakly_Supervised_Contrastive_Learning_Video_Thermal_Spill_2026.md` | Researcher-supplied; sources not yet verified by the team |

## Maintenance

- The Principal Investigator is the only writer of this directory.
- Status and history are kept here and nowhere else. Other files link here.
- A number enters these files only with an evidence label and a path.
- After each approved experiment: add a ledger row, update the claim ladder,
  update hypothesis statuses, and record the decision memo's location.
