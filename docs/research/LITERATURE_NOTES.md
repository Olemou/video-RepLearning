# Literature notes

This file records which external sources the project has, how far each has
been verified, and what each can and cannot mean for this repository. It does
not copy bibliographies. Evidence labels are defined in
[SCIENTIFIC_STANDARDS.md](SCIENTIFIC_STANDARDS.md); ledger tags such as L08
refer to [RESEARCH_STATUS.md](RESEARCH_STATUS.md).

## 1. The three pools of sources

| Pool | Where | Verification status on 2026-11-10 |
|---|---|---|
| Sources cited inside the experiment reports | Listed by topic in [EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) section 8 | REPORTED. Not re-opened in the audit |
| The researcher's focused review of weakly supervised contrastive video learning | [`Weakly_Supervised_Contrastive_Learning_Video_Thermal_Spill_2026.md`](../../Claude_Scientific_Discovery_Lab/Weakly_Supervised_Contrastive_Learning_Video_Thermal_Spill_2026.md) | Checked on 2026-11-10 (section 7). All seven sources exist as described. Three were read in full text; four are behind a publisher paywall and were confirmed from metadata, an abstract or released code only |
| Sources found by the Literature and Novelty Researcher on 2026-11-10 | Section 7.3 | Opened by the agent; depth stated per item |
| Leads recalled by the Principal Investigator | [OPEN_QUESTIONS_AND_HYPOTHESES.md](OPEN_QUESTIONS_AND_HYPOTHESES.md) section 7 | Unverified leads |

Nothing in any pool may be cited as support for an experiment until the
Literature and Novelty Researcher has opened it and recorded how far it was
checked. A source checked from metadata or an abstract supports only what
that metadata or abstract says.

## 2. The researcher's review in brief

The review covers methods that learn from video-level labels. Its own reading
order and classification:

| Priority | Short name | What the review says it contributes | Uses MIL | Localises in space |
|---|---|---|---|---|
| P1 | DSCIL | Selecting likely normal and likely anomalous instances inside positive videos, then contrasting them | Not stated as MIL | No; instances are segments |
| P1 | NF-DCL | Debiased contrastive loss with synthetic normal features | Yes, with top-k ranking | No |
| P1 | Learning Where and When | Patch-level anomaly scores from video labels | Yes, patch-level | Yes |
| P2 | MGFN++ | Magnitude contrastive learning; scene and motion variation | Not stated | Mainly temporal |
| P2 | A2VAD | Foreground and background contrastive alignment with vision–language prompts | Not stated | Segment alignment |
| P2 | Two-Stage Pseudo-Label Refinement | Treating first-stage outputs as noisy pseudo-labels | Not stated | Temporal |
| Background | STPrompt (2024) | Full-frame features can follow the dominant background | Not stated | Yes |

Every cell above is the review's description (REPORTED). The "Uses MIL"
column is corrected by the verification in section 7.2: none of the seven was
confirmed to be free of MIL-style pooling. Titles, authors, venues and links
are in the review and are not repeated here. The review jumps
from section 6 to section 8; no section 7 is present.

## 3. What this repository already knows about the review's questions

The review poses three hypotheses and six studies "to design, not run". The
audit can already answer or constrain several of them.

| Review item | Project hypothesis | What existing evidence says |
|---|---|---|
| Its H1: a global positive label makes the encoder group background patches with spill patches | H7 in the hypotheses file | The authoritative pipeline forms no patch pairs at all: positives are whole-video embeddings of the same class (REPORTED, code audit of `src/vision_model/training/vico.py`). The patch-level variant did inherit the video label for every patch, and patch features homogenised: within-image cosine 0.367 → 0.952 (L08). The gradient audit found that such labels admit a class-constant solution (L11) |
| Its H2: a global contrastive objective under-constrains location | H2 | The mathematical part is established (L18). Whether it is the binding constraint is untested |
| Its H3: thermal shortcuts and domain shift dominate the decision | H1 | Shortcut availability is measured (L16); label and acquisition cohort are coupled in 155 of 156 sources; the one cross-cohort file is a pooled image set, not a recording |
| Study: audit the current contrastive pairs | — | Answerable by reading code, and answered above. No new experiment needed |
| Study: scene-disjoint evaluation | — | Not possible today. No scene or session identifiers exist (roadmap questions A6 and A11), and cold sources appear to be two-view pairs that cross the existing splits |
| Study: spatial validity benchmark with IoU, pointing game, deletion and insertion | — | Deletion tests exist (L18, L20). Insertion tests do not. Mask-based metrics need a reference the project does not have |
| Study: positive-bag instance-selection ablation | — | One limited precedent: latent MIL probes kept 16/16 and produced scores that followed token norm (L06). Excluded by the current no-MIL constraint |
| Study: debiased contrastive ablation | — | Recognition metrics are saturated at the no-training baseline, so AUROC and F1 cannot show a gain on the current sources |
| Study: temporal consistency, only if videos exist | — | Genuine frame sequences exist, but frame timing is assigned at conversion and the model sees a 3 s centred window (status file, section 3) |
| Caveat: check whether the data are videos or independent images | — | Videos built from image folders; 15 fps assigned at conversion |

## 4. Limits on transfer to this project

These are the audit's inferences from project evidence (INFERRED). The
Literature and Novelty Researcher and the Skeptical Scientific Reviewer must
test each one against the papers themselves.

1. **The bag assumption is broken by the cohort coupling.** Instance selection
   inside positive videos works when the normal instances of a positive video
   resemble the instances of negative videos. Here they do not: tokens over
   empty letterbox padding in a cold video already carry the class (L22), and a
   spatially constant field satisfies automatic patch targets better than the
   real one (L14). Under these conditions a top-k or instance-selection rule
   has no reason to prefer liquid over any other patch of a cold-cohort video.
   This applies to DSCIL-style selection, to patch-level MIL and to debiased
   contrastive losses alike. It is the same obstacle the audit identifies
   behind the fourteen levers already tried. The dataset audit supports the
   premise for the 80 study sources, of the 156, whose contact sheets were
   viewed: cold sources are steep high-angle, dark views and clean sources
   are not, and no matched subset exists
   ([RESEARCH_STATUS.md](RESEARCH_STATUS.md) section 3.1). The other 76
   sources were not viewed.
2. **Most of the methods are MIL, in name or in function.** Three are
   explicit MIL and three more pool the top-k instance scores against the
   video label, which is the MIL mechanism without the name (section 7.2).
   The project excludes MIL. Adopting one is a change of constraint, which
   only the researcher can make.
3. **Instances in these papers are mostly temporal segments.** The project's
   failure is spatial, inside a window in which the scene is nearly static.
4. **Surveillance RGB is not long-wave infrared.** Vision–language alignment
   trained on RGB has no demonstrated meaning for thermal frames. The review
   says so itself.
5. **Spatial claims in these papers are scored against annotations.** In
   every source that could be read, temporal claims use frame or segment
   labels and spatial claims use manually drawn boxes on the test set
   (section 7.2). The project has none, so their localization metrics cannot
   be reproduced here as things stand.
6. **Reported gains are on those papers' benchmarks.** None is evidence about
   this dataset.

The review's own cautions agree with points 3 to 6. Point 1 is specific to
this repository and is not in the review.

## 5. Two places where the review meets a standing constraint

| Review recommendation | Standing constraint | Handling |
|---|---|---|
| Compare global pooling with a patch-scoring MIL variant, and test instance selection | No MIL | Not adopted. Listed as a researcher decision in the roadmap. Even if allowed, point 1 above makes it non-decisive before matched data exists |
| "Commission a small, held-out evaluation-only set" of masks if none exist | No manual spatial annotation | Not adopted. Listed as a researcher decision in the roadmap. The roadmap's default remains an annotation-free reference from matched dry and spill recordings |

## 6. Work for the Literature and Novelty Researcher

Tasks 1 to 6 were run on 2026-11-10; results are in section 7.

1. Open each of the seven sources in the review. Confirm title, authors, venue
   and date, and record original text, abstract only, or not accessible.
   **Done; four not accessible in full text.**
2. For each P1 paper, extract the exact assumption that links video labels to
   instance labels, and state whether point 1 of section 4 holds against it.
   **Done for "Learning Where and When" from the text and for DSCIL from its
   code; not possible for NF-DCL.**
3. For "Learning Where and When", record how patch scores are defined, what
   annotations its evaluation uses, and the smallest region it resolves.
   **Done.**
4. Search for work that addresses weak localization when the label is
   confounded with the recording source. **Done; three sources found.**
5. Search for annotation-free evaluation of localization, including
   paired before-and-after recordings and synthetic insertion. **Partly done:
   one search. Synthetic insertion has precedent on images; nothing found on
   paired recordings, which is not a claim of absence.**
6. Report anything in these papers that contradicts the audit's inferences.
   **Done; nothing found.**

Still open:

7. Obtain the full text of DSCIL, NF-DCL, A2VAD and MGFN++ (the researcher may
   have institutional access) and check the review's method descriptions.
8. Verify the sources cited inside the reports
   ([EVIDENCE_ANALYSIS.md](EVIDENCE_ANALYSIS.md) section 8) and the leads in
   the hypotheses file, section 7.
9. Search specifically for attribution sanity checks on video transformers and
   for paired-recording localization references.

## 7. Verification results (10 November 2026)

Run by a general-purpose agent following the Literature and Novelty
Researcher definition, read-only with web access. "Opened" means the agent
fetched that page; web pages reach an agent through a tool that summarises
them, so quoted phrases must be rechecked against the source before
publication. The Principal Investigator re-opened three arXiv abstract pages
(STPrompt, "Learning Where and When" and the frame-level AUC audit) and
confirmed their titles, authors and the quoted abstract sentences. Nothing
else here was re-checked by the Principal Investigator.

### 7.1 The seven sources in the review

| Short name | Exists as the review describes | Depth reached | Review's claim |
|---|---|---|---|
| DSCIL (*Pattern Recognition* 172, 2026) | Yes | Metadata and the authors' released code; paper not readable | Partly supported. The code selects low-magnitude clips of abnormal videos and trains them toward the normal label. Its only "contrastive" term is a ratio of nearest-neighbour distances inside each bag, not a contrast between the selected sets. Whether the code matches the paper is UNKNOWN |
| NF-DCL (*CVIU* 269, 2026) | Yes | Metadata and keywords | Not checked against text |
| Learning Where and When (arXiv 2606.29498, June 2026) | Yes | Full text | Supported. Scores on a 7×7 grid per 16-frame segment; the training positive is the strongest 2×2 block; explicit multiple instance learning with a cross-entropy loss; no contrastive loss |
| MGFN++ (*Pattern Recognition* 180, 2026) | Yes | Abstract; predecessor MGFN in full | Abstract supports the description. The authors state that existing methods rely heavily on feature magnitude and overlook scene variation |
| A2VAD (*Pattern Recognition* 180, 2026) | Yes | Metadata and keywords | Foreground and background alignment claim not checked |
| Two-Stage Pseudo-Label Refinement (*CSSE* 50(1), 2026) | Yes | Full text | Supported. The paper states its own limits: one dataset, a single run |
| STPrompt (arXiv 2408.05905, ACM MM 2024) | Yes | Full text | Supported, including the statement that frame-level methods "may be misled by the dominant background information" |

### 7.2 What each method assumes

| Method | MIL or top-k pooling against the video label | What it assumes about normal content inside positive videos | Reference used to score localization |
|---|---|---|---|
| DSCIL | Yes in function (code) | Low-magnitude clips of abnormal videos are normal, judged against normal-video magnitudes | UNKNOWN |
| NF-DCL | Yes (keywords; review) | Normal-video features model the normal part of abnormal videos (REPORTED) | UNKNOWN |
| Learning Where and When | Yes, explicit, temporal and spatial | Every patch of a normal video is a negative instance | Manually drawn boxes on test sets |
| MGFN++ | Predecessor uses top-k magnitudes; MGFN++ UNKNOWN | Magnitude separates classes | Temporal frame labels; no spatial |
| A2VAD | UNKNOWN | UNKNOWN | UNKNOWN |
| Two-Stage | Top-k pooling with cross-entropy in both stages | First-stage scores are "noisy but informative" | Temporal segment annotations |
| STPrompt | Yes, explicit | Background equals normal; anomalies are intense human motion | Manually drawn spatio-temporal boxes on test sets |

Three consequences for this project (INFERRED):

- No reviewed method is verified to be free of MIL-style pooling.
- Every assumption in the third column requires the normal part of a positive
  video to look like negative videos. Under the cohort coupling it does not,
  which is point 1 of section 4.
- No reviewed paper measures localization without annotated ground truth.

### 7.3 Additional sources found

| Source | Depth | What it says | Why it matters here |
|---|---|---|---|
| "Auditing Frame-Level AUC in Weakly Supervised Video Anomaly Detection" (arXiv 2608.11985; ECCV 2026 workshop) | Abstract re-opened by the Principal Investigator; body read by the agent | On normal frames alone, "every model we examine separates videos by recording properties, such as resolution and color encoding"; pooled AUC "credits both event localization and differences between video sources" | The closest published account of this project's cohort confound. Its negatives-only bias audit needs no masks |
| "Localizing to Debias" (arXiv 2608.12045) | Read by the agent | Adding patch-level localization reduced a source-bias measure only slightly; the authors call it a baseline, not a solution | Evidence from elsewhere that localization modules do not remove source bias |
| "Rethinking Metrics and Benchmarks of Video Anomaly Detection" (arXiv 2505.19022) | Read by the agent | Benchmarks lack normal and abnormal videos of the same scene; on generated same-scene normals, weakly supervised methods give high false-alarm rates | Supports matched same-scene recordings (roadmap stage D) as the decisive test |
| "Sanity Checks for Saliency Maps" (arXiv 1810.03292) | Abstract | Some saliency methods are independent of model and data | Precedent for the randomisation check (roadmap B1); image CNNs only |
| "Benchmarking Attribution Methods with Relative Feature Importance" (arXiv 1907.09701) | Abstract | Ground truth for attribution by construction | Precedent for the planted-cause control (roadmap stage C); images only |
| "Warm Liquid Spill Detection and Tracking Using Thermal Imaging" (Rouhafzay, Tian, Payeur) | PDF read by the agent | Classifier accuracy 99.2% while Grad-CAM localization scored IoU 0.56 against 0.69 for a watershed method, on one test sequence scored with manual boxes | The same gap between recognition and localization in thermal spill data; warm liquid, one scene |
| "Detecting spills using thermal imaging, pretrained deep learning models, and a robotic platform" (arXiv 2510.08770) | Abstract | Up to 100% classification accuracy on 4,000 images; no localization | Saturated thermal spill recognition without a localization claim |
| MGFN (arXiv 2211.15098) | Read by the agent | Feature magnitude "also depends on other attributes of the video" | A norm outlier is not evidence of the target |

No attribution sanity-check study specific to video transformers was found,
and no work on near-total label and source coupling. Neither is a claim of
absence. Unverified leads the agent listed but did not open: UMIL (CVPR
2023), SMERF, SaCo (arXiv 2404.01415), "Spatio-Temporal Perturbations for
Video Attribution" (arXiv 2109.00222), D²MIL (CVPR 2026).
