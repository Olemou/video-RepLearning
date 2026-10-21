# Video-supervised contrastive framework: semantic separation with preserved dynamics

Status: proposed architecture and experimental protocol. The existing ViCo
Eq. 10 baseline remains the measured implementation. The temporal adapter,
auxiliary objectives and cross-environment claims below are not trained or
validated by this specification.

## 1. Design conclusion

Use a rich ordered spatiotemporal representation for transfer, a dedicated
binary semantic projection for spill/no-spill discrimination, and a temporal
prediction/alignment branch. Apply the selected ViCo-LWIR Eq. 10 to the
semantic projection. Do not force every timestep, patch, or transferable
feature into two nearly identical class points.

No design can be declared universally optimal from 14 videos. The empirical
target is a model on the Pareto frontier of class separation, temporal
fidelity and held-out-domain performance. The current baseline establishes
optimization capability, not that frontier: raw loss 6.595e-9, 14/14 training
predictions, 13/14 held-out predictions, matching untouched pretrained features.

The frozen backbone already encodes temporal context. Averaging its outputs
does not prove complete loss of temporal information, but it removes the
explicit indexed sequence from the exported embedding. The current binary
embedding's near-antipodal class geometry is not evidence of preserved
within-class process dynamics. The upstream frozen features remain available
and should be retained as a separate output.

## 2. What should and should not be separated

- Complete observed spill episodes and clean episodes should separate in
  the semantic projection, including warm and cold spills in one spill class.
- Temporal states should reflect physically observable onset, spreading,
  persistence and recovery where those changes are visible.
- A clean prefix of a future spill episode may be identical to a prefix of
  an episode that remains clean. A deterministic causal encoder must produce
  the same state for identical observations. Global future labels must not
  force those prefixes apart.
- Static portions of a trajectory should be allowed to remain static.
  Different timestamps are not automatically different physical states.
- Object, surface and context information can remain useful in the rich
  representation even when the binary class projection should ignore it.

Thus, separable complete-trajectory embeddings and shared pre-event states
are compatible. Requiring disjoint state trajectories at every timestamp
would encode future-label leakage or nuisance shortcuts.

## 3. Architecture and exported representations

```text
Radiometrically consistent video, timestamps, validity masks
  -> whole-frame preprocessing and safe whole-clip augmentations
  -> pretrained V-JEPA spatial and temporal features, initially frozen
  -> global context token + a few learned spatial summary tokens
  -> small residual temporal adapter with explicit time differences
       |
       +-- rich ordered states H=[h_1,...,h_T] + retained dense features
       |      -> transferable features for new tasks
       |
       +-- coverage-aware mean + event-attention pooling
       |      -> semantic MLP -> unit z_sem in R^128
       |      -> selected ViCo Eq. 10 and class prototype readout
       |
       +-- temporal head d_t + causal multi-horizon predictor
              -> aligned-view consistency and fixed-teacher future targets
```

Keep the pretrained spatial grid until after optional learned spatial pooling.
Blindly averaging all spatial tokens before adaptation can dilute a small
spill. The spatial summaries receive only weak video supervision; their
attention weights must not be called ground-truth spill localization.

A concrete starting architecture for an expanded dataset is two residual
temporal blocks operating through a 128-dimensional bottleneck with four
attention heads, explicit elapsed-time/time-difference encoding and dropout
0.1. Upscale residuals back to the backbone feature width. These are initial
hyperparameters to validate, not measured optima. Start the residual path at
zero so the existing pretrained representation is recovered initially.

Use a compact 128-dimensional semantic projection, initially following the
tested 64-hidden-unit MLP. Combine masked mean pooling with event-attention
pooling so persistent scene information and short spill evidence are both
available. Check whether attention merely selects a hot object or background.

Export all of the following, with timestamps and masks:

1. `semantic_embedding`: compact unit vector for spill/no-spill retrieval
   and classification.
2. `trajectory_states`: rich ordered states for phase, onset, prediction
   and transfer; do not constrain the whole state to a class prototype.
3. `temporal_embedding`: the temporal head's ordered output.
4. `dense_features` or a documented link to their cache for spatial tasks.
5. Optional calibrated event scores, trained with appropriate temporal labels.

Semantic compactness is localized to the class head. Temporal losses and
retention constraints act upstream or on the temporal branch, not directly
on the binary projection. This avoids applying the old pretrained-feature
tether to a class vector that needs to change for discrimination.

For the current 14-video dataset, keep the backbone frozen and retain its
ordered features. Preserve the measured compact projector as the baseline.
Do not treat training a new temporal transformer on those 14 videos as
evidence of generalization. Learn the richer adapter first from a larger
diverse unlabeled-video corpus, then validate label adaptation on grouped data.

## 4. Preserve the selected semantic loss

For complete-video semantic embeddings z_i, use:

L_sem = mean_i mean_(j in P_i)
  -log( w_ij exp(cos(z_i,z_j)/tau) /
       [w_ij exp(cos(z_i,z_j)/tau)
        + sum_(k in N_i) alpha_ik exp(cos(z_i,z_k)/tau) + epsilon] ).

P_i contains same-label video views except the anchor. N_i contains opposite
labels. Other positives never enter the pair's denominator. Same-source
views remain positives, and independent same-class episodes are essential
additional positives. Labels are clean=0 and warm/cold spill=1.

Keep tau=0.1 fixed initially and epsilon=1e-6. Use log-space softplus
evaluation. Treat mining coefficients as detached selection statistics.
Do not obtain a smaller displayed loss by lowering temperature, subtracting
an entropy constant or shrinking negative mass during evaluation.

The current controls did not show an accuracy benefit from the paper's
positive uncertainty ranking. Therefore the simplest proposed default is
w_ij=1, with the literal Eq. 8 curriculum retained as a named ablation. Use
balanced negatives initially; introduce verified hard-negative selection as
an ablation after stable class learning. Preserve Eq. 7's square-root total
negative-mass convention when comparing uniform and hard-negative weights.
This changes coefficient policy, not the selected Eq. 10 partition function;
it is a proposed simplification, not a claim that the current literal-paper
run already used this policy.

Prefer cross-environment same-class positives and same-environment
opposite-class negatives, while maintaining broad sampling coverage.
Examples include clean versus spill captures on the same floor/camera and
spills of different temperatures across different rooms. Exclude ambiguous
labels, duplicate/overlapping sources and unverifiable negatives from mining.

## 5. Preserve dynamics with complementary temporal objectives

Let h_i,t be a rich state, d_i,t its temporal-head embedding, and F_0 a frozen
noncollapsed pretrained teacher. Let pi(t) map known corresponding timestamps
between two augmented views. A minimal joint objective is:

L_joint = L_sem + lambda_a L_align + lambda_p L_predict + lambda_k L_keep.

### Aligned-view consistency

L_align = mean_(i,t) rho(d_i,t^(a) - d_i,pi(t)^(b)).

Use known timestamp correspondence from the same source video. Align spatial
coordinates explicitly for patch-level comparisons after a flip. This loss
does not pull different times together and does not label every crop of a
spill video as containing a spill.

### Multi-horizon prediction

L_predict = mean_(i,t,Delta) rho(
  q_Delta(d_i,<=t) - stopgrad(LN(F_0(V_i, context ending at t+Delta)))) .

Use several physical-time horizons, for example 0.5, 1 and 2 seconds when the
camera/frame rate supports them. Predict standardized feature targets with
Huber loss as a simple starting point. An optional change-target term uses
teacher displacements between those times, emphasizing dynamics rather than
only persistent scene appearance. Do not equate all temporal changes with
spills; camera motion and unrelated object motion are relevant controls.

The frozen target makes predicting a constant sequence costly when the
observed process actually changes. Temporal-view agreement alone would not
provide that protection. Unpredictable futures retain irreducible error;
a probabilistic forecasting head can be evaluated later if calibration data
supports it. It would be an auxiliary dynamics model, not a replacement for
the selected semantic objective or an automatic import of prob.pdf's KL loss.

### Retention of broad pretrained information

L_keep = mean_(i,t) rho(LN(h_i,t) - stopgrad(LN(h_i,t^0))).

Use corresponding frozen features before the semantic projection. This
constrains forgetting in the transferable representation while allowing
the nonlinear class projection to become highly discriminative. Preserve
the frozen dense features separately so adaptation never becomes the only
available representation for downstream tasks.

Start auxiliary weights by matching their gradient scale on the shared
adapter at initialization, then choose among a small predefined set using
grouped validation. Do not call any arbitrary set of lambda values optimal.
Log gradient norms and conflicts; if class training damages temporal probes,
freeze the shared representation for the semantic refinement phase rather
than adding increasingly strong competing penalties to z_sem.

Optional variance/covariance regularization belongs on the rich or temporal
branch, and only if collapse measurements justify it. Do not impose full-rank
128-dimensional covariance on 14 independent trajectories. Do not force
motion in a genuinely static clean segment. Cross-video phase alignment is
optional and requires compatible processes or phase annotations; unrelated
videos must not be aligned just because both have label spill.

## 6. Causality is an encoder-input requirement

The existing V-JEPA encoder uses bidirectional attention within a clip. A
causal temporal adapter on top of full-episode cached features does not
remove future information already present in those features.

For forecasting, onset-delay measurements and online operation, encode only
prefixes or rolling windows ending at the decision time. A bidirectional
backbone is permissible inside a window that contains only past observations.
Cache the maximum source timestamp used by every state. Future prediction
targets may contain future frames; predictor inputs may not.

Maintain explicitly separate offline episode and causal prefix evaluations.
An acceptance test must verify that appending future frames cannot change
already emitted causal states or predictions. The current full-video cache
cannot by itself establish that property; new causal-context caches are
required for this part of the design.

## 7. Data coverage, sampling and normalization

Capture trajectories across independent rooms, surfaces, cameras, liquids,
objects, amounts, temperatures, emission/thermal conditions, people and
interactions. Include matched dry/hot objects and reflections as hard clean
controls. Capture both classes in every environment where feasible. If class
and environment are perfectly confounded, the data cannot identify whether
a cue is a spill or its environment, regardless of loss convergence.

Record acquisition-run, room, camera, surface, object, liquid and source
identifiers. Add reliable onset/phase annotations to a subset for temporal
validation. A positive video label is a bag label: at least one observed
portion contains a spill. It is not supervision for every frame or patch.
Clean-video windows can be treated as clean only if their video label was
verified. Uncertain windows from positive episodes remain locally unlabeled.

Tile long recordings into overlapping windows that collectively cover the
episode. A single 64-frame window is not a complete long trajectory. Use a
hierarchical episode aggregation step and retain window timestamps. Sample
multiple physical timescales without changing event order. All derived
windows and views inherit their acquisition group's split.

Use whole-frame aspect-preserving resizing/letterboxing and checkpoint
normalization for the frozen backbone. Preserve radiometric information
and calibration metadata where available. Avoid per-video intensity
normalization that would erase physically informative thermal contrast.

Use whole-clip horizontal flips where physically valid and mild sensor-realistic
noise as candidate augmentations. Apply geometry consistently across frames.
Avoid temporal shuffling/reversal, thermal inversion, arbitrary brightness
changes and erasing as invariances for the dynamics branch. Spatial or temporal
crops may inherit a positive label only when they are known to retain the spill;
otherwise use them only with correspondence/self-supervision that is valid.

Batch by class and acquisition group, seeking multiple independent videos
per class and environment diversity. The current full-data 28 views come
from only 14 independent videos; augmentation is not additional independent
evidence. An optional contrastive queue must remain training-only, track
source/group identity and control staleness. Larger batches/queues change
negative mass and the attainable loss reference.

## 8. Training schedule

1. Establish the existing frozen-backbone ViCo baseline and grouped splits.
   Preserve its converged classification result as a regression target.
2. On broader unlabeled training-only data, learn the temporal/spatial adapter
   with alignment, prediction and retention. Verify causality where required.
3. Add video labels and train the semantic head with Eq. 10 while maintaining
   temporal objectives. Prefer group-balanced sampling and worst-group
   validation over unvalidated nuisance-removal penalties. Domain information
   can be useful in the rich features; invariance is a task-specific goal.
4. Freeze the backbone, shared adapter and temporal head. Refine only semantic
   pooling/projector parameters on fixed training representations. Use the
   tested optimizer-state reset after a fixed curriculum/phase transition.
   Keep temperature, mining mass and the audit pair set fixed.
5. Select the deployable checkpoint using grouped validation and the temporal
   transfer criteria. Fit a probability calibrator only on validation data.
   Report a completely converged training checkpoint separately if continued
   semantic fitting harms held-out performance.

For cached deterministic head training, the measured starting recipe is
AdamW lr=1e-3, eps=1e-12, weight decay=0, a 100-update curriculum and a
single Adam-moment reset. Those settings are not automatically appropriate
for backbone fine-tuning or noisy minibatch temporal training. Keep the
backbone frozen with the current tiny dataset; unfreeze small adapters or
last blocks only when adequate data and held-out-domain evidence justify it.

## 9. What convergence must mean

The joint objective does not generally have a zero optimum: uncertain future
states, retention tradeoffs and incomplete labels can leave residual losses.
Converged binary training is not equivalent to a transferable trajectory model.

Use three independent acceptance gates:

1. **Semantic optimization:** raw Eq. 10, worst positive pair, class geometry,
   fixed-pair gradients and numerical replay stabilize. Require every audited
   valid positive pair to meet the declared tolerance on clean-label fits.
2. **Temporal fidelity:** aligned-view agreement, held-out prediction/change
   errors and independently annotated phase/onset probes remain useful.
   Avoid constant-state or timestamp-only solutions and artificial motion.
3. **Generalization:** locked group-held-out class and temporal metrics satisfy
   the operational requirements, including worst-group behavior and transfer.

Do not reuse the present absolute 1e-8 target unchanged for a larger batch.
For fixed alpha and w at an audit step, a pairwise unit-vector reference is

L_min(i,j) = log1p((C_i exp(-1/tau) + epsilon) / (w_ij exp(1/tau))),
where C_i = sum_negative_k alpha_ik.

It is the conditional optimum when all positives can coincide and the binary
class points can be antipodal with coefficients held fixed. Recompute the
appropriate reference if mining coefficients are recomputed at that ideal
geometry. Use a batch-aware absolute/relative excess tolerance, plus geometry
and held-out checks. With sqrt(N) negative mass and tau=0.1, sufficiently
large negative sets make an absolute 1e-8 threshold unattainable even for
perfect antipodal class points. Never report that as failed learning.

The existing result—6.595e-9 mean loss and all folds under 1e-8 worst-pair
loss—is a numerical result for its specific 28-view audit, not a universal
threshold or a promised epoch count for this larger framework.

## 10. Generalization and temporal evaluation

Split by acquisition run or scene before extracting windows or augmentations.
Use training, group-validation and locked test sets; exact-byte deduplication
alone is insufficient. Add leave-one-environment, object-family and task-out
protocols where the data actually support those distinctions. Keep a separate
set of new conditions for the final claim. Model selection must be predefined.

| Goal | Required evidence |
| --- | --- |
| Spill/no-spill discrimination | Balanced accuracy, AUROC/AUPRC, recall at an operational false-alarm rate |
| Online performance | Event recall, false alarms per unit time, onset delay using strictly causal inputs |
| Calibration | Brier/reliability measurements, validated selective-risk/abstention behavior |
| Temporal meaning | Phase probes, future/change prediction, annotated alignment, response to actual time reordering |
| Object/environment robustness | Matched-condition hard controls and held-out-group results, not only random folds |
| Cross-task transfer | Frozen or few-shot readouts on rich features for new tasks, compared with untouched V-JEPA |
| Spatial spill localization | Independently annotated localization evaluation; video labels alone do not prove it |
| Retention without collapse | Within-class and temporal diversity/effective-rank measurements relative to independent sample support |

Random temporal shuffling should disrupt genuinely dynamic trajectory probes,
but static segments need not respond. Check whether the semantic head works
from a single background frame or timing alone. Compare whole-video,
static-frame and order-ablated baselines. Inspect foreground/background
interventions only when physically and thermally valid. Domain predictability
in the rich transfer state is not itself a defect; domain leakage in the
binary head must be assessed conditional on label and data support.

Required ablations are: current converged baseline; rich-state export only;
temporal adapter; aligned-view loss; future/change targets; retention;
event pooling; uniform versus hard negatives; positive ranking; and limited
backbone adaptation. Add components incrementally and keep split/model-selection
rules fixed. Report run/scene-level uncertainty rather than treating correlated
frames as independent test samples.

## 11. Implementation interfaces and acceptance tests

Proposed cache schema contains source/group IDs, labels, view IDs, timestamps,
spatial grids, valid masks, dense teacher tokens, prefix-context end times and
the normalization/model provenance. Existing spatially pooled feature caches
support the baseline but cannot recover discarded patch detail or prove
causal encoding. Build separate versioned caches for those capabilities.

Required tests before claiming this design works:

- Appending future frames leaves already emitted causal features unchanged.
- Original/mirrored views align at the correct timestamps/spatial coordinates.
- A clean prefix never inherits a future spill-state label without evidence.
- No acquisition group, overlap window or derived view crosses data splits.
- Padding contributes to neither attention, aggregation nor auxiliary losses.
- Synthetic static and changing sequences produce appropriate temporal behavior.
- Semantic-only refinement leaves all temporal/backbone parameters unchanged.
- Retention targets are detached; predictor inputs never contain future features.
- Every active loss has an independent value/gradient check and clear reporting.
- Saved checkpoints replay all exported representations and their masks/times.

## 12. Research basis and limits of inference

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) supports using pretrained
  video features and feature prediction as a foundation. It does not prove
  transfer on these thermal recordings.
- [TCLR](https://arxiv.org/abs/2101.07974) motivates explicit temporal learning
  beyond global video contrast. Its different-time discrimination is not
  blindly copied for static or repeated physical states here.
- [Temporal Cycle-Consistency Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html)
  motivates alignment and phase-oriented evaluation. This proposal initially
  uses known same-video correspondences, avoiding assumed cross-video phases.
- [The Details Matter: Preventing Class Collapse](https://www.mdpi.com/2813-0324/3/1/4)
  motivates preserving information beyond a binary class label. The split
  representation and retention policy here are an engineering proposal.
- [VICReg](https://arxiv.org/abs/2105.04906) supports variance/covariance controls
  as optional collapse diagnostics/interventions, not a demand for artificial
  temporal variation or a replacement for the selected supervised objective.
- [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)
  motivates explicit model-selection and domain-held-out protocols rather
  than assuming a specialized loss guarantees domain transfer.

This specification combines research-supported principles with project-specific
design choices. Its new components require implementation and controlled
validation. The honest immediate result is a converged class-head baseline
plus a concrete path toward a temporally meaningful transferable framework.
