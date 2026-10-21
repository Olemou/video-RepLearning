# Proposed Research Plan — Spatial Semantic Learning with Optional Mirror Equivariance

## Current scope and authorization

The initial task was limited to updating this document. The subsequent instruction
“/home/olemou/video-RepLearning/info.md do whst inside” explicitly authorized
implementation and the bounded experiment described below.

Execution results are recorded in
[`reports/experiments/spatial_semantic_mirror_warmup_80_sources_20261021/report.md`](reports/experiments/spatial_semantic_mirror_warmup_80_sources_20261021/report.md).
The spatial branch and mirror objective were implemented without a decoder.
Both variants retain 16/16 source-validation classification, but warmup mirror
regularization collapses attention to essentially uniform maps. Localization
is not established. The unchanged temporal checkpoint remains the supported
recognition baseline. The remaining sections preserve the experimental plan
and its scientific constraints; optional joint fine-tuning was not justified.

## Mission

Act as a **Principal Research Scientist and Senior PyTorch Distributed Systems Engineer**, specializing in self-supervised visual representation learning, V-JEPA, weakly supervised localization, contrastive learning, equivariant attention, and multi-GPU training.

The proposed future task is to implement, scientifically validate, and experimentally evaluate a **new spatial semantic branch alongside the existing causal temporal branch**.

The objective is to investigate whether class-discriminative spatial attention can improve liquid-spill localization while preserving the measured functionality of the existing trajectory representation model. Do not assume that the existing model is fully converged or that the proposed branch will localize liquid.

### Research provenance

The proposed architecture and combined objective are an adaptation for this repository. **No single cited paper proposes this complete V-JEPA + trajectory + ViCo + mirror-loss system.**

**Class-specific localization reference:**

Xu et al., *Multi-class Token Transformer for Weakly Supervised Semantic Segmentation*, CVPR 2022 (MCTformer).

https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Class_Token_Transformer_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf

MCTformer learns class-to-patch attention using image-level labels. Its localization-map generation stage is distinct from its subsequent segmentation training. Adapting that idea to frozen causal video features is a new experiment; its published segmentation results do not establish performance for this pipeline.

**Equivariance reference:**

Wang et al., *Self-Supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation*, CVPR 2020.

https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf

SEAM provides a published precedent for consistency regularization of class activation maps under spatial transformations. The specific horizontal-flip attention loss named `L_mirror` below was proposed in this conversation. It is **not a term from ViCo-LWIR**, not an exact reproduction of SEAM's complete objective, and not a demonstrated best choice for this dataset. Its coefficient is unvalidated.

**Related attention/feature regularization references:**

- Yang et al., *MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation*, AAAI 2025: https://arxiv.org/html/2412.11076v1
- Ru et al., *Token Contrast for Weakly-Supervised Semantic Segmentation*, CVPR 2023 (ToCo): https://openaccess.thecvf.com/content/CVPR2023/papers/Ru_Token_Contrast_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf

MoRe studies irrelevant class-attention activations; ToCo studies patch-feature over-smoothing. Their attention/feature objectives may inform later ablations. Their complete systems contain segmentation components; do not describe them as entirely decoder-free or import all their training augmentations into this pipeline.

After implementation is authorized, inspect the original methods before choosing an adaptation. Record which equations are reproduced and which are newly proposed.

Do not implement a segmentation decoder.

---

## 1. Existing repository and experiment

Repository:

`/home/olemou/video-RepLearning`

Existing experiment:

`/home/olemou/video-RepLearning/reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021`

ViCo-LWIR reference:

`/home/olemou/video-RepLearning/loss_doc/VicoLWIR.pdf`

The completed audit is documented at:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/research_diagnosis/report.md`

Use these measured findings as the starting point, rather than repeating the initial investigation:

- The run fitted 64 source videos and used 16 source-validation videos. The validation split also informed earlier learning-rate/batch selection; it is not an independent final test.
- All 80 videos are byte-unique: 40 `clean` and 40 `cold`, with **no warm videos** in this experiment. Class labels are `clean -> 0`, `cold/warm -> 1` generally.
- The fitted checkpoint achieves 16/16 source-validation prototype classification and perfect measured class retrieval. Frozen V-JEPA mean/first-state prototypes also achieve 16/16, as does a classifier using four global intensity statistics.
- The neighboring t-SNE contains both training and validation points. Its checkpoint has exactly the same model tensors as the audited checkpoint, although the serialized checkpoint hashes differ. The exact historical source version of the original visualizations was not recorded.
- The extractor averages the final tubelet's 256 spatial tokens before caching. `TrajectoryModel` receives `[B,T,1024]`; its attention selects times, not image patches. This spatial-grid bottleneck is confirmed.
- The semantic objective constrains whole-episode embeddings. It supplies no verified spill-region labels. Scene/acquisition shortcuts are a strong competing explanation; the audit does not prove that they are the model's exclusive decision mechanism.
- Existing patch-preserving, intermediate-layer, MIL and small-adapter probes all achieved 16/16 classification without establishing localization improvement. Retaining patches or adding attention alone is therefore insufficient evidence of a fix.
- The empty saved temporal positive-gradient map comes from clipping nonzero negative gradients. It is not evidence of a disconnected gradient path or no temporal dependence. Frozen V-JEPA internal sensitivity maps are not learned spill masks.
- Validation future-prediction loss is 0.0269353 versus frozen persistence 0.0390788 on the selection split. Its late slope still improves; do not call the run fully converged.
- The user confirmed that there are **no spill masks**. Without subsequent verified spatial annotations, localization accuracy and improvements remain unproven.

The new hypotheses to test are whether class-specific spatial selection improves over the earlier generic pooling probes, and whether optional equivariance regularization improves useful spatial consistency. Neither hypothesis is established by the cited papers or the previous classification results.

### Non-negotiable requirements

- Preserve the existing temporal architecture and its trained checkpoint.
- Keep V-JEPA frozen for the first experimental stage.
- Preserve existing alignment, prediction, and retention losses.
- Preserve the causal temporal mask.
- Do not assign episode-level labels to every intermediate temporal state or spatial patch.
- Do not modify the original experiment artifacts.
- Use the same 80 source videos for the controlled experiment.
- Preserve the original 64/16 source split for direct comparisons and identify it as a selection split. Scene-held-out evaluation requires verified scene groups and a separate recorded protocol.
- Keep official checkpoint normalization and `preserve_field_of_view=True`; preserve the supported clip-consistent thermal augmentation policy.
- Keep `past_only_prefix_last_tubelet_v1`. Never substitute tokens from a full-video encode for causal-prefix features.
- Treat mirror consistency as an optional candidate regularizer, not a required or validated localization loss.
- Do not claim that good classification or t-SNE separation proves correct spill localization.

## 2. Architecture

Implement two parallel branches operating on frozen V-JEPA representations.

### Branch A — Spatial semantic representation

Retain the spatial patch tokens from V-JEPA before global spatial averaging.

Target spatial resolution:

\[
H_p=W_p=16,\qquad N=256
\]

For observed prefix/tubelet index \(t\), not an individual decoded frame:

\[
P_t\in\mathbb R^{256\times D}
\]

Verify the actual V-JEPA patch/tubelet grid. Do not reshape arbitrary tokens into 16×16.

For this checkpoint at 256×256, the native grid is 16×16, with two decoded frames per tubelet and width \(D=1024\). Preserve its native grid. A different checkpoint or resolution is a separate recorded experiment; do not silently resample or change it.

Preserve the mapping between tokens and video frame timestamps, especially if the backbone uses temporal tubelets.

For the first proposed architecture, use a lightweight trainable spatial Transformer/class-token adapter after the frozen encoder, with class-conditioned attention. This differs from assuming that a generic single-query attention pool or the prior MIL probe solves localization. The following expresses the intended class-specific scoring, not an equation copied from MCTformer:

\[
s_{t,i}^{(c)}=f_{\theta,c}(p_{t,i})
\]

where \(c\) identifies a semantic class and \(i\) identifies a spatial token.

Produce normalized class-conditioned attention maps:

\[
A_{t,i}^{(c)}
=
\operatorname{softmax}_i
\left(\frac{s_{t,i}^{(c)}}{\tau_s}\right)
\]

The module should support separate spill and no-spill class scores, but must not assume that a no-spill video contains a meaningful foreground spill mask.

Exclude invalid tubelets and letterbox-only patches from spatial pooling and mirror-map comparisons. Document the policy for partially covered content patches. Preserve the original global teacher pooling separately, so the temporal baseline does not change.

Aggregate spatial patch features:

\[
v_t^{(c)}=\sum_i A_{t,i}^{(c)}p_{t,i}
\]

Then construct a whole-episode semantic embedding:

\[
z_{\text{spatial}}
=
\operatorname{Normalize}
\left(
g_\theta(\operatorname{TemporalAggregate}(v_1,\ldots,v_T))
\right)
\]

The spatial branch must remain lightweight and compatible with frozen V-JEPA.

Use the existing ViCo loss on the episode embedding.

**Important:** Class-specific spatial attention must be learned through a genuine discriminative objective. Do not simply generate two softmax maps and assume that they represent distinct semantic regions.

If class-conditioned maps are used, document how each class receives supervision and how the correct class map is selected for training and visualization.

Construct every episode embedding with the **same label-independent routing**, for example a fixed-order combination of both class-token summaries. Do not choose the embedding's class token using the ground-truth video label: that could encode the answer without learning the image. At inference, class-map selection must use model predictions or an explicitly requested target, not validation ground truth. A clean-class query is not a ground-truth background mask.

Any temporal aggregation in this branch must respect validity masks. If prefix decisions are exposed, they must consume only observed prefixes. Episode-level labels must not be broadcast as positive labels onto every prefix or patch.

### Branch B — Existing temporal representation

Preserve the original temporal pipeline:

\[
X_t=\operatorname{GlobalPool}(P_t)
\]

\[
S=\operatorname{TrajectoryModel}(X)
\]

Maintain the current:

- Causal Transformer.
- Residual temporal adaptation.
- Dynamics representation.
- Alignment loss.
- Prediction loss.
- Retention loss.
- Timestamp handling.
- Padding and validity masking.

Do not replace the original global teacher features with spatial attention-pooled features in the temporal branch.

Keep these frozen global means as the detached future and retention targets. The spatial branch has its own learned semantic input path. This separation prevents the retention objective from directly forcing spatial selection back toward the global mean.

### Cache reuse and gradient connectivity

The accepted audit cache already contains identity/mirror patch tokens at blocks 5, 11, 17 and 23:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/research_diagnosis/spatial_matched_precision/spatial_tokens.pt`

Its shape is `[80,2,6,4,256,1024]`. Its final-layer spatial means reproduce the original cached features exactly. It can support the first bounded spatial-branch experiment. The original eight thermal views have pooled features, but their spatial tokens would need additional extraction if those views are included in a subsequent matched protocol.

Do not use the earlier audit `research_diagnosis/spatial/` cache: it omitted the original autocast context and is explicitly excluded. Preserve the accepted extraction's precision, preprocessing and frame alignment.

Frozen cached patches may remain detached; trainable layers applied to them still receive gradients. The new spatial adapter, aggregation and semantic head must be registered in the model, optimizer and DDP wrapper and remain connected to ViCo and the optional mirror objective.

The current `TrajectoryModel.forward()` defaults to `detach_teacher=True`, and the current trajectory-loss/distributed wrappers detach their inputs. Do not route a learned spatial representation through these detach operations. Prefer the independent spatial semantic branch described here. If a future variant feeds learned features into `TrajectoryModel`, explicitly separate its differentiable observed input from detached teacher targets and audit every wrapper's gradient path.

---

## 3. Mirror-equivariant attention

After the no-mirror baseline, test horizontal reflection consistency as an explicitly proposed SEAM-related adaptation.

Let:

\[
F(x)=\operatorname{HorizontalFlip}(x)
\]

Let \(A_\theta(x)\) be the predicted spatial attention map.

The desired equivariance relationship is:

\[
A_\theta(F(x))\approx F(A_\theta(x))
\]

This is equivariance, not invariance.

The attention map should reflect spatially when the image is reflected.

### Mirror loss

Compute:

\[
A=A_\theta(x)
\]

\[
A'=A_\theta(F(x))
\]

Undo the reflection on the second prediction:

\[
\widetilde A'=F^{-1}(A')
\]

For horizontal reflection:

\[
F^{-1}=F
\]

Define:

\[
\boxed{
L_{\text{mirror}}
=
\frac{1}{|\mathcal V|}
\sum_{t\in\mathcal V}
D\left(A_t,\widetilde A'_t\right)
}
\]

where \(\mathcal V\) contains valid tubelet times. Define the reduction over class maps and spatial support explicitly; the expression is a proposed template rather than a paper's exact objective.

Choose one fully specified distance before the first comparison. One candidate for normalized attention distributions is total variation:

\[
D(A,\widetilde A')=\tfrac12\sum_{i\in\mathcal P_{\mathrm{valid}}}|A_i-\widetilde A'_i|.
\]

Use the same valid content support for both aligned maps and normalize attention over that support. Record whether both maps receive gradients; use the same shared head and allow gradients through both in the initial symmetric version. MSE or symmetric-divergence alternatives are later ablations, not interchangeable reductions with the same coefficient.

Do not select the loss merely because its numerical value is small.

### Implementation requirements

1. Apply horizontal reflection to the original video frames before V-JEPA feature extraction.
2. Extract mirrored patch features using the same frozen backbone.
3. Run both original and mirrored patch features through the same spatial attention head.
4. Invert the spatial reflection of the predicted mirrored map.
5. Compare maps at corresponding timestamps and spatial positions.
6. Mask invalid tubelets and letterbox-only patches, and verify the inverse mapping for any asymmetric letterbox padding.
7. Average the loss consistently across frames, videos, and distributed ranks.

Do not assume flipping an already extracted V-JEPA patch grid is equivalent to re-encoding a flipped video. Use the actual mirrored video as the primary experiment.

A feature-grid flip may be implemented as a cheaper ablation, but it must be labeled as an approximation.

### Video-specific constraints

Horizontal reflection must not:

- Reverse temporal order.
- Change timestamps.
- Mix frames between videos.
- Break temporal alignment.
- Introduce invalid patch indices.
- Change the episode label.

Verify that the spill-detection label is invariant to horizontal reflection.

Do not introduce crop, erase, cutout, elastic deformation, temporal reversal or frame-local random geometry. Horizontal reflection is the geometric training transformation for this proposal. Verify that the preprocessing produces the assumed grid correspondence; account for asymmetric letterboxing rather than silently flipping a misaligned grid.

Use the same photometric augmentation parameters for paired views initially, so the first experiment isolates geometric equivariance.

---

## 4. Prevent trivial attention collapse

Mirror consistency alone does not guarantee meaningful localization.

Uniform maps satisfy mirror consistency perfectly.

Background-focused maps may also satisfy it.

Therefore, implement diagnostics for:

- Uniform attention collapse.
- Excessively concentrated single-patch attention.
- Class-independent attention.
- Background shortcut learning.
- Spatial map instability.
- Attention saturation.
- Vanishing gradients.

Measure normalized attention entropy:

\[
H_{\mathrm{norm}}(A_t)=
\frac{-\sum_{i\in\mathcal P_{\mathrm{valid}}}A_{t,i}\log(A_{t,i}+\epsilon)}{\log |\mathcal P_{\mathrm{valid}}|}
\]

Report the valid patch count and handle a single valid patch separately. Uniform attention has normalized entropy near one. Attention entropy is a diagnostic, not a spill-mask metric.

Also calculate:

- Effective number of attended patches.
- Maximum attention probability.
- Attention concentration.
- Similarity between class-conditioned maps.
- Mirror consistency error.
- Attention variability across videos.

Do not force low entropy or sparse maps without evidence that this improves localization.

Consider optional regularization only after observing the baseline behavior.

---

## 5. Joint objective

The proposed full joint-training objective is:

\[
\boxed{
L=
L_{\text{ViCo}}(z_{\text{spatial}})
+0.1L_{\text{align}}
+L_{\text{predict}}
+0.1L_{\text{keep}}
+\lambda L_{\text{mirror}}
}
\]

Preserve the original definitions of alignment, prediction, retention, and ViCo losses.

**Provenance:** `L_ViCo` is the repository's ViCo-LWIR Eq. 10 implementation. Alignment, future prediction, retention and their coefficients are the existing trajectory objective. Applying ViCo to `z_spatial` changes its representation pathway. `L_mirror` and its coefficient are proposed additions. The complete equation is **not from SEAM, MCTformer, MoRe, ToCo or ViCo-LWIR as a published combined objective**.

Keep per-positive competition, temperature 0.1, float64 semantic arithmetic, detached mining weights, matched-source positive weights and the existing curriculum. Do not substitute shared-denominator SupCon or relabel a new spatial regularizer as ViCo Eq. 10.

When the temporal branch is frozen, the actual optimized objective is:

\[
L_{\mathrm{optimized,spatial}}=L_{\mathrm{ViCo}}(z_{\mathrm{spatial}})+\lambda L_{\mathrm{mirror}}.
\]

The three temporal terms may be logged in a monitored combined total, but they are constants with respect to the spatial parameters. They must not be described as training or regularizing an independent spatial branch. Original frozen-head semantic metrics remain a separate baseline.

Inspect the existing code to determine the exact reductions, units, and gradient scales of each loss.

### Critical optimization issue

Different losses can have very different magnitudes.

Do not assume that a coefficient of 0.1 produces a small gradient contribution.

Monitor:

\[
\left\|\nabla_\theta L_k\right\|_2
\]

for each loss component on its affected trainable parameters.

For the spatial-only stage, compare ViCo and mirror gradients on spatial parameters. Temporal-loss gradients on that branch should be absent. Monitor gradient cosine similarity only where objectives actually share trainable parameters; do not manufacture a shared path to obtain that diagnostic.

If the mirror objective produces gradients that dominate the semantic objective, reduce its weight or use a controlled warmup.

### Mirror coefficient search

Treat \(\lambda\) as an experimental hyperparameter.

Start with `lambda=0`. After confirming spatial gradients and measuring the chosen distance's scale, predefine a small, bounded comparison with a nonzero coefficient. Expand the search only if the results justify it. There is no paper-backed best coefficient for this architecture; values suggested previously were illustrative candidates, not validated defaults.

Compare candidates from identical initialization with identical source/view exposures, optimizer schedules and update budgets. Do not give the mirror condition extra training by continuing a completed no-mirror run unless a matched continuation control receives the same additional budget.

Use the source-validation split for declared model-selection metrics, and report that use. Mirror error alone favors trivial solutions. With no spatial ground truth, classification, collapse diagnostics and controlled perturbations can support selection for consistency/robustness, but cannot select a demonstrated best spill localizer. Keep any final independent scene test untouched.

---

## 6. Preserve the measured temporal baseline

This is a high-priority requirement.

The original 100-update run is a fixed-budget result, not proof of full convergence. Its tuner also did not reset AdamW state at update 21: saved state counters are 100. Record this deviation instead of retroactively claiming the default reset was used. Define and record the new experiment's protocol, including any reset, consistently across its comparisons.

The existing temporal experiment must remain a reproducible baseline.

### Stage 1 — Freeze temporal branch

Load the existing trained temporal checkpoint.

Freeze its parameters and train only the new spatial branch.

Verify that:

- Temporal states are unchanged.
- Existing alignment loss is unchanged for identical inputs.
- Existing prediction outputs are unchanged.
- Existing retention metrics are unchanged.
- The original checkpoint remains loadable.

In this stage, existing temporal losses may be monitored but must not be misrepresented as optimizing frozen temporal parameters.

### Stage 2 — Train spatial branch without mirror loss

Set:

\[
\lambda=0
\]

Train the spatial semantic branch using ViCo supervision.

Establish whether class-specific spatial attention emerges without equivariance regularization.

### Stage 3 — Add mirror consistency

Train fresh matched replicas with each predefined nonzero mirror coefficient.

Compare against Stage 2.

Determine whether mirror consistency improves measured consistency or robustness. Claim localization improvement only if independent spatial evaluation becomes available.

### Stage 4 — Optional joint fine-tuning

Only if the spatial branch is stable and the initial results justify it, test joint optimization of the spatial and temporal branches.

Use separate optimizer parameter groups and conservative learning rates.

Preserve a frozen temporal control.

If the original temporal semantic head is no longer used by an optimized loss, keep that head frozen and exclude its parameters from optimization. Explicitly audit unused parameters and DDP participation; adding a spatial branch does not automatically connect the original semantic head to the new ViCo embedding.

Do not claim preservation of convergence merely because the total loss decreases.

### Regression protection

Compare the following against the original baseline:

- Alignment loss.
- Prediction loss.
- Retention loss.
- Semantic contrastive loss.
- Validation classification performance.
- Temporal representation drift.
- Gradient norms.
- Numerical stability.
- Held-out generalization.

Set quantitative acceptance tolerances using baseline variance or repeated runs rather than arbitrary thresholds.

If joint training materially degrades temporal performance, reject that configuration and retain the frozen-temporal version.

---

## 7. Run the experiment on 80 videos

Use the 80 real source videos associated with the existing experiment.

Reuse the audit's source inventory and byte-duplicate checks. Verified scene/session grouping remains unresolved; recorded source-sequence names are not physical scene labels. The contact sheet shows recurring environments across the original train/validation split.

Use **video-level or source-group-level splits**, never random frame-level splits that leak highly correlated frames between training and validation.

Preserve the 64-training/16-source-validation split for direct comparisons. The new branch fits only the 64 training sources; frozen feature extraction of all 80 is not permission to fit representations or prototypes on validation sources. Do not claim this source split is scene-held-out.

If reliable scene groups become available, create a separate scene-grouped protocol. An evaluation source must be excluded from every learned operation, including the earlier temporal checkpoint if that checkpoint is used by the evaluated decision. Rearranging already fitted sources into a new split does not create a held-out test. Prototype galleries and any feature statistics fitted from data use training sources only.

### Required experiment matrix

| Experiment | Spatial branch | Mirror loss | Temporal branch |
|---|---|---|---|
| E0 | None | None | Existing checkpoint |
| E1 | Trainable | Off | Frozen |
| E2 | Trainable | On | Frozen |
| E3 | Trainable | On | Joint fine-tuning, optional |

For E2, begin with the bounded mirror-coefficient comparison described above; do not automatically launch a broad search.

Keep training budget, data split, and evaluation procedures comparable.

### Mirror test protocol

For every validation video:

1. Compute the original spatial attention map.
2. Horizontally flip the original video.
3. Use verified cached mirrored features if available; otherwise re-extract them with the same causal-prefix policy.
4. Compute attention on the mirrored video.
5. Reverse the mirrored attention map spatially.
6. Compare both maps.
7. Record consistency metrics.
8. Save side-by-side visualization.

Produce a comparison showing:

`Original frame | Original attention | Flipped frame | Flipped attention | Unflipped attention | Difference map`

Repeat this at representative timestamps throughout each video.

Do not choose only visually successful examples.

Include failure cases.

---

## 8. Evaluate semantic learning and localization separately

### Semantic metrics

Measure:

- Held-out classification accuracy.
- Precision, recall, and F1.
- ROC-AUC and average precision where applicable.
- Intra-class and inter-class embedding similarities.
- Retrieval quality.
- t-SNE or UMAP as supplementary visualization.

### Spatial metrics

Where spill masks or region annotations exist, calculate:

- Pointing-game accuracy.
- Pixel average precision.
- Threshold-swept IoU.
- Attribution mass inside the spill region.
- Background attention mass.

Where ground-truth masks do not exist, explicitly mark localization results as qualitative or proxy-based.

There are currently no masks, so do not report IoU, pixel AP, inside-spill mass or pointing accuracy without obtaining the required verified annotations. A clean/no-spill video is not required to contain a highlighted foreground region. Separately report cold and clean behavior; warm-spill results are unavailable for this 80-video dataset.

Include input-level removal controls that re-encode every causal prefix after equal-area high-, low- and seeded-random content-patch removal. Exclude padding from all selections. Mark distribution shift and limited sample size; cached-token deletion is not a substitute for an input-level control and neither test supplies a ground-truth spill boundary.

Inspect associations between map responses and token norm because the earlier audit found substantial correlations. Do not suppress high-norm tokens or select favorable heads merely to improve appearance. Display all heads with documented aggregation, a scale shared across comparable samples/times, and no per-frame display renormalization.

### Equivariance metrics

Calculate:

\[
E_{\text{mirror}}=
\left\|
A(x)-F^{-1}(A(F(x)))
\right\|
\]

Report the metric definition, normalization, mean, variance, and per-video distribution.

Also compare original and mirrored classification predictions.

A mirror-consistent map is not necessarily a correct spill-localization map.

---

## 9. Multi-GPU distributed training

Preserve the existing DDP architecture.

Requirements:

- Every GPU receives distinct assigned training samples, subject to explicitly handled sampler padding.
- Use global rank and world size correctly.
- Synchronize gradients across all participating ranks.
- Calculate global effective batch size correctly.
- Aggregate validation statistics across all GPUs.
- Ensure paired original/mirrored samples remain aligned.
- Avoid repeated V-JEPA extraction where a valid feature cache can be reused.
- Cache original and mirrored features separately when necessary.
- Preserve exact frame ordering and source-video identifiers.

Only global rank 0 writes consolidated checkpoints, visualizations, and experiment reports.

For large rank-local attribution outputs, use unique temporary files followed by a synchronized rank-0 merge.

Ensure no rank enters a collective operation that other ranks skip.

---

## 10. Engineering and testing

Implement tests for:

**Spatial geometry:** A 16×16 map must reverse correctly under horizontal flipping.

**Double reflection:** Applying the horizontal flip twice must recover the original map.

**Gradient flow:** The spatial head receives gradients from ViCo and mirror losses.

**Frozen backbone:** V-JEPA parameters remain unchanged.

**Temporal regression:** The frozen temporal branch produces identical outputs before and after adding the spatial branch.

**Loss independence:** Setting \(\lambda=0\) removes mirror-loss gradients.

**Batch correctness:** Batched mirror loss agrees with per-video calculations.

**Padding correctness:** Invalid frames do not contribute to the objective.

**Distributed correctness:** Global metrics match a single-process reference on a deterministic test dataset.

**Numerical stability:** No NaN or Inf values in attention, losses, gradients, or embeddings.

**Checkpoint compatibility:** Existing checkpoints remain usable without migration.

Implement a small synthetic test first, then a short real-video smoke test, before launching the full 80-video experiment.

---

## 11. Deliverables

Produce working code and experimental artifacts.

The subsequent execution request authorized this deliverable list. See the
execution results linked at the beginning of this document for completed
outputs and the scientific decision against promoting a localization model.

Required outputs:

1. Spatial semantic branch with class-conditioned patch attention.
2. Mirror-equivariant attention loss.
3. Original/mirrored V-JEPA feature extraction.
4. Combined training objective.
5. Regression tests protecting the temporal branch.
6. Distributed training compatibility.
7. Quantitative evaluation on the 80 videos.
8. Original-versus-mirrored attention visualizations.
9. Ablation report comparing E0, E1, E2, and optional E3.
10. Recommended configuration supported by experimental evidence.

Save all new results under a new, uniquely named experiment directory.

Never overwrite the original experiment.

## 12. Final research decision

Answer these questions using experimental evidence:

1. Does the spatial branch improve held-out spill discrimination?
2. Does it produce spatially meaningful attention maps?
3. Does mirror equivariance improve spatial consistency?
4. Does improved consistency correspond to improved spill localization?
5. Does the new branch preserve the original temporal model's performance?
6. Is freezing V-JEPA sufficient, or is selective adaptation justified?
7. Which mirror coefficient best meets the declared consistency/robustness criteria under the available budget, and is there sufficient spatial evidence to rank localization at all?

Provide a final comparison table, failure-case analysis, source/configuration hashes, and the exact checkpoint for any supported recommendation. Retain the original baseline if no new configuration shows a supported benefit. Distinguish measured results, mathematical implications and unresolved hypotheses.

**Success criteria:** A consistency result requires improved held-out mirror consistency without collapse or material degradation of semantic/temporal metrics. A localization result additionally requires independent spatial evidence. These are separate outcomes. The current mask-free dataset may support the former while leaving the latter unresolved.

Do not declare success based only on lower training loss, better t-SNE separation, or visually appealing attention maps.

**Execution constraint:** Reuse the completed audit, validate correctness and temporal regression, then run the bounded comparison using the original 64/16 split. Preserve original experiment artifacts and create a new output directory for every changed dataset, loss, augmentation, seed or trainable parameter set. The authorized bounded comparison has now been run. There is no currently validated best decoder-free localization configuration for these videos.
