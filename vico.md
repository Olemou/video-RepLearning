# Research Task: All-Patch ViCo for V-JEPA — No MIL

## 1. Current request and scope

Update this document first using the completed experiments and a direct reading of `loss_doc/VicoLWIR.pdf`. This document update does not launch another experiment or modify implementation files.

The user's intended next investigation is **all-patch ViCo**, preserving the original-image identity used by the paper's pair weighting:

- Use every valid spatial patch of each selected image/tubelet.
- Give distinct positive instances from the same original image weight **1**.
- Weight positive instances from different original images of the same class using the uncertainty curriculum.
- Apply hard-negative weighting to comparisons between different image classes.
- **Do not use MIL, attention-weighted bag pooling, top-k evidence pooling, or log-sum-exp bag classification as the proposed solution.**

The experimental question is whether direct supervision of retained patch representations improves spatial discrimination compared with supervision only after global pooling. This is a hypothesis, not an established localization improvement.

## 2. What the actual ViCo-LWIR paper says

Primary source: [`loss_doc/VicoLWIR.pdf`](loss_doc/VicoLWIR.pdf).

### Patch representations and backbone training

Section IV-A.2, printed page 7, explicitly states:

> Fine-tuning was performed using patch-level representations

The same paragraph reports:

- A pretrained **ViT-B/16**, rather than V-JEPA 2.
- An MLP projector with output dimension **256**.
- AdamW learning rates **9e-5 for the projector** and **1e-4 for the backbone**.
- Batch size **64** and contrastive temperature **0.1**.
- Backbone freezing **after** representation training, for downstream classification.

Therefore, the paper must not be summarized as training only a global projector on permanently frozen backbone features. It explicitly describes patch representations and backbone fine-tuning. Our frozen V-JEPA experiment differs from that training pathway.

### Same original image versus different same-class images

Section III-B.2, Eq. 6, printed page 5, defines `orig(i)` as the original, non-augmented instance from which an augmented instance derives. For distinct instances:

\[
w_{ij}=\begin{cases}
1,&\operatorname{orig}(i)=\operatorname{orig}(j),\ y_i=y_j,\ i\ne j,\\
\theta_{ij}(u_{ij},e),&\operatorname{orig}(i)\ne\operatorname{orig}(j),\ y_i=y_j,\ i\ne j,\\
0,&\text{otherwise}.
\end{cases}
\]

This is the weighting distinction the user wants preserved when applying ViCo to all patch instances. Same-image positives must not accidentally receive cross-image uncertainty weights. A pair with weight one still participates in the contrastive loss; it is not removed or detached from embedding learning.

### What is specified, and what needs implementation verification

The paper explicitly states patch-level representation training and the original-image weighting rule. Its equations use sample indices, image-level class labels, and a matrix `[N,d]`; they do not write an explicit `[image,view,time,patch]` loss tensor or fully describe its patch reduction. Section III-B mentions embedding width 768, whereas IV-A.2 reports a 256-dimensional projector.

Use the user's clarification—**retain all patches and preserve image provenance**—as the intended next formulation. Before calling an implementation an exact reproduction, verify the author's executable patch indexing, projector placement, aggregation/reduction, and embedding dimension if that code is available. Do not infer that the absence of explicit patch indices means the paper did not use patch features. Conversely, do not invent a patch-selection mechanism and attribute it to the paper.

The abstract describes image-level weak supervision and primarily image-level evaluation. Figure 7, printed page 11, presents final-layer attention and rollout qualitatively. These figures are not quantitative spill-mask validation.

## 3. Completed evidence that the next experiment must incorporate

Original baseline:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021`

Original diagnosis:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/research_diagnosis/report.md`

Completed spatial-branch report:

`reports/experiments/spatial_semantic_mirror_warmup_80_sources_20261021/report.md`

| Existing variant | Source-validation classification | Mirror TV | Normalized map entropy | Clean/spill query-map cosine |
|---|---:|---:|---:|---:|
| E0: original temporal checkpoint | 16/16 | Not applicable | Not applicable | Not applicable |
| E1: spatial branch, global ViCo, no mirror | 16/16 | 0.203394 | 0.971721 | 0.999958 |
| E2: spatial branch, global ViCo, mirror warmup | 16/16 | 0.000115497 | 1.0 | 1.0 |

The first guarded mirror schedule also failed: its final TV was 0.438966 and its coefficient decayed to approximately 4.92e-10. Preserve that result rather than reporting only the later schedule.

The warmup variant's effective attended area is 192 patches, with maximum probability about 0.00521181. Uniform attention over the 192 content patches has probability `1/192 = 0.00520833`. Thus its excellent mirror consistency is an essentially uniform solution, not learned spill localization.

Four-video input-removal controls did not validate high-attention patches as more important than random patches. All temporal parameter/state/future-output drift checks were exactly zero. The previous full test suite completed with **151 passed, 4 skipped**; these are results for the existing implementation, not tests of the proposed all-patch extension.

Do not repeat the failed generic attention/mirror experiment under a new name. The new intervention must move the contrastive objective **before spatial averaging**, onto the retained patch representations.

## 4. Data, split and causal feature constraints

- Dataset: **80 byte-unique videos**, 40 clean and 40 cold; no warm videos in this run.
- Labels: `clean -> 0`, `cold/warm -> 1` generally.
- Preserve the original **64 training / 16 source-validation** split.
- The 16-source validation set has already informed model selection. It is not an independent final test.
- Physical environments recur across the split; verified scene groups are unavailable. Do not call it scene-held-out.
- No verified spill masks are available. Do not report IoU, pixel AP, pointing accuracy, spill-region mass or background-region mass without annotations.

Backbone: `facebook/vjepa2-vitl-fpc64-256`, loaded strictly with `load_pretrained_encoder()`.

Preserve official normalization, full-field-of-view 256x256 letterboxing and `past_only_prefix_last_tubelet_v1`. Encode each observed prefix separately and retain its final tubelet. Do not substitute full-video tokens for causal-prefix features.

The accepted existing cache is:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/research_diagnosis/spatial_matched_precision/spatial_tokens.pt`

Shape: `[80,2,6,4,256,1024]`, representing source, actual identity/mirror view, observed prefix, selected block, spatial patch and feature channel. Blocks are 5, 11, 17 and 23. Its final-layer means reproduce the original cached temporal features exactly.

Do not use the older `research_diagnosis/spatial/` cache with the precision mismatch.

For a selected layer:

\[
P\in\mathbb R^{B\times V\times T\times N\times1024},\qquad
Q=\operatorname{Normalize}(g_\theta(P))\in\mathbb R^{B\times V\times T\times N\times d}.
\]

Here `V=2`, `N=256` on the native 16x16 grid. The current 4:3 videos have **192 content patches** after letterbox masking. Use every valid content patch; padding is not a target patch. Record the policy for partially covered patches. Repeated tail tubelets are invalid.

`[B,N,d]` is only shorthand for one selected observed time/view, not permission to discard time or provenance. Do not average moving image content across time before the patch loss.

## 5. Define patch provenance before constructing the loss

Every patch instance `u` must retain:

`(video_id, original_image_id, view_id, tubelet_time, patch_row, patch_col, valid, image_label)`.

Distinguish these identifiers:

- `video_id` controls splitting, source leakage and exposure accounting.
- `original_image_id` controls Eq. 6's same-image weight-one rule.
- `view_id` identifies identity or a genuine re-encoded mirror.
- Spatial coordinates identify patches, including the inverse mirror mapping.

For this video adaptation, define an original image/tubelet unit explicitly, for example `(video_id, observed_prefix_index)` for the cached last-tubelet representation. Its two real views share that original-image identity. Different prefix times must not silently become the same original image merely because they share a video.

Under the user's intended all-patch formulation, distinct patches belonging to that original-image unit—including its paired views—have same-image positive weight one. This is an image-provenance relation, not a claim that all those patches contain the same physical object. A coordinate-matched mirror pair is a known correspondence; arbitrary different positions are not physical correspondences just because their weights are one.

The current video-level code's `source_ids` contract cannot be reused blindly: it currently means original video. Patch loss needs its own explicit original-image identifiers and tests.

## 6. All-patch ViCo formulation to investigate

The original whole-episode implementation is:

`src/vision_model/training/vico.py`.

Preserve it unchanged. Implement any patch extension as a separately named objective with an explicit relation graph.

For normalized patch embeddings, let `s_uv = q_u^T q_v`, and define:

\[
\mathcal P(u)=\{v\ne u:y_v=y_u\},\qquad
\mathcal N(u)=\{k:y_k\ne y_u\}.
\]

For this proposed all-patch experiment, `y_u` denotes the **originating image's weak class label**, not a verified patch annotation. Every valid patch participates. Do not covertly replace the requested all-patch experiment with selected confident patches or learned region pooling.

Set positive weights according to original-image provenance:

\[
w_{uv}=\begin{cases}
1,&\operatorname{image}(u)=\operatorname{image}(v),\\
\theta_{uv}(u_{uv},e),&\operatorname{image}(u)\ne\operatorname{image}(v),\ y_u=y_v.
\end{cases}
\]

Compute cohesion, non-cohesion, uncertainty and ranks using a clearly recorded comparison population, following Eqs. 2, 5 and 8. Same-image weights remain one regardless of those ranks. Mining weights and rank statistics are detached; patch embeddings remain differentiable.

For negatives, preserve Eq. 7's square-root count factor:

\[
\alpha_{uk}=
\sqrt{|\mathcal N(u)|}\,
\frac{\exp(s_{uk}/\tau)}
{\sum_{r\in\mathcal N(u)}\exp(s_{ur}/\tau)+\epsilon},\quad k\in\mathcal N(u).
\]

The per-positive competition is:

\[
\ell_{uv}=
\operatorname{softplus}\!\left[
\log\!\left(\sum_{k\in\mathcal N(u)}\alpha_{uk}\exp(s_{uk}/\tau)+\epsilon\right)
-\left(s_{uv}/\tau+\log w_{uv}\right)
\right].
\]

Average over valid positives for each anchor, then reduce anchors with a declared image/video-balanced policy. If examples have unequal valid times or patch counts, a flat patch average and a source-balanced average are different objectives; test and record the chosen reduction. With equal counts they should agree.

This retains ViCo's per-positive-versus-negative competition. Do not substitute shared-denominator SupCon, subtract an entropy constant, or silently divide cosine by temperature twice. Start with the existing `tau=0.1`, `epsilon=1e-6` and float64 semantic arithmetic, and verify numerical/memory feasibility.

The PDF's Eq. 7 includes a square-root count factor absent from its Algorithm 1 pseudocode. The repository follows Eq. 7. Preserve and document that choice. Verify the implementation of Eq. 8 against the actual paper/code rather than silently changing the established curriculum.

### Weak-label risk that the experiment must measure

A spill-positive image includes background. Giving all its patches the image label is a **weak supervisory assumption**, not ground truth about each patch. Weight one for same-image pairs does not resolve that ambiguity. Cross-class negatives can also repel physically similar background patches from clean and spill-positive images.

The primary requested experiment deliberately tests this all-patch assumption. Document its possible failure rather than assuming either success or failure. In particular, test whether patch embeddings become class-separated but spatially uniform within each image. That would reproduce the semantic/localization disconnect at patch level.

Do not describe this protocol as using verified patch labels. Do not infer that the paper supplied pixel masks merely because it used patch representations.

## 7. Architecture and losses: preserve the temporal branch

The new pathway should be direct and decoder-free:

```text
frozen causal V-JEPA patch tokens [B,V,T,256,1024]
  -> shared pointwise trainable patch projector
  -> normalized patch embeddings [B,V,T,256,d]
  -> direct all-patch ViCo comparisons with original-image provenance
```

No MIL head, attention-selected bag representation, top-k bag classifier or segmentation decoder is part of this experiment. Do not start with the previously collapsed two-query pooling head.

Keep the existing trajectory branch, global teacher means, causal masking, future targets and checkpoint unchanged. Retain the established whole-episode objective as the baseline:

\[
L_{\rm original}=L_{\rm ViCo}(z_{\rm episode})
+0.1L_{\rm align}+L_{\rm predict}+0.1L_{\rm keep}.
\]

For a frozen temporal branch and a separate new patch projector, the optimized new objective is initially:

\[
L_{\rm optimized,patch}=L_{\rm ViCo,patch}(Q).
\]

The original episode ViCo, alignment, prediction and retention losses are then frozen monitoring values, not gradients training that independent patch projector.

An optional later comparison may train a new **fixed-mean** episode readout of `Q` with original ViCo, alongside patch ViCo. This is ordinary global pooling, not a learned instance-selection mechanism. It must receive a matched global-only control, an explicit gradient audit and a separate output directory. Do not claim that detached temporal losses regularize it.

## 8. Mirror handling and visualization

Use actual mirrored video encodes already present in the accepted cache. Do not pretend a post-encoding grid flip is equivalent to re-encoding a mirrored input.

A separate mirror attention loss is **off initially**. The previous experiment measured uniform-map collapse, so it is not a justified default. Identity/mirror instances may still participate in the all-patch objective using their correct image provenance.

Record token correspondence after inverse reflection for evaluation. Spatial correspondence and class-positive pair membership are different relations.

Do not visualize patch embeddings as a spill probability automatically. Define a class-related scalar or classifier in the same embedding space, fit all prototypes/readouts using training sources only, and specify how each map relates to that decision. If a patch-to-class similarity is used, call it a similarity diagnostic unless its decision connection and calibration are established.

Use all heads where heads exist, shared scales across comparable sources/times and no per-frame display normalization. Compare layers 5, 11, 17 and 23 under identical targets. Do not select a favorable head, manual spill ROI or visually successful subset.

Cached features support gradients through a new projector, but not end-to-end gradients into V-JEPA. Any later backbone-gradient attribution or fine-tuning requires a separate differentiable image-to-encoder forward pass.

## 9. Compute and distributed feasibility before training

All-patch training must not be implemented by passing an enormous flattened array to the current small-batch function unchanged.

For one view/time, `B=32, N=256` produces 8,192 patch instances and 67,108,864 similarities: **256 MiB float32 / 512 MiB float64** for one matrix alone.

With 64 sources, two views, six times and 192 content patches, there are 147,456 patch instances. One full similarity matrix alone is **81 GiB float32 / 162 GiB float64**, before gradients, ranks and other intermediates.

The current uncertainty implementation constructs a three-dimensional tensor for rank comparisons. Do not use that cubic-memory routine on patch populations. Implement rowwise sorting with correct tie handling, chunked comparison statistics, and exact streamed/recomputed reductions, then test against the small original reference.

A bounded first protocol may deterministically schedule one valid tubelet per source per update while using **all spatial patches** of that selected tubelet. Preserve the source split, disclose the changed temporal exposure, and use identical schedules in matched controls. This is temporal scheduling, not dropping confident/unconfident spatial patches. Do not silently change the protocol to fit memory.

Use the original 64-source effective batch where feasible; never silently reduce it below 32. Image/video and patch batch sizes are different quantities.

Each GPU must process distinct assigned sources and contribute gradients. Gather trainable embeddings with valid gradient paths, preserve image/source IDs, avoid sampler duplicates, reduce variable-count losses correctly under DDP averaging, aggregate metrics across ranks, and let only global rank zero write consolidated artifacts. Verify unequal-shard distributed gradients against a single-process reference.

## 10. Controlled next experiments

Reuse completed audits rather than repeating expensive feature extraction. Preserve all original artifacts and create a new output directory for every changed configuration.

| ID | Experiment | Purpose |
|---|---|---|
| P0 | Existing E0 and saved E1/E2, evaluated as historical controls | Establish original recognition, pooling and collapse findings |
| P1 | Small synthetic direct-patch ViCo, all patches with original-image IDs | Verify weights, reductions, memory, geometry and gradients |
| P2 | Frozen V-JEPA, pointwise projector, direct all-patch ViCo | Test the user's requested patch-level supervision pathway without MIL |
| P3 | Matched image-weight ablation: correct same-image weight one versus deliberately uniform positive weights | Isolate the uncertainty curriculum's contribution without changing data exposure |
| P4, conditional | Intermediate versus final-layer patch representations under the same direct-patch objective | Test where useful spatial information survives |
| P5, optional | Selective final-block/adaptor fine-tuning after frozen-feature evidence justifies it | Test the paper/current-pipeline backbone-training difference |

P3's uniform-positive control is an intentional ablation, not a replacement for the requested main formulation. Do not automatically launch all rows or full-backbone fine-tuning. Start with P1 and a short real-data smoke test; continue only after correctness and cost checks.

The first frozen-feature comparison should use the established seed 239, AdamW settings and explicit update/exposure budget where applicable. Record any deviation from the old run. The original 100-update tuner did not reset AdamW at update 21; the later spatial experiments did. Apply one recorded protocol consistently to each new matched comparison.

Do not claim fixed-budget training is fully converged merely because semantic loss approaches zero. Repeat promising results with additional seeds before stronger recommendations.

## 11. Evaluation and acceptance criteria

Report class discrimination and spatial evidence separately.

**Semantic:** source-validation accuracy, precision/recall/F1, ROC-AUC/AP, retrieval, minimum same-class cosine, maximum cross-class cosine, minimum per-anchor separation, worst positive-pair loss, and fixed-curriculum curves. t-SNE is supplementary and must identify which sources fit the projection.

**Patch diagnostics:** within-image feature variance, effective rank, within-image patch cosine, class separation, mirror correspondence retrieval, spatial diversity, token-norm associations, class-target sensitivity and cross-view stability. A constant embedding per class may score well on classification and still fail spatial discrimination. Arbitrary spatial diversity can also be irrelevant to spills.

**Input controls:** for predetermined clean/cold validation examples, compare equal-area high-, low- and seeded-random patch replacement at the input, re-encoding every prefix. These are evaluation-only distribution-shift controls, not training augmentation or ground-truth spill boundaries. Preserve the supported preprocessing and augmentation restrictions.

**Localization:** without verified spatial annotations, mark results qualitative/proxy-based. Do not produce invented IoU, pointing accuracy or pixel AP. Clean videos need not highlight a spill; warm-spill metrics are unavailable here.

**Temporal protection:** parameter, state, semantic embedding and horizon-1/2/4 prediction drift must remain zero in the frozen stage. Preserve original alignment/prediction/retention values and persistence comparison. No original checkpoint migration or rewriting.

A lower patch loss, sharper map, improved correspondence or perfect classification alone does not establish spill localization. Report whether direct patch training improved a measured spatial proxy, whether it remained class-discriminative, and what the evidence cannot resolve.

## 12. Required tests and report.md for later execution

Before full experiments, test:

1. Same-original-image positive weights are exactly one, including across actual views.
2. Different-image, same-class positives use the uncertainty curriculum; different prefix times do not accidentally share image IDs.
3. Self pairs and padding are excluded; all valid content patches participate.
4. Dense and chunked losses/gradients agree on a small deterministic example, including ranks/ties and variable valid counts.
5. No false claim that image-level weak labels are verified spill-patch labels.
6. No NaN/Inf, zero-positive division, detached projector path or unintended trainable backbone parameter.
7. DDP results match a single-process reference with unequal shards and no duplicate evaluation sources.
8. Temporal checkpoint compatibility and exact frozen output regression.

Run focused tests and the required full suite using the available environment. `.venv/bin/python` is absent; the previous runs used `/home/olemou/miniconda3/envs/torchgpu/bin/python` with the matching library path. Record environment differences and skipped tests.

Write the new executed experiment's report to:

`reports/experiments/<new_experiment_id>/report.md`.

Include paper evidence with page/equation references, user-directed all-patch indexing, exact formulas and reductions, weak-label risks, baseline/failure comparisons, tensor and gradient audits, distributed and memory checks, split/exposure schedules, checkpoint/source/cache hashes, measured metrics and figures, failure cases, and an evidence-based decision. State explicitly which outcomes improve and which remain unproven. Never invent results or label a proposed configuration as already successful.

**Final question:** Does preserving all patches and applying ViCo directly—with same-image weight one and uncertainty-weighted different-image positives—learn spatially meaningful spill evidence, or merely reproduce image-class separation independently at every patch? Resolve that experimentally, without MIL and without presenting qualitative attention as verified localization.
