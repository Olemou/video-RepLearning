# Research Task — Learn Spill-Focused Spatial Evidence and Test Final V-JEPA Attention Fine-Tuning

## 1. Current scope and research priority

Latest implementation: the [finite-field local study](reports/experiments/spill_local_relative_80_20261022/report.md)
tests the [scene-confounding proposal](reports/experiments/spill_scene_confound_audit_20261022/report.md).
The new separate local detector uses relative thermal measurements, same-image
component/ring ranking and clean-absence supervision. There is no global
contextual-token bypass, manual annotation, MIL or decoder. Both pointwise and
bounded local convolutional heads were evaluated on the original64/16 split,
at matched100/1000-update budgets on two H100s. The original four trajectory
losses and teacher/checkpoint remain unchanged.

Do not claim the localization requirement is complete: inspected cold56,
cold10 and cold13 still contain missed liquid and wrong-region responses.
Exact distant-context invariance and lower pseudo-pair loss do not identify
liquid. FP16-boundary and evaluation-batch-shape replay were fixed without
relaxing tolerances; successful input checks have zero baseline replay error.
Latest full regression: **236 passed, 4 skipped**. The [commands and limits](docs/local-thermal.md)
describe this opt-in branch; frozen last-block attention has not become a
learned liquid mask.

The proposed dense path is now implemented as an opt-in experiment; see
[commands and exact objectives](docs/dense-spill.md) and the new run directory
`reports/experiments/spill_dense_fix_80_20261022`. The strict frozen V-JEPA 2.1
ViT-L encoder matches the upstream FP32 encoder exactly on two H100s at both
256 and 384 resolution. Both 80-source matched-resolution caches include extra
windows only from training-clean sources. The spatial head uses a fixed valid
patch mean and training-only prototypes; its signed contributions reconstruct
its actual decision. The local loss balances sources and requires FG/BG in the
same positive frame. The historical automatic path now rejects insufficient
within-spill support instead of accepting clean-video negatives as sufficient.

The [completed implementation report](reports/experiments/spill_dense_fix_80_20261022/report.md)
records seven matched 100-update readouts on real two-H100 allocations. Pure
episode ViCo gives 16/16 selection recognition but diffuse positive evidence;
within-frame novelty supervision gives 14/16 and stronger physical-input
sensitivity, while wrong-region activations remain. A cold-pixel teacher and
an explicit local cold-contrast input also give 14/16. Preserving patch feature
amplitudes instead of unit-normalizing every patch improves this to 15/16,
but inspected cold10/cold13 maps still fail the required liquid focus. These
are experimental feature contributions, not verified liquid masks. No variant
is promoted as the solved spill localizer. The original four-loss checkpoint stays
unchanged. No manual annotations, decoder or MIL were introduced. The final
full regression after the implementation is **228 passed, 4 skipped**; the
CUDA-only attribution regression separately passed on H100.

The [22 October deeper readout audit](reports/experiments/spill_focus_deep_audit_20261022/report.md)
adds a specific explanation for the annotation-free failure: the 32 training
cold sources have 178 foreground candidate cells but only **one** background
candidate cell inside a cold source (identity view). Cold selection sources have
zero background candidates. Thus partial BCE almost exclusively contrasts clean
scenes with sparse cold foreground, leaving cold background unconstrained.
A frozen-probe global-mean broadcast marks 97.92% of cold selection content
positive and obtains lower candidate-validation BCE than the patch field.
Final letterbox-position tokens alone also retain 16/16 original recognition;
these tokens already contain whole-scene context, not just blank border pixels.
That read-only audit proposed the frozen V-JEPA 2.1 comparison; the implementation
report above now supersedes its unexecuted status. Matched-resolution extraction,
richer training-clean references and within-positive coverage gates were tested.
They repair concrete graph/supervision failures but do not establish liquid
identity. The original four-loss model, checkpoints and figures remain unchanged.
The read-only audit itself took zero optimizer updates; the new implementation
study took seven separate 100-update dense-readout fits.

**Latest instruction: no manual spatial annotations.** The completed
[annotation-free two-H100 study](reports/experiments/spill_annotation_free_80_20261021/report.md)
supersedes the annotation-readiness pause below. It evaluated 16 normal-reference
feature configurations on all 80 sources; two 100-update normal-only readouts;
a 100-update automatic-candidate probe; and matched 100-update shared-attention
trajectory-only/joint pilots. No manual masks or points, MIL or decoder were used.
The original four trajectory coefficients and immutable future teachers were kept.
The supplied original/final-layer example specifies concentrated response on the
visible target. For this task the highest evidence must follow the actual liquid,
with lower response outside it; concentrating on a person or scene shortcut does
not satisfy this requirement. The example is not a source-aligned training label.
Every learned fit excluded the 16 selection sources; all reference banks used
training clean videos only.

Measured conclusion: the automatic local head can fit its candidates while
marking roughly 73% of cold-image content positive. Joint and trajectory-only
decision maps remain almost identical. Recognition is 16/16 and prediction
remains below persistence, but the desired liquid focus is not established.
Early-feature normal novelty performs better than the normal-only learned
projection in actual input controls. These are candidate anomaly maps, not
verified liquid masks or original attention. New curves, t-SNE, 80-video
comparisons and all-16 source-validation input-region audits are linked in the
report. Full regression: **213 passed, 4 skipped**. The earlier point-supervised
path is optional historical work, not a requirement for the current task.

Earlier, the user approved implementation of the [dual-loss audit proposal](reports/experiments/spill_attention_dual_loss_audit_20261021/report.md).
The opt-in [verified FG/BG trajectory path](docs/spill-localization.md) is implemented:
the original four trajectory terms plus an explicitly weighted partial local
FG/BG loss, with a shared student patch path and immutable detached teacher
targets. It includes frozen intermediate/final feature probes, a selective
final-attention joint pilot, strict verified-point loading, DDP checks, and
attribution to the actual joint trajectory decision. No MIL, decoder, mirror
term or all-positive-video patch labelling is used in this path. This supersedes
the audit's historical implementation pause; it does not supersede its evidence.

No verified liquid/background points were available for that earlier path. Its
point-supervised foreground/background fitting was blocked by annotation readiness;
the supplied empty template is not ground truth. See the [implementation and
verification report](reports/experiments/spill_fg_bg_implementation_20261021/report.md).
No new trained 80-video localizer or successful spill-focused map is claimed.
The earlier experiments and their failure findings below remain historical.
Implementation verification: **204 passed, 4 skipped**; the focused checks
include an end-to-end two-process probe/joint run and distributed gradients
against a global reference. A two-source real-cache smoke check confirms that
the four original losses now reach shared final attention while teacher targets
and original trajectory parameters stay frozen. No optimizer update was taken
in that real-cache check.

The final-attention protocol has now been implemented and evaluated at the user's request. The completed research report is [final_attention_80_sources_20261021/report.md](reports/experiments/final_attention_80_sources_20261021/report.md). The original documentation-only restriction was superseded by the later request to implement, test and fix the pipeline.

Completed status: frozen FT0 versus final-attention FT1, 100 matched continuation updates, 64 training and 16 source-validation videos, seed 239. Attention learning rates 3e-6 and 1e-4 both retain 16/16 classification but **do not demonstrate the required liquid focus**. The stronger setting changes Q/K weights by about 3.23%/2.85% in relative L2, yet conditional gradient maps remain almost identical to the frozen control and are highly similar after projector randomization. Do not promote either checkpoint as a spill localizer.

The latest requested fix concerns overlay logic. Global contextual class votes are now excluded from localization overlays; zero relevance is transparent; opacity follows the raw evidence magnitude on a shared scale; native cells are preserved without blur. See the [corrected 80-source gallery](reports/experiments/final_attention_80_sources_20261021/attention_lr1em4_100/figures_v2/index.html) and [input-intervention gallery](reports/experiments/final_attention_80_sources_20261021/input_occlusion_figures_v2/index.html). Old figures/checkpoints are preserved. Rendering correctness is separate from a learned spill-localization capability.

The [corrected 80-video MP4 export](reports/experiments/test_80_fixed_20261021/index.html) contains 960 actual sampled frames. Its audit found zero pixel error across the 960 compared figure-overlay panels and verified all encoded frame counts. Resize the source RGB **before** applying the overlay; resizing a colored overlay afterward would reintroduce diffusion across native cell boundaries. The decoded-video regression now tests this explicitly. All original figures and model scores remain unchanged.

The latest [requirements review](reports/experiments/test_80_fixed_20261021/report.md) rechecks all 160 source/model attribution sets and adds training/validation classification curves, separate validation t-SNE and original temporal curves. The updated MP4 gallery offers an explicit blue-to-red palette with raw heatmaps and automatic numerical peak markers. Low sensitivity is blue; blue is not a verified background label. Raw scores, checkpoints and the original magnitude-opacity videos are preserved.

The deeper numerical audit finds mean time-aggregated spatial gradient entropy **0.99729** in the trained cold videos (1 is uniform), and spill/clean target-reversal map cosine **0.96818**. The last-frame maximum in `cold_folderImage10` is at native cell `[9,1]`, source center `[60,300]`, away from the visible liquid. The required spill focus remains unachieved; changing the palette does not train it.

Full regression after these corrections: **185 passed, 4 skipped**, with one pre-existing warning. The remaining protocol below records the tested design and conditions for future work; proposed mirror/loss extensions remain unexecuted.

**The primary objective is to identify the actual liquid region in spill images, not merely distinguish spill-labelled videos.**

The user's supplied heatmap illustrates the desired appearance: concentrated high relevance over the spill, with lower relevance over unrelated image content. Use the corresponding original frame to judge whether a highlighted region really is liquid. The example heatmap alone does not define a verified spill mask or a template to copy into model outputs.

Required interpretation of success:

- When a spill is visible, class-related spatial evidence should concentrate on the liquid rather than uniformly covering the scene, person, furniture, borders or acquisition artifacts.
- Good t-SNE separation, near-zero contrastive loss, converged curves and correct video classification **do not satisfy the localization objective**.
- Sharper, more diverse or more mirror-consistent maps also do not establish correct spill focus.
- Spill-focused visual evidence must be checked across representative and failure cases, using the matching source frames and timestamps. Strengthen it with matched-area input controls and independent spatial evaluation where annotations exist.
- A clean video is not required to highlight a foreground spill. A spill-positive episode can also include frames before the spill is visible; do not require a liquid region in those frames without verifying its presence.

**No MIL, learned instance-selection pooling or segmentation decoder.** Keep the original temporal branch reproducible. Do not recolor, shift, crop, sharpen, threshold selectively or choose favorable heads to manufacture agreement with the example.

## 2. Completed evidence and exact artifacts

Repository: `/home/olemou/video-RepLearning`.

Original temporal baseline:

`reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021`

Completed direct-patch investigation:

[`reports/experiments/all_patch_vico_80_sources_20261021/report.md`](reports/experiments/all_patch_vico_80_sources_20261021/report.md)

Historical contextual-contribution gallery:

[`reports/experiments/test_80/index.html`](reports/experiments/test_80/index.html)

The `test_80` request was explicitly clarified to mean the **completed 80-video experiment**, not an 80-update checkpoint. It contains:

- **80 comparison MP4s**, 80 tubelet storyboards and 80 thumbnails.
- 64 training and 16 source-validation videos, labelled separately.
- Final **100-update** checkpoints for direct all-patch ViCo and local retention with weight 1.
- Exact checkpoint hashes, source hashes, sampled frame indices, timestamps and replay checks in `test_80/manifest.json`.

The clips visualize the cached 12-frame sampled window, with six two-frame tubelets. They are not full-source-recording localization evaluations. Each tubelet map is displayed on its two sampled frames; it is an episode-target contribution, not an online pixel prediction for each individual frame.

| Completed model | Source-validation accuracy | Within-image patch cosine | Feature variance | Mirror patch correspondence |
|---|---:|---:|---:|---:|
| Frozen raw V-JEPA | 16/16 | 0.366867 | 0.629836 | 27.327% |
| Untrained pointwise projector | 13/16 | 0.430858 | 0.566178 | 25.315% |
| Fixed-mean episode ViCo control | 16/16 | 0.910578 | 0.088956 | 22.738% |
| Direct all-patch ViCo | 16/16 | 0.951685 | 0.048063 | 22.157% |
| Direct patch ViCo + local retention 0.1 | 16/16 | 0.945394 | 0.054322 | 22.130% |
| Direct patch ViCo + local retention 1 | 16/16 | 0.754891 | 0.243833 | 23.513% |

Local retention restores **5.073 times** the direct-patch feature variance and improves correspondence by **1.356 percentage points**. These are partial representation improvements. They do not establish spill localization, and correspondence remains below the raw/untrained feature baselines.

In the all-16 input-removal evaluation, removing the highest-contribution cells did not show a supported advantage over three matched seeded random removals. Their paired-source intervals include zero. Current spill-image maps remain broadly positive over the scene; they do not demonstrate the compact spill-focused behavior requested by the user.

The previous mirror-warmup experiment reached essentially uniform attention despite 16/16 classification. Its entropy and clean/spill map cosine reached 1. Mirror consistency is therefore **not a justified default localization loss**.

Historical direct-patch regression result: **167 passed, 4 skipped**, with one pre-existing warning. Original temporal parameters, states, embeddings and horizon-1/2/4 predictions had exactly zero measured drift in that study. The new final-attention study rehashed all 18 original baseline artifacts and found them unchanged. None of these checks substitutes for a spatial localization test.

## 3. Do not confuse the current signed maps with raw attention

The current `test_80` maps are **signed contributions to a trained prototype decision**, not self-attention probabilities, Grad-CAM, LeGrad, segmentation masks or calibrated spill probabilities.

Let `mu` be the mean valid patch embedding, `z = mu / ||mu||`, and `d = prototype_spill - prototype_clean`, using training-only prototypes. The displayed value is

\[
r_{t,n}=\frac{q_{t,n}^{\mathsf T}d}{\|\mu\|}.
\]

Its valid-patch mean reconstructs the actual episode margin `z^T d`.

The measured `test_80` display convention is:

- **Red:** positive contribution to the spill-versus-clean margin.
- **Blue:** negative contribution, supporting the clean side of that decision.
- **White:** zero contribution.
- One shared scale, approximately **[-2.05783844, +2.05783844]**, across both models, sources and times.
- Overlay alpha **0.45**, with no per-frame normalization.

For a signed map, blue does **not** mean small absolute importance or verified background. A large negative contribution can be important evidence. Do not relabel these values as nonnegative attention.

For a future nonnegative attention/relevance map, blue may mean low numerical relevance and red high relevance, provided the target and method are explicitly defined. Red means high measured model relevance; whether it corresponds to the spill is an empirical outcome, not a color-mapping rule.

Keep raw numerical maps and their sign. If positive and negative evidence are displayed separately, state the transformation and preserve the signed version. Do not clip negative evidence silently or invert the target merely to obtain a prettier red spill.

## 4. What was fixed and what remains unresolved

Confirmed engineering corrections already implemented:

1. Retain patch tokens before the new spatial loss rather than passing only `[B,T,1024]` means to the temporal model.
2. Distinguish original-image/tubelet IDs from video/split IDs.
3. Replace cubic-memory uncertainty rank comparisons with exact rowwise ranks and tie handling.
4. Use a chunked direct-pair computation with a tested first-order stop-mining gradient, including both anchor and candidate contributions.
5. Preserve float64 contrastive arithmetic and fix an unintended float32 rank conversion.
6. Provide signed feature-contribution maps whose mean reconstructs the classifier margin.

The `test_80` checkpoint replay checks reached maximum errors of approximately `2.98e-7` for patch features, `1.25e-6` for signed maps and `3.58e-7` for margin reconstruction. Source hashes, frame alignment and H.264 frame counts were checked. These checks support checkpoint/map identity and indexing; synthetic corner-map tests are a separate geometry requirement.

The direct all-patch objective still admits a spatially homogeneous class code. Assigning `q=+a` to all patches of one class and `q=-a` to all patches of the other can satisfy class separation without identifying a liquid region. Same-image weight one does not identify which positions contain liquid.

Frozen V-JEPA tokens are also contextual: a token at a background position may encode information from a spill elsewhere. Feature-position contributions are not automatically input-pixel attribution.

Frozen backbone limitations, image-label ambiguity and dataset shortcuts remain competing explanations. Do not declare that the colormap, final attention block or freezing alone caused the failure before testing them.

## 5. Actual ViCo-LWIR provenance and supervision

Read [`loss_doc/VicoLWIR.pdf`](loss_doc/VicoLWIR.pdf) and `src/vision_model/training/vico.py` before changing the objective.

The paper explicitly describes **patch-level ViT-B/16 representations** in Section IV-A.2, printed page 7. It reports a 256-dimensional projector and backbone fine-tuning before freezing for downstream classification. This differs from the current permanently frozen V-JEPA stage.

Eq. 6, printed page 5, provides the rule clarified by the user:

- Distinct positive instances from the **same original image** have weight **1**.
- Positive instances from **different images of the same class** receive uncertainty/curriculum weights.
- Different-class negatives receive the Eq. 7 hard-negative weighting.

The direct-patch experiment implements the user's all-patch interpretation, retaining every valid content cell and its original-image identity. Its patch labels are **weak associations inherited from the source image/episode**, not verified spill-region annotations. Background cells in positive sources also participate under that association. Report this explicitly rather than claiming correct patch-level labels.

Preserve the original per-positive competition:

\[
\ell_{ij}=\operatorname{softplus}\!\left[
\log\!\left(\sum_{k\in\mathcal N(i)}\alpha_{ik}e^{s_{ik}/\tau}+\epsilon\right)
-\left(s_{ij}/\tau+\log w_{ij}\right)
\right].
\]

Keep Eq. 7's square-root negative-count factor, detached mining weights, temperature applied once, the established curriculum, float64 semantic arithmetic and fixed-final-curriculum evaluation. Do not substitute shared-denominator SupCon or attribute a new local retention/mirror/gating term to the paper.

The paper's equations do not fully specify a video `[source,view,time,patch]` implementation. Document this adaptation and verify author code if available before claiming exact reproduction.

**No MIL is included in the next comparison.** Preserve the direct all-patch result as a diagnostic control. If a new confidence-weighted spatial objective is proposed, define valid positive/negative relations mathematically; attention rank or confidence alone does not make a patch a verified spill. Do not silently change the supervision while attributing an improvement to backbone fine-tuning.

## 6. Spatial visualization audit before further training

Audit the actual mapping:

`source frames → official preprocessing → causal tubelet tokens → named target/map → native grid → inverse letterbox → color mapping → overlay`.

Verify:

- Complete field of view, 256x256 letterboxing, official checkpoint normalization and a single rescaling.
- Native 16x16 grid from 16x16 patches and two-frame tubelets; width 1024.
- Actual patch order, horizontal/vertical orientation and valid content cells.
- Temporal/key indexing and source timestamps, including repeated-tail exclusion.
- Inverse resize and padding, including asymmetric-padding handling where relevant.
- RGB/BGR conversion, interpolation and reproducible alpha blending.

For the corrected spatial evidence overlay, use

\[
I_{\rm overlay}(x)=(1-\alpha(x))I(x)+\alpha(x)H(\operatorname{sign}A(x)),
\qquad \alpha(x)=0.4\min(|A(x)|/s,1).
\]

Here `s` is the shared fixed scale, and `H` is red for positive evidence and blue for negative evidence. For nonnegative conditional key gradients use red magnitude opacity. Zero evidence must reproduce the original RGB exactly. Do not supply contextual episode contributions or raw attention as if they were local spill evidence. Retain those fields separately for audit. Use nearest-cell inverse letterboxing for the audit overlay; do not blur weak cells into neighboring regions.

An alternative **explicitly named blue-to-red display** is now supported for nonnegative conditional gradients: `I_overlay = 0.6 I + 0.4 C(A/s)`, with fixed color knots blue/cyan/green/yellow/red at `0, 0.25, 0.5, 0.75, 1`. In this mode zero is deliberately blue, while the magnitude-opacity mode above keeps zero transparent. Record the mode in the manifest and never confuse their zero/sign conventions. Keep the same global scale and nearest native cells; show the raw field alongside the frame and mark the unmodified numerical maximum. This display convention supplies no spill/background supervision.

Keep the existing alpha-0.45 figures unchanged. The new renderer changes the display contract, not the underlying evidence values or model. Use a scale shared across comparable models, videos and times; per-frame or per-sequence scaling must not hide uniformly weak or spatially flat responses.

Before interpreting new attention, test synthetic maxima at top-left, top-right, center, bottom-left and bottom-right, plus a double flip, true mirrored-map alignment and padding exclusion. Verify placement after the exact inverse preprocessing.

For every representative spill and clean source, save original image, native grid values, upsampled map, overlay and comparison. Keep raw tensors and scale metadata. Do not show only successful examples.

## 7. Executed controlled final-attention fine-tuning

The tested hypothesis was whether selective attention adaptation improves **spill-region focus**, while the rest of the system is controlled. The completed bounded comparisons did not satisfy that criterion; results, precision differences, failures and limitations are in the linked report.

### Actual trainable module

For the audited local V-JEPA 2 ViT-L implementation, there are 24 blocks, indexed 0–23. Inspect the loaded architecture before selecting the module.

Start by making only these final-block parameters trainable:

- `encoder.blocks[23].attn.qkv.weight` and `.bias` — fused Q/K/V projection.
- `encoder.blocks[23].attn.proj.weight` and `.bias` — attention output projection.

Keep blocks 0–22, patch embedding, block-23 MLP, block-23 norms and final encoder norm frozen initially. At width 1024 with the audited biases, the expected attention parameter count is **4,198,400**; verify the actual runtime count and full trainable parameter list.

The existing pointwise patch projector can remain trainable in both the frozen control and selective-attention variant. Do not introduce the previously collapsed class-query head or any MIL module as an unrecorded architecture change.

### Differentiable path and valid cache location

The accepted cache contains final/intermediate **last-tubelet output** tokens at blocks 5, 11, 17 and 23. It cannot train block-23 attention. Setting `requires_grad=True` on attention parameters while consuming its detached output cache has no training effect.

Use either a fresh differentiable causal forward or a newly verified cache of the **input to block 23**:

\[
h^{\rm pre23}_{p}\in\mathbb R^{B\times(p\cdot256)\times1024}
\]

for each separately encoded observed prefix `p`. Preserve **all prefix tokens** needed as attention keys/values, not only the final tubelet. A block-17 output or a detached post-attention token is not a substitute.

The student path must execute:

`pre23 prefix tokens → frozen norm1 → trainable attention → residual → frozen norm2/MLP/residual → frozen encoder norm → final tubelet → spatial projector → declared loss`.

Frozen downstream parameters do not imply `no_grad()` around their operations. Their derivatives must remain available to the trainable attention output. Verify frozen forward replay in the declared precision before the first optimizer step, including autocast, 3D positions, masks and grid dimensions. The executed comparison used FP32 suffix/master parameters in both arms to preserve small updates: original BF16 teacher means replay exactly, while selected-query versus full FP32 suffix error is at most 0.000143052. Record the nonzero BF16-to-FP32 suffix difference separately; do not claim bitwise equality between those precision paths.

Keep identity/mirror inputs as actual independently encoded views. Do not reuse a flipped feature grid as an exact re-encode. Additional augmentations require compatible new prefix caches.

### Temporal teacher stays separate

The original temporal branch consumes its original frozen global-mean cache. Do not replace its inputs or future/retention targets with outputs of the newly trainable attention block.

Its existing ViCo/alignment/prediction/retention terms remain monitored constants while that branch is frozen. They do not regularize an independent spatial branch through a nonexistent gradient path. Any local feature-retention term is separately named, uses detached original teacher targets and is held fixed across matched controls.

### Optimization

Begin with one bounded pair: frozen attention versus trainable final attention, from identical spatial-head initialization/checkpoint and original backbone weights. Keep data exposure, loss, local-retention choice, seeds and evaluation identical.

The initial comparison used final-attention LR `3e-6` and projector LR `1e-4`. An exploratory stronger attention LR `1e-4` reused the identical frozen control; both comparisons used constant learning rates and the update-21 reset. This is not an exact reproduction of the paper's ViT-B/16 fine-tuning. Do not launch a larger grid just to obtain attractive maps.

If warmup/cosine replaces the existing constant schedule, apply it to the matched frozen control too. Record the update-21 AdamW-reset policy consistently. Do not attribute a schedule, head, supervision or budget change to attention fine-tuning.

Measure per-loss gradient norms, conflicts, final-attention weight changes, representation drift and exact frozen-parameter regression. The previous local-retention beta-1 gradient cosine was approximately -0.9933, showing that preservation and all-patch attraction can strongly oppose each other.

## 8. Class-related attention and input attribution

Raw attention probabilities are not class-specific explanations automatically. Define the trained target scalar, for example the training-prototype spill-minus-clean margin, and explain how the map depends on it.

For attention-gradient methods, capture the **differentiable probabilities actually used in value aggregation**. Verify native-versus-captured forward and gradients. Record all-head/layer aggregation and positive/negative gradient mass. A returned/disconnected tensor or an empty ReLU-clipped map must not be misinterpreted as no spill or no dependence.

Final-tubelet queries can attend to keys from earlier observed tubelets. Keep the key's time and spatial coordinates. Do not collapse past key locations into a map over the current image without a justified alignment operation.

If a method is called LeGrad, Grad-CAM or another published name, verify its scalar, capture and aggregation against that method. Distinguish explanation of the backbone's own features from explanation of the trained spill decision.

The completed full-input integrated-gradient diagnostics are **numerically unreliable** on the four fixed sources: 16/64-point bf16 integration did not consistently pass completeness, and a separately labelled float32 replay also failed. Do not use those maps as localization proof. The current helper flags failed numerical checks. Resolve baseline/quadrature/gradient reliability before expanding that attribution evaluation; float32 replay is not identical to cached bf16 inference.

## 9. Mirror consistency is conditional

The previous mirror loss produced nearly uniform maps. Keep a matched **mirror-off** experiment first.

Only test

\[
L_{\rm mirror}=D(A(x),F^{-1}(A(F(x))))
\]

once the map's target, geometry, discriminative path and non-collapse behavior are verified. Define distance, class/time/source reduction and content-support normalization explicitly. Preserve gradients through both real views where intended, and verify gradients to every active projector/attention parameter.

Uniform or background-focused consistency is a failure, even if the mirror error is almost zero. Do not promote it using t-SNE or classification alone.

## 10. Dataset, experiments and distributed requirements

Use the existing **80 byte-unique sources: 40 clean and 40 cold, no warm videos**. Preserve the **64/16 source split** for comparisons. Every derived view stays with its source. Verified physical scene/session groups and spill masks are unavailable; environments recur across the split.

This is a previously used **source-validation selection set**, not an untouched scene-held-out test. Do not fabricate scene labels or declare scene generalization from it. A genuinely new test must exclude its sources/groups from every fitted component and prototype gallery.

Use distinct experiment names, without overwriting historical E0/E1/E2 identifiers:

| Proposed stage | Attention | Supervision/readout | Mirror |
|---|---|---|---|
| Historical controls | Frozen | Existing temporal, direct-patch and retention results | As recorded |
| FT0 | Frozen | Predeclared spatial objective, same projector as FT1; no MIL | Off |
| FT1 | Final attention trainable | Exactly the FT0 objective and exposure schedule | Off |
| FT2, conditional | Final attention trainable | Exactly FT1; only mirror term changes | On |
| Separate loss ablation, conditional | Matched attention state | Explicitly justified confidence-weighted patch relations | Off initially |

If direct image-label patch associations are used to isolate the attention change, identify their weak-label risk in both FT0 and FT1. If the objective instead avoids assigning positive image labels to background patches, define its reliable relations first and run a separate matched loss ablation. Do not claim a confidence score proves a patch is spill.

Default to the established 64-source effective batch where feasible; never silently reduce below 32. Distinguish source, view, tubelet and patch exposure counts. All valid content patches participate in the declared all-patch control; temporal scheduling must be explicit.

All participating GPUs own distinct assigned sources. Synchronize every trainable parameter through DDP, preserve gradients through candidate gathering, compensate for DDP reduction with unequal counts, avoid duplicated evaluation sources and aggregate metrics across ranks. Only global rank zero writes consolidated outputs. Verify global gradients against a single-process reference, including the trainable attention parameters.

The previous comparison used 100 updates and one seed. Do not call it fully converged from semantic loss. Record semantic/temporal/spatial slopes separately. Multiple seeds and a defensible validation protocol are needed before stronger recommendations.

## 11. Localization is the acceptance criterion

For every evaluated spill source, compare:

`Original | Ground truth, if available | Frozen map | Fine-tuned map | Difference`.

The primary question is whether the red/high-response region follows the **actual visible liquid**, rather than background, people or a fixed scene location. Where the liquid spreads or moves, inspect whether the evidence follows that change at correctly aligned times. The supplied example is a visual goal, not a reason to change numerical values.

Where verified annotations exist, report pointing accuracy, pixel AP, threshold-swept IoU and inside/outside spill-region mass on unseen scenes. Do not calculate these against unverified boxes or invented masks.

There are currently no masks. Report qualitative source-frame agreement honestly and use matched-area input removal as supporting evidence: high-response versus low-response versus several seeded random background/content selections, with every prefix freshly re-encoded. A larger change under high-response removal can support decision relevance but does not alone prove the region is liquid; redundancy and OOD effects must be considered.

Report classification, retrieval and original ViCo separation metrics separately. No extra acceptance credit is given for perfect classification already achieved by the baseline. Preserve full failure galleries and fixed scales. Unverified spill focus must remain an unresolved localization claim.

**A run fails the intended localization objective if it only produces good t-SNE clusters, low/converged loss, uniform or scene-wide red maps, or consistent background activation.** Geometry correctness, trainable attention and good gradients are engineering prerequisites, not substitutes for actual spill focus.

## 12. Required verification and new report

Before full fine-tuning, pass focused tests for:

1. Native-grid orientation, synthetic corner/center maxima and inverse letterboxing.
2. Signed versus nonnegative color semantics, RGB/BGR handling and alpha blending.
3. True mirrored-view alignment, timestamp/tubelet indexing and valid-frame exclusion.
4. Exact pre23-cache replay and the loaded architecture's actual final attention module.
5. Trainable Q/K/V/output projections receiving finite nonzero gradients.
6. Frozen earlier blocks, frozen final MLP/norms and original temporal outputs remaining unchanged.
7. Downstream frozen operations preserving the upstream attention gradient path.
8. Checkpoint compatibility, same-image weight-one rules, detached mining weights and source-balanced reductions.
9. Unequal-shard DDP gradients, source ownership and global metric aggregation.
10. Uniform/background-map diagnostics and attribution completeness/reliability guards.

Run focused tests and the required full suite. `.venv/bin/python` is absent; checks use `/home/olemou/miniconda3/envs/torchgpu/bin/python` with its matching library path. Record environment differences and skipped tests. The completed 180-test suite includes final-attention/DDP gradients, native-cell geometry, transparent zero evidence, and actual encoded-video opacity/shared-scale regression. These engineering tests do not certify physical spill localization.

Write a new `report.md` in a unique experiment directory. Preserve existing checkpoints and figures, including `test_80`.

The report must distinguish:

- **Measured evidence:** before/after source frames, raw tensors, overlays, relevant metrics and controls.
- **Engineering verification:** exact trainable layer/parameter count, gradient path, geometry, checkpoint/source/cache hashes and DDP correctness.
- **Scientific outcome:** whether attention actually concentrates on the spill, whether that evidence generalizes, and what remains unproven.

Include all executed stages, failed settings, exact checkpoints, model-selection use, optimization/exposure budgets, collapse diagnostics, classification and spatial outcomes. Recommend a checkpoint as a spill localizer only when localization evidence supports that claim. If selective fine-tuning does not improve spill focus, retain the supported baseline and report the failure.

**Final research decision:** Has the model learned where the spill is, or only how to separate the supplied video labels? The answer must come from spatial evidence on the actual liquid, not t-SNE or loss convergence alone.
