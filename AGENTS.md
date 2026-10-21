# Repository instructions for spill video learning

These instructions apply to the entire repository. The current authoritative
pipeline is the causal trajectory pipeline implemented by
`scripts/extract_causal_video_features.py`, `scripts/train_trajectory.py`,
`src/vision_model/training/trajectory.py`, and
`src/vision_model/training/vico.py`.

## Scientific objective

Learn a video representation that separates class 0 (`clean`) from class 1
(`cold` or `warm` spill), preserves temporal changes, and can eventually support
spatial spill localization. Treat these as separate claims:

1. a low semantic loss demonstrates fit of the supplied class labels;
2. temporal prediction below the persistence baseline demonstrates useful
   trajectory modeling on the evaluated data;
3. spill localization requires spatial evidence and localization validation;
4. generalization requires videos or scene groups excluded from every training
   operation, including representation fitting and prototype construction.

Never infer claim 3 or 4 from claim 1 alone.

## Data and labels

- The authoritative binary mapping is `clean -> 0`, `cold/warm -> 1` in
  `src/vision_model/data/labels.py`.
- Reject byte-identical source videos before training. A view, frame, or
  augmentation derived from one source must remain in the same split as that
  source.
- The current 14-video run is an all-data optimization study. Source-excluded
  prototypes do not make it a held-out test when the source trained the model.
- Preserve `temporal_context_policy=past_only_prefix_last_tubelet_v1` for future
  prediction. Features from a full video are not valid causal-prefix targets.
- Repeated tail frames are padding and must not contribute to attention,
  pooling, alignment, or prediction losses.

## Pretrained V-JEPA 2 model

- Use checkpoint `facebook/vjepa2-vitl-fpc64-256` unless an experiment explicitly
  records another model ID.
- Load with `load_pretrained_encoder()`. It validates the architecture, consumes
  every Hugging Face encoder key, maps Q/K/V explicitly, and uses strict state
  loading. Do not silently accept missing or unexpected weights.
- The current training is parameter-efficient fine-tuning: V-JEPA 2 is frozen;
  the causal trajectory adapter and its heads are trained. Do not describe it as
  end-to-end backbone fine-tuning.
- The encoder accepts `[B,3,T,H,W]`, uses tubelets of two frames and 16x16
  patches, and returns 1024-dimensional tokens. At 256x256 there are 256 spatial
  tokens per tubelet.
- Prefix feature extraction must encode each observed prefix separately and
  average only the final tubelet's 256 spatial tokens. Reusing tokens from a
  full-video encode would leak future context.
- The trainable temporal adapter receives `[video,time,1024]`. Spatial tokens
  have already been averaged, so its attention can select times but cannot learn
  a patch-level spill mask.

## Spill-preserving preprocessing and augmentation

- Use `OfficialVJEPA2Transform(preserve_field_of_view=True)`.
- Resize the complete frame into the 256x256 checkpoint grid and letterbox with
  channel-mean pixels. Do not center-crop a possible spill.
- Keep checkpoint normalization unless a new experiment explicitly validates
  and records a replacement: mean `(0.485,0.456,0.406)`, standard deviation
  `(0.229,0.224,0.225)`, after one `1/255` rescaling.
- The supported augmentation is `ThermalAugmentor` in
  `src/vision_model/data/transforms.py`. One parameter set is shared by the
  entire clip:

  `x_aug = round(b * (c*x + (1-c)*mu))`,

  where `b ~ U(0.9,1.0)`, `c ~ U(0.85,1.0)`, `mu` is the per-frame/channel
  spatial mean, and whole-clip horizontal flip has probability 0.5.
- Do not introduce crop, erase, cutout, elastic deformation, temporal reversal,
  frame-local random geometry, or an assumed protected center without verified
  spill masks. Any of these can delete the target or change its dynamics.
- Sampling parameters must be reproducible with a local `torch.Generator` and
  stored with the feature cache.
- The current eight augmented views are a finite pre-encoded bank. Do not call
  them fresh augmentation at every epoch.

## Authoritative combined loss

The training objective is

`L = L_ViCo + 0.1 L_align + 1.0 L_predict + 0.1 L_keep`.

`L_ViCo` is ViCo-LWIR Eq. 10 as implemented in
`src/vision_model/training/vico.py`. For normalized whole-video embeddings
`z_i`, cosine `s_ik=z_i^T z_k`, temperature `tau=0.1`, and `epsilon=1e-6`, each
positive pair has its own positive-versus-negative competition:

`ell_ij = softplus(log(sum_{k:y_k!=y_i} alpha_ik exp(s_ik/tau) + epsilon)
                   - (s_ij/tau + log(w_ij)))`.

Average `ell_ij` over positives for each anchor and then over anchors. This is
not shared-denominator SupCon and has no `log(number_of_positives)` floor. Do
not subtract an entropy constant or relabel another objective as Eq. 10.

- `alpha_ik` follows Eq. 7, including its square-root negative-count factor.
- `w_ij` follows the uncertainty-rank curriculum in Eq. 8. Mirrored or augmented
  views of the same source retain weight one.
- Mining weights are detached selection statistics. Do not optimize through
  ranks or hard-pair weights.
- Compute the small semantic loss in float64 for numerical stability.
- The 20-epoch curriculum changes pair weights. Evaluation curves use fixed
  final-curriculum weights at every recorded epoch so points are comparable.

`L_align` is Smooth L1 between matched-view dynamics states at valid times.
`L_predict` predicts normalized frozen future teacher features at horizons
1, 2, and 4 tubelets using the real timestamp delta. Normalize its summed error
by the number of valid scalar targets. `L_keep` is Smooth L1 between adapted
states and normalized frozen teacher states. Targets remain detached.

## Training protocol

- Default run: 100 epochs, where one epoch is one full-batch optimizer update.
- Seed 239, AdamW, learning rate `1e-3`, weight decay 0, epsilon `1e-12`, and
  gradient norm clipping at 5.
- Reset AdamW state once before update 21, immediately after the mining
  curriculum. Record this reset in reports.
- With an augmentation bank, select two distinct, deterministically seeded
  views per source per epoch. Evaluate only the fixed identity and whole-clip
  mirror views.
- Save the cache metadata, model state, optimizer state, settings, history,
  representations, view exposures, and exact number of optimizer updates.
- Never call a fixed 100-epoch run fully converged only because its semantic term
  is near zero. Report the semantic and temporal terms separately and state
  whether their slopes have flattened.

## Evaluation and attention maps

- Report minimum same-class cosine, maximum cross-class cosine, minimum
  per-anchor separation, worst positive-pair loss, and prototype accuracy.
- Compare future prediction with the frozen persistence baseline.
- Keep training-fit metrics separate from source-held-out and scene-held-out
  metrics.
- Raw V-JEPA block-23 attention is frozen in the current run and cannot have
  learned spill focus from the class labels.
- The final-block diagnostic is one-block attention-times-gradient, not a
  segmentation map, Grad-CAM, full relevance propagation, or pixel probability.
  Blue means below the visualization threshold, not `no spill`.
- Use all heads, a shared color scale, no per-frame renormalization, no chosen
  favorable head, and no manual spill ROI when producing audit figures.
- Validate a localization claim with real masks and metrics such as IoU or
  pointing accuracy on unseen scenes, plus input-level removal controls. Generic
  attention and a visually pleasing heatmap are insufficient.
- To make attention learn spill location, retain spatial tokens in a trainable
  local branch and train it with verified spatial supervision or a validated
  localization objective. Unfreezing the final backbone block alone does not
  rule out scene shortcuts.

## Verification and documentation

- Run focused tests for edited modules, followed by `.venv/bin/python -m pytest -q`
  for changes to model, loss, transforms, caching, or attribution.
- Preserve existing experiment directories. Write a new output directory for a
  changed dataset, loss, augmentation, seed, or trainable parameter set.
- A report must distinguish measured results, mathematical implications, and
  unresolved scientific claims.
- Keep the root `README.md` synchronized with current executable defaults.
  Historical objectives belong in their report directories and must not be
  presented as the current loss.

