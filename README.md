# V-JEPA 2 supervised contrastive trajectory learning for spill videos

This repository learns a binary video representation for clean trajectories
and warm/cold spill trajectories. The current system combines a frozen
pretrained V-JEPA 2 visual backbone with a trainable causal temporal adapter,
a ViCo-LWIR Eq. 10 semantic objective, view alignment, future prediction, and
teacher retention.

The latest 100-epoch run achieves strong separation on all 14 fitted source
videos. It does not yet prove that the representation recognizes the physical
spill rather than scene-correlated cues, and its attention maps do not establish
pixel localization. The repository includes explicit audits for those limits.

## Current pipeline

```text
source MP4
  -> sample 64 frames at 4 fps, repeat short tails only as masked padding
  -> full-frame 256x256 letterbox + checkpoint normalization
  -> optional clip-consistent thermal augmentation
  -> frozen pretrained V-JEPA 2, separately encoded causal prefixes
  -> mean of the final tubelet's 256 spatial tokens: [time, 1024]
  -> trainable two-block causal temporal adapter
  -> semantic embedding [128] + dynamics states + future predictor
  -> ViCo Eq.10 + view alignment + future prediction + teacher retention
```

The labels are:

| Label | Source categories | Meaning |
|---:|---|---|
| 0 | `clean` | no spill |
| 1 | `cold`, `warm` | spill |

The mapping in `src/vision_model/data/labels.py` is authoritative. The current
manifest contains 14 byte-unique videos: 7 clean and 7 spill.

## 1. Loading pretrained V-JEPA 2

The backbone is `facebook/vjepa2-vitl-fpc64-256`. The repository builds its
audited local `VJEPA2Encoder` and uses Hugging Face only as a checkpoint source:

```python
from vision_model.models.encoder import load_pretrained_encoder

model = load_pretrained_encoder(
    "facebook/vjepa2-vitl-fpc64-256",
    local_files_only=True,
)
model.freeze_encoder(True)
encoder = model.encoder.eval()
```

Loading validates the checkpoint architecture, converts every encoder tensor,
concatenates query/key/value weights into the local layout, rejects unconsumed
keys, and performs strict state loading. The custom RoPE layout and final-block
replay are tested against the installed Transformers implementation.

The ViT-L backbone has 24 transformer blocks, width 1024, 16 attention heads,
16x16 image patches, and two-frame tubelets. A 256x256 tubelet contains 256
spatial tokens. The model has no CLS token.

This is parameter-efficient fine-tuning. The pretrained backbone is frozen and
the trajectory model is trained. It is not end-to-end V-JEPA 2 fine-tuning.
That choice fits the small dataset and GPU budget, but raw backbone attention
cannot adapt to the spill labels.

## 2. Spill-preserving preprocessing and augmentation

`OfficialVJEPA2Transform` preserves the complete camera field of view. It scales
the source frame into a 256x256 canvas and fills the letterbox with channel-mean
pixels. Center cropping is disabled because a spill can occur near an edge.

After the standard `1/255` rescaling, the default checkpoint normalization is:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

`ThermalAugmentor` samples one transformation for the entire clip:

\[
x'_{t,c,u,v}=\operatorname{round}\left[
b\left(c x_{t,c,u,v}+(1-c)\mu_{t,c}\right)\right],
\]

where `b ~ Uniform(0.9,1.0)`, `c ~ Uniform(0.85,1.0)`, and `mu` is the spatial
mean of the same frame and channel. A whole-clip horizontal flip is sampled
with probability 0.5. The transform is monotone, identity at `b=c=1`, and uses
the same geometry and intensity parameters throughout a video.

Crop, erase, affine clipping, elastic deformation, and frame-local random
geometry are disabled. Without verified masks, no image region can safely be
deleted. The executed policy lives in `src/vision_model/data/transforms.py`;
the broad `ThermalAugConfig` dataclass is a legacy parameter schema and does not
execute those old augmentation fields.

The latest experiment pre-encodes eight seeded thermal views plus identity and
whole-clip mirror, producing a finite bank of 10 views per source. During each
epoch, two distinct bank views are selected for every source. This is not fresh
pixel augmentation on every update.

## 3. Genuine causal feature extraction

The sampler requests 64 frames at 4 fps. A short video repeats its last frame to
fill the tensor, but the validity mask excludes repeated tail tubelets.

For every real tubelet endpoint `p`, extraction runs the frozen encoder on only
the frames available through `p`. It then averages the 256 tokens belonging to
the last observed tubelet:

\[
f_p=\frac{1}{256}\sum_{q=1}^{256}E(x_{1:2p})_{p,q}.
\]

Encoding each prefix separately is required. V-JEPA attention is bidirectional
inside an encoded clip, so slicing a single full-video encode would leak future
frames into earlier features.

The cache records features, valid times, timestamps, binary labels, source IDs,
video SHA-256 values, manifest hash, model ID, normalization, spatial policy,
augmentation seeds, and the policy
`past_only_prefix_last_tubelet_v1`. Training rejects duplicates, modified
manifests, and caches with another temporal-context policy.

## 4. Trainable trajectory model

For a frozen prefix feature `f_t in R^1024`, the trainable model computes:

```text
LayerNorm(f_t) -> Linear(1024,128) + Linear(timestamp,128)
  -> two causal TransformerEncoder blocks, 4 heads, FFN width 256
  -> zero-initialized Linear(128,1024) residual added to f_t
  -> normalized state s_t
```

The normalized states feed three paths:

- an event score creates temporal weights `beta_t`;
- a dynamics MLP produces 128-dimensional states for view alignment and future
  prediction;
- pooled video state `0.5 * mean_t(s_t) + 0.5 * sum_t beta_t s_t` enters
  `Linear(1024,64) -> GELU -> Linear(64,128) -> L2 normalization`.

The output `z` is the semantic embedding used for class separation. Intermediate
time states are not assigned the whole-video spill label.

Because the encoder's 256 spatial tokens are averaged before this model, its
trainable attention operates over time only. It has no trainable patch-level
spill attention.

## 5. ViCo-LWIR Eq. 10 semantic loss

The authoritative semantic objective is Eq. 10 from `loss_doc/VicoLWIR.pdf`,
implemented in `src/vision_model/training/vico.py`. It is deliberately different
from conventional shared-denominator supervised contrastive loss.

Let `z_i` be an L2-normalized whole-video embedding, `s_ik=z_i^T z_k`,
`P(i)={j != i : y_j=y_i}`, and `N(i)={k : y_k != y_i}`. For each positive pair:

\[
\ell_{ij}=
-\log\frac{w_{ij}\exp(s_{ij}/\tau)}
{w_{ij}\exp(s_{ij}/\tau)+
 \sum_{k\in N(i)}\alpha_{ik}\exp(s_{ik}/\tau)+\epsilon}.
\]

Equivalently, the stable implementation is:

\[
\ell_{ij}=\operatorname{softplus}\left(
\log\left[\sum_{k\in N(i)}\alpha_{ik}e^{s_{ik}/\tau}+\epsilon\right]
-\left[s_{ij}/\tau+\log w_{ij}\right]\right).
\]

Then:

\[
L_{\mathrm{ViCo}}=\frac{1}{B}\sum_i
\frac{1}{|P(i)|}\sum_{j\in P(i)}\ell_{ij}.
\]

Defaults are `tau=0.1` and `epsilon=1e-6`. Each positive competes against the
anchor's negatives in its own denominator. Other same-class positives are not
placed in that denominator, so there is no `log(|P(i)|)` minimum. A near-zero
ViCo value is mathematically possible when positives approach cosine 1 and
negatives approach cosine -1.

Negative weights `alpha` follow Eq. 7, including its square-root negative-count
factor. Positive weights `w` use the uncertainty-rank curriculum from Eq. 8;
views from the same source always have weight one. Pair-mining weights are
detached, rank ties use average rank, temperature is applied once, and the small
loss is evaluated in float64.

## 6. Combined learning objective

The full trajectory loss is:

\[
L = L_{\mathrm{ViCo}}
  + 0.1L_{\mathrm{align}}
  + 1.0L_{\mathrm{predict}}
  + 0.1L_{\mathrm{keep}}.
\]

The terms have distinct roles:

| Term | Definition | Purpose |
|---|---|---|
| `L_ViCo` | Eq. 10 on two whole-video embeddings per source | separate clean and spill classes |
| `L_align` | Smooth L1 between matched-view dynamics at valid times | augmentation/view invariance |
| `L_predict` | Smooth L1 between predicted and frozen future normalized features at 1, 2, and 4 tubelets | preserve temporal evolution |
| `L_keep` | Smooth L1 between adapted and frozen normalized states | prevent uncontrolled representation drift |

The future predictor receives the dynamics state and actual timestamp delta.
Errors are normalized by the number of valid scalar targets across all horizons.
The frozen-teacher persistence error is recorded as a baseline. All teacher
targets are detached.

## 7. Reproduce the current 100-epoch experiment

Run from `/home/olemoufe/jepa/vision-model`. The checkpoint is already cached,
so these commands use offline mode:

```bash
cd /home/olemoufe/jepa/vision-model

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 USE_TF=0 \
.venv/bin/python scripts/preview_thermal_augmentation.py \
  --output reports/trajectory_thermal_aug_100

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 USE_TF=0 \
.venv/bin/python scripts/extract_causal_video_features.py \
  --manifest reports/experiments/unique_7_clean_7_spill/manifest.jsonl \
  --dataset-root dataset \
  --output reports/trajectory_thermal_aug_100/causal_features.pt \
  --thermal-views 8

.venv/bin/python scripts/train_trajectory.py \
  --cache reports/trajectory_thermal_aug_100/causal_features.pt \
  --output reports/trajectory_thermal_aug_100 \
  --epochs 100 \
  --seed 239 \
  --learning-rate 0.001 \
  --curriculum-epochs 20 \
  --align-weight 0.1 \
  --predict-weight 1.0 \
  --keep-weight 0.1

.venv/bin/python scripts/report_thermal_trajectory.py
```

One epoch is one full-batch optimizer update. Training uses AdamW with weight
decay 0, epsilon `1e-12`, and gradient clipping at 5. Optimizer moments are reset
once before update 21, after the mining curriculum. Evaluation after every
update uses the fixed identity/mirror views and final-curriculum weights, making
the curve comparable over time.

Principal outputs are:

```text
reports/trajectory_thermal_aug_100/
  causal_features.pt       frozen causal features and metadata
  checkpoint.pt            trajectory model + optimizer + settings
  representations.pt       fitted states and embeddings
  history.csv              epoch 0 through epoch 100 metrics
  report.json              protocol, settings, exposure counts, full history
  curves.png / curves.pdf  combined and component losses
  comparison.png           augmented versus baseline run
```

## 8. Measured epoch-100 result

The reported curve evaluates the same fixed identity/mirror features throughout:

| Metric | Epoch 100 |
|---|---:|
| Total combined loss | `0.0163736491` |
| ViCo Eq. 10 semantic loss | `2.05862565e-8` |
| Worst positive-pair loss | `5.26989741e-8` |
| Future prediction loss | `0.0157210547` |
| View alignment loss | `0.0010875592` |
| Teacher retention loss | `0.0054381778` |
| Minimum same-class cosine | `0.9414769` |
| Maximum cross-class cosine | `-0.8557208` |
| Minimum per-anchor class gap | `1.7971977` |
| Training prototype accuracy | `14/14` |

The semantic objective is effectively at its numerical floor for this fitted
set and the classes are strongly separated. The total loss is about `0.0164`
because future prediction, alignment, and retention are real nonzero objectives.
It should not be expected to equal the semantic loss or reach zero. The joint
curve was still decreasing at epoch 100, so this is a fixed-budget result rather
than a certified global optimum.

Across all 140 pre-encoded views, the minimum positive cosine is `0.940685`, the
maximum negative cosine is `-0.851294`, and source-excluded prototype accuracy
is 140/140. The model was still fitted with those source videos, so this remains
a training-set robustness diagnostic.

## 9. Attention and spill-localization audit

Use the provided audits rather than interpreting a colored overlay by eye:

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 USE_TF=0 \
.venv/bin/python scripts/investigate_final_block.py

.venv/bin/python scripts/report_temporal_final_block.py

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 USE_TF=0 \
.venv/bin/python scripts/audit_trajectory_localization.py \
  --run reports/trajectory_thermal_aug_100
```

The final-block audit reconstructs all 16 heads of frozen V-JEPA block 23,
computes a class-conditioned one-block attention-times-gradient diagnostic, and
compares it with an identically initialized untrained head. It also inspects the
trained four-head temporal block and performs final-attention and full-block
ablations.

For `warm_scene_1_hot`, the fitted spill-minus-clean margin is `1.891416`, but:

- initial and trained class-gradient spatial maps have Spearman correlation
  `0.9243`;
- trained map mass in the broad bottom-35% floor proxy is `37.08%`, compared
  with `37.37%` for the initial head;
- removing the final attention branch retains a spill margin of `1.646636`;
- replacing source content with channel-mean pixels still gives a positive
  margin of `1.422834`.

The trainable temporal attention did change, but it sees time-by-feature tensors
after spatial averaging. These results support learned class separation on the
fitted videos; they do not establish that the model learned the liquid region.
Raw V-JEPA attention stayed frozen for all 100 epochs.

The visualization color blue means low displayed attribution under a shared
threshold. It is not a calibrated clean/background probability. The diagnostic
is not a segmentation mask, full relevance propagation, Grad-CAM, or pixel-level
causal attribution.

Detailed evidence and videos are in:

- `reports/trajectory_thermal_aug_100/README.md`
- `reports/trajectory_thermal_aug_100/final_block/README.md`
- `reports/trajectory_thermal_aug_100/final_block/*/final_block_h264.mp4`
- `reports/trajectory_thermal_aug_100/localization_audit/report.json`

## 10. What is required for learned spill focus

The next model revision must preserve a spatial token axis through a trainable
local branch. A sound design is:

```text
V-JEPA spatial tokens [time,16,16,1024]
  -> content-validity mask for letterbox cells
  -> trainable spatial attention/localization head
  -> spatial spill evidence per time
  -> temporal trajectory adapter and global semantic embedding
```

Keep the global ViCo objective for video-level separation and the temporal losses
for dynamics. Add a localization objective only when its target is defensible:
verified spill masks with BCE/Dice, or a separately validated weak-localization
method with matched clean/spill scenes and input-level intervention tests.
Fine-tuning the final V-JEPA block can then be evaluated, but unfreezing alone
does not prevent scene or border shortcuts.

A claim of spill localization should require unseen-scene evaluation with real
masks and quantitative metrics such as IoU or pointing accuracy. A claim of
generalization across tasks, objects, and environments requires group-held-out
data where every related source and derived view stays outside training.

## 11. Tests and environment

The local environment uses Python 3.11-3.13, PyTorch 2.7, torchvision 0.22,
Transformers 4.57.6, and Accelerate 1.15. FFmpeg and FFprobe are required for
video export and validation.

```bash
cd /home/olemoufe/jepa/vision-model
.venv/bin/python -m pytest -q
```

The latest verified suite contains 104 passing tests, including loss invariants,
augmentation behavior, causal trajectory handling, checkpoint conversion,
final-block reconstruction, gradient fidelity, and parity with the official
V-JEPA 2 layer.

Historical Gaussian, shared-denominator SupCon, margin, and duplicate-containing
experiments remain under `reports/` for auditability. They are not the current
training objective. The complete rules for future changes are in `AGENTS.md`.
