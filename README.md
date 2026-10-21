# V-JEPA 2 supervised contrastive trajectory learning for spill videos

This repository learns a binary video representation for clean trajectories
and warm/cold spill trajectories. The current system combines a frozen
pretrained V-JEPA 2 visual backbone with a trainable causal temporal adapter,
a ViCo-LWIR Eq. 10 semantic objective, view alignment, future prediction, and
teacher retention.

The latest completed 100-epoch run achieves strong separation on all 14 fitted source
videos. A new balanced manifest for `/home/olemou/videos` contains 78 clean and
78 cold-spill byte-unique videos; it has not yet been trained. The completed run
does not yet prove that the representation recognizes the physical
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

The mapping in `src/vision_model/data/labels.py` is authoritative. The earlier
optimization study used 14 byte-unique videos: 7 clean and 7 spill. The new
balanced manifest has 156 sources after removing 17 byte-identical clean copies.

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

## Temporal attribution of a trained trajectory

`scripts/attribute_trajectory.py` explains a named scalar target through the
trainable temporal adapter's causal self-attention. Use a checkpoint and the
**same** causal feature cache used for training:

For a new MP4 without a prepared feature cache, use the direct-video entry
point. It samples the video with the training cache's fixed policy and encodes
each observed prefix separately with V-JEPA 2. The default output includes
`spatial_final_block_overlay.mp4`, which shows the original sampled frames
beside a target-conditioned V-JEPA patch-sensitivity overlay. It also saves
`temporal_attention_video.mp4`, which shows the trajectory's temporal
relevance strip, plus heatmaps, a storyboard, tensors, and JSON metadata:

```bash
python scripts/visualize_trajectory_video.py \
  --video /home/olemou/videos/cold/example.mp4 \
  --model /path/to/training/checkpoint.pt \
  --encoder-dir /home/olemou/hf \
  --target state --timestep 20 --component 0 \
  --output /path/to/new/attention_output
```

Omit `--timestep` to explain the final observed tubelet. `--model` and
`--checkpoint` are equivalent. For a class-related scalar, use
`--target semantic_similarity --prototype-class 1` when the checkpoint's
training directory contains `representations.pt`; that prototype is a
fitted-source reference. The direct path needs FFmpeg/FFprobe and may take
several minutes because it runs V-JEPA 2 once per prefix. It verifies the
encoder weight hash when the checkpoint recorded one and verifies processor
normalization against the checkpoint's feature-cache metadata. Older
checkpoints without a recorded encoder hash are accepted with
`encoder_weight_verification=model_id_only_hash_unrecorded` in the output;
their local encoder weights cannot be proven byte-identical to training.
Use `--temporal-only` to skip the spatial replay.

The spatial overlay follows the selected **trained trajectory scalar** back
through its frozen teacher features. For each independently encoded prefix,
it computes the scalar's gradient with respect to the prefix feature, then
uses that vector to differentiate the V-JEPA final block's post-softmax
attention probabilities. It applies `ReLU` to the attention gradients,
averages all 16 heads and all 256 final-tubelet query patches, and accumulates
each key patch across causal prefixes containing it. This is the positive
gradient rule inspired by [LeGrad](https://openaccess.thecvf.com/content/ICCV2025/papers/Bousselham_LeGrad_An_Explainability_Method_for_Vision_Transformers_via_Feature_Formation_ICCV_2025_paper.pdf)
and its [official code](https://github.com/WalBouss/LeGrad/tree/main/legrad).
V-JEPA has no CLS token here; the 16×16 patch map is mapped back through the
full-frame letterbox transform to the original frame. Blue marks lower and red
marks higher positive attribution on one color scale shared by every frame;
blue does not mean no spill. `spatial_final_block_maps.pt` keeps positive,
signed, and raw-attention maps, while `attribution.json` records numerical
replay errors and the target.

This is a **final-block adaptation**, not the paper's all-layer LeGrad or an
OpenCLIP model. The backbone is frozen, so its spatial attention did not learn
from the spill labels. The overlay is target-conditioned patch sensitivity,
not learned spatial attention of the trajectory adapter, a validated spill
mask, pixel probability, or localization result. The trajectory adapter sees
one spatially averaged vector per tubelet, so it can learn which times matter
but cannot identify an image patch on its own.

To add the temporal MP4 to an attribution directory made by an older version
of the script, reuse its saved scores without rerunning the encoder:

```bash
python scripts/render_temporal_attention_video.py \
  --attribution-dir reports/attention_folderImage6
```

For attribution of a video already in a feature cache, reuse its features:

```bash
python scripts/attribute_trajectory.py \
  --checkpoint reports/your_run/checkpoint.pt \
  --cache reports/your_run/causal_features.pt \
  --video-id YOUR_VIDEO_ID --target state --timestep 20 --component 0 \
  --output reports/your_run/attribution_state20
```

`--target` also accepts `dynamics` (requires `--component`),
`future_prediction` (requires `--component` and `--future-timestep` at a
1, 2, or 4 tubelet horizon), `future_error` (requires `--future-timestep`),
and `semantic_similarity` (requires `--reference` or `--prototype-class`).
The prototype from `representations.pt` is a fitted-source reference, not a
held-out estimate. `--full-maps` saves raw Transformer attention, signed
gradients, and relevance maps for one selected video; regular results save
linear-size summaries. `--layers`, `--heads`, and `--max-attention-elements`
control capture size. `--faithfulness` runs a high/low relevance feature
replacement check. `--all-videos` partitions sources across ranks with
`DistributedSampler` and removes its padded duplicates; only rank 0 writes
the consolidated output. Launch with `torchrun --nproc_per_node=2` for two
GPUs. Storyboards decode selected original video frames on CPU using
TorchCodec. Each output has `attribution.json`, `timeline.png`,
`temporal_heatmap.png`, and `storyboard.png`; `--full-maps` also adds a
causal query/key heatmap and tensor file.

For a data-free smoke example:

```bash
python scripts/example_temporal_attribution.py --output /tmp/synthetic_temporal_attribution
```

The default `positive_gradient` rule computes `ReLU(dy/dA)` for a chosen
scalar `y` and the post-softmax temporal attention `A`, then averages valid
query positions, selected heads, and selected layers. `attention_gradient`
instead computes `ReLU(A * dy/dA)` before averaging. Signed `dy/dA` is kept
in the optional full-map tensor file. These are target-specific temporal
sensitivity summaries, not attention rollout. The default adapts the
positive-gradient aggregation from [LeGrad (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Bousselham_LeGrad_An_Explainability_Method_for_Vision_Transformers_via_Feature_Formation_ICCV_2025_paper.html)
and its [official code](https://github.com/WalBouss/LeGrad); it uses the final
trajectory scalar instead of LeGrad's intermediate ViT feature classifiers,
so it is LeGrad-inspired rather than an exact reproduction.

Attribution metadata includes positive and negative gradient mass even when
full attention tensors are omitted. Positive clipping can produce an entirely
empty map from nonzero negative gradients. Such a map does not establish that
the model ignores time; attention-edge gradients also omit residual paths.
The [80-source research diagnosis](reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/research_diagnosis/report.md)
separates measured recognition from the unvalidated localization claim.

Timestep targets use only the observed feature prefix. Each feature itself
comes from a separately encoded V-JEPA 2 prefix and averages the final
two-frame tubelet's spatial tokens. Thus a token can depend on earlier
observations, but not later frames. Source frame indices and timestamps are
saved in new feature caches. Older caches can be used only when their
deterministic sampling metadata reconstructs indices that match timestamps.
The plots show which temporal feature tokens are sensitive to the selected
scalar; they cannot establish pixel-level spill location. Perturbing a frozen
feature can be out of distribution and is evidence of faithfulness, not proof
of physical causality.

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

## 7. Run the balanced two-GPU experiment

Run from `/home/olemou/video-RepLearning` with the project dependencies installed.
The local checkpoint is `/home/olemou/hf`. Each command needs a new output path;
existing experiment outputs are preserved.
For the current single-command cluster workflow, GPU checks, staged execution,
and visualization, use [cluster.md](cluster.md).
The runner supports the default 64-frame sampling and optional shorter clips
for separate smoke experiments. It uses FFmpeg decoding when TorchCodec is
unavailable and saves a Pillow `curves.png` when matplotlib is unavailable.
Use `--videos-per-class N` to select exactly `N` byte-unique sources from each
class and `--learning-rate RATE` to override its fixed-rate training default
of `0.001`. These settings are recorded in the resumable pipeline contract.

```bash
cd /home/olemou/video-RepLearning

python scripts/prepare_balanced_videos.py \
  --manifest /home/olemou/videos/manifest.jsonl \
  --dataset-root /home/olemou/videos \
  --output-dir reports/experiments/balanced_78_clean_78_spill_NEW

torchrun --standalone --nproc_per_node=2 scripts/extract_causal_video_features.py \
  --manifest reports/experiments/balanced_78_clean_78_spill_NEW/manifest.jsonl \
  --dataset-root /home/olemou/videos \
  --checkpoint-dir /home/olemou/hf \
  --output reports/experiments/balanced_78_clean_78_spill_NEW/causal_features.pt \
  --thermal-views 8

torchrun --standalone --nproc_per_node=2 scripts/train_trajectory.py \
  --cache reports/experiments/balanced_78_clean_78_spill_NEW/causal_features.pt \
  --output reports/experiments/balanced_78_clean_78_spill_NEW/training \
  --epochs 100 \
  --seed 239 \
  --learning-rate 0.001 \
  --curriculum-epochs 20 \
  --align-weight 0.1 \
  --predict-weight 1.0 \
  --keep-weight 0.1
```

Training atomically replaces `training/checkpoint.pt` after every completed
epoch. To continue an interrupted run, launch the same command with the same
cache, output directory, world size, seed, learning rate, curriculum, and loss
weights, adding `--resume`. `--epochs` is the **total target epoch**, so a run
stopped after epoch 30 can continue to epoch 100 with `--epochs 100 --resume`.
The checkpoint restores the model, AdamW state, per-rank RNG state, recorded
history, and view exposures. The single optimizer reset before update 21 still
occurs exactly once. If the last update completed but report writing was
interrupted, rerun with the same epoch target and `--resume` to finish outputs.
Only checkpoints written in this resumable format support continuation.

The builder writes a manifest and provenance JSON with the source hashes and
selected class counts. Extraction strictly rejects duplicate bytes and writes
label schema, local checkpoint hash, and causal-prefix metadata. Extraction
assigns disjoint sources by global rank. Training uses `DistributedSampler` with
the global rank and world size and calls `set_epoch(epoch)` before every update.
Sampler repeats added to fill uneven shards are removed before the forward pass;
an all-rank audit checks that every source appears exactly once. DDP synchronizes
gradients, and the complete set of rank outputs enters ViCo, temporal losses,
and fixed-view evaluation. Only rank zero writes logs, the report, checkpoint,
representations, and plots. The report records per-rank source counts and the
number of removed sampler repeats for each epoch.

For multiple nodes, launch both distributed commands with the same `--nnodes`,
`--nproc_per_node`, rendezvous address, and port on every node, setting each
node's distinct `--node_rank`. `torchrun` provides the global `RANK` and
`WORLD_SIZE` used by the sampler and DDP.

One epoch is one full-batch optimizer update. Training uses AdamW with weight
decay 0, epsilon `1e-12`, and gradient clipping at 5. Optimizer moments are reset
once before update 21, after the mining curriculum. Evaluation after every
update uses the fixed identity/mirror views and final-curriculum weights, making
the curve comparable over time.

### Source-held-out learning-rate and batch search

`scripts/tune_trajectory_lr_batch.py` is a separate, bounded tuning protocol
for a full balanced causal feature cache. It holds complete source sequences
out of every optimizer update, with 16 clean and 16 spill videos in the default
selection set. Its global batch is measured in **source videos**; each source
contributes two views. The minimum candidate is 32 sources. Each source appears
at most once within an update, and sampler padding is discarded. The reported
effective batch is `per_rank_microbatch × world_size × accumulation_steps`.
The adapter evaluates the complete accumulated global batch in one ViCo and
temporal objective before DDP synchronizes the gradients.

The search compares constant, square-root, and linear learning-rate scaling
from a 32-source reference batch, then refines promising peak rates. All
candidates use AdamW with the project's epsilon, betas `(0.9, 0.999)`,
gradient clipping at 5, step-based warmup, and cosine decay. Weight decay
defaults to zero to preserve the current small-adapter protocol; it can be set
explicitly, with biases and one-dimensional parameters excluded. These choices
adapt the [official I-JEPA optimizer and schedule](https://github.com/facebookresearch/ijepa/blob/main/src/helper.py)
to the frozen-backbone trajectory adapter; I-JEPA's large-backbone learning
rate and weight-decay values are not copied into this run. Candidates have
approximately equal source-exposure budgets, which means batch size changes
optimizer steps and warmup length. Unstable
losses or gradients are pruned; the two leading candidates are repeated with
three additional seeds under the default 19-trial budget. The winner is
selected from complete full-budget trials by source-held-out validation loss,
averaging repeats when available. This set is for model selection, not an
independent test.

Run the search with a **new** output path after extracting a balanced causal
feature cache. For the standard 64-frame eight-view cache:

```bash
torchrun --standalone --nproc_per_node=2 scripts/tune_trajectory_lr_batch.py \
  --cache reports/experiments/trajectory_two_gpu_run1/causal_features.pt \
  --output reports/experiments/trajectory_lr_search_NEW \
  --accumulation-steps 2 \
  --coarse-source-budget 512 \
  --full-source-budget 1024 \
  --max-trials 19
```

See [cluster.md](cluster.md) for the complete two-GPU cluster setup. The output contains
`comparison.md`, `comparison.csv`, `report.json`, per-trial checkpoints, and
training-only `representations.pt`. A selected search checkpoint can be passed
directly to `scripts/visualize_trajectory_video.py`; its class prototype uses
only the search's training split.
Only global rank 0 writes those files. The original all-data 100-update
protocol above still uses its fixed `0.001` learning rate and has no scheduler.

**Measured exploratory search:** On two H100 GPUs, a 19-trial search over a
156-video, 12-frame identity/mirror cache used 124 training sources and 32
source-held-out validation videos. The lowest observed four-seed mean
validation loss was **0.04193** at global source batch **32** (64 view
trajectories) and peak LR **`0.0005`**; its four losses were `0.04231`,
`0.04540`, `0.04179`, and `0.03822`. At the reference batch of 32, constant,
square-root, and linear scaling give the same LR, so this run does not identify
a best scaling formula. Its mean future-prediction loss was **0.03905**, above
the frozen persistence baseline **0.03800**. The result does not establish
useful future prediction or scene generalization and does not determine the
best settings for the standard 64-frame eight-view cache.

Read the [I-JEPA source review](docs/trajectory-lr-search.md), the
[complete comparison](reports/experiments/trajectory_lr_search_real12_20261021/tuning_v5/comparison.md),
the [machine-readable report](reports/experiments/trajectory_lr_search_real12_20261021/tuning_v5/report.json),
or the [selected trial checkpoint](reports/experiments/trajectory_lr_search_real12_20261021/tuning_v5/trial_18_checkpoint.pt).

**Measured batch-aware 80-video test:** A separate 12-frame, ten-view run used
40 clean and 40 spill sources, with 8 of each held out for selection. The search
compared global source batches 32 and 64 on two H100 GPUs. Batch **32** with
peak LR **`0.0005`** had the lowest observed four-seed mean validation loss,
**0.04693** (SD `0.00378`); the batch-32 `0.001` alternative averaged `0.05064`.
The selected model completed 32 optimizer updates and rendered a spill video
from the selection split. Its mean prediction loss `0.04228` remained above the
`0.03908` persistence baseline. See the [run report and overlay links](docs/trajectory-80-batch-search.md).
The 20-video full-batch runs are separate smoke tests; their batch of 20 does
not satisfy the tuner's minimum batch of 32.

**Measured fixed-configuration GPU training:** After that search, the selected
batch and peak rate were run for 100 real optimizer updates in a new directory:
[`trajectory_80_sources_real_ddp_100updates_20261021/training`](reports/experiments/trajectory_80_sources_real_ddp_100updates_20261021/training).
The job used both H100 GPUs, 64 training sources and 16 source-held-out
validation sources. Each update used 32 distinct sources (16 per GPU, in two
microbatches of eight), followed by synchronized gradients and an AdamW step.
The [trial record](reports/experiments/trajectory_80_sources_real_ddp_100updates_20261021/training/trials.json)
records 100 checked updates, nonzero gradient norms, and an L2 parameter change
of `6.81493` on each rank. Validation total loss fell from `0.88580` before
training to `0.02783` after update 100; prediction loss was `0.02694` versus
the frozen persistence baseline `0.03908`. This validation split was already
used to choose the batch and rate, so these are continued optimization results,
not an independent test. The [checkpoint](reports/experiments/trajectory_80_sources_real_ddp_100updates_20261021/training/trial_01_checkpoint.pt)
and [training report](reports/experiments/trajectory_80_sources_real_ddp_100updates_20261021/training/report.json)
are in that new folder.

To repeat that fixed configuration with a fresh output directory, run:

```bash
torchrun --standalone --nproc_per_node=2 scripts/tune_trajectory_lr_batch.py \
  --cache reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/causal_features.pt \
  --output reports/experiments/trajectory_80_sources_real_ddp_NEW/training \
  --validation-per-class 8 --accumulation-steps 2 \
  --base-learning-rate 0.0005 --coarse-source-budget 32 \
  --full-source-budget 3200 --max-trials 1 --single-run
```

To record the loss after **every** optimizer update, add
`--record-every-update`. The [all-loss rerun](reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/)
has [`loss_history.csv`](reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/training/loss_history.csv)
for updates 0–100 and [`epoch_loss_history.csv`](reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021/training/epoch_loss_history.csv)
for all 50 complete data passes. With 64 training sources and a global source
batch of 32, one data-pass epoch contains two optimizer updates. The rerun's
model weights are identical to the prior 100-update checkpoint.

The historical 14-video run produced:

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
cd /home/olemou/video-RepLearning
LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib \
PYTHONPATH=src /home/olemou/miniconda3/envs/torchgpu/bin/python -m pytest -q
```

The suite covers loss invariants, distributed gradients and rank accounting,
augmentation behavior, causal trajectory handling, checkpoint conversion,
final-block reconstruction, FFmpeg decoding, pipeline orchestration, and
video attribution. Distributed tests need local socket access; run them on a
node where that access is available. The real two-GPU smoke run is documented
in [cluster.md](cluster.md). The latest full suite completed with **143 passed,
4 skipped**.

Historical Gaussian, shared-denominator SupCon, margin, and duplicate-containing
experiments remain under `reports/` for auditability. They are not the current
training objective. The complete rules for future changes are in `AGENTS.md`.
