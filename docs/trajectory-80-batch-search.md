# Batch-aware 80-video GPU test

This run addresses the mismatch between the earlier 20-source full-batch
smoke test and the LR search's minimum global batch of 32 sources. It is a
separate experiment on 80 byte-unique real videos: 40 clean and 40 spill.
V-JEPA 2 stayed frozen; its genuine past-only features were extracted on two
H100 GPUs from 12 sampled frames per source. The cache contains identity,
whole-clip mirror, and eight fixed seeded thermal views.

## Search and selection

The source-group split held out 8 videos per class for model selection, leaving
64 training sources. Every optimizer update used a class-balanced global batch
of 32 or 64 **distinct source videos**, two views per source, across both GPUs.
With two accumulation chunks, the global source batch of 32 used 8 sources per
rank per chunk. All-rank checks verified distinct sources and excluded any
`DistributedSampler` padding. The full-budget comparison gave every candidate
about 1024 source exposures: 32 updates at batch 32 or 16 at batch 64.

AdamW used betas `(0.9, 0.999)`, epsilon `1e-12`, zero weight decay, gradient
clipping at 5, update-wise warmup and cosine decay. This bounded search tested
constant, square-root, and linear LR scaling, refined the best regions, and
repeated the two leading full-budget settings with three more seeds. It
completed 16 trials within the 19-trial limit.

| Configuration | Validation total loss |
|---|---:|
| Batch 32, peak LR `0.0005`, 4 seeds | **0.04693 mean**, SD 0.00378 |
| Batch 32, peak LR `0.001`, 4 seeds | 0.05064 mean, SD 0.00726 |
| Batch 64, best completed trial, peak LR `0.002828` | 0.09268, one seed |

The selected checkpoint is trial 11, the lowest individual validation loss
among the winning batch-32/LR-`0.0005` configuration. It completed 32 updates
and saw 1024 source exposures. At reference batch 32, all three LR-scaling
rules yield the same numerical rate, so this experiment does not identify a
best scaling formula. The two repeated batch-32 configurations overlap across
seeds; `0.0005` is the lower **observed mean**, not a guaranteed optimum.

The selected configuration's mean validation future-prediction loss was
`0.04228`, above the frozen persistence baseline `0.03908`. Thus the run does
not demonstrate useful trajectory prediction on this selection set. The
selection sources are not an independent final test, and source groups do not
establish scene independence.

## Spill visualization

The video `cold/folderImage13.mp4` was in the **selection split**, outside all
training updates. Its class-1 prototype was computed only from the 32 spill
sources in the training split. The chosen checkpoint produced both a temporal
attention video and a frozen V-JEPA final-block spatial patch-sensitivity
overlay. The 12-frame MP4s decode at 4 fps. The attribution metadata records
`source_matches_training_cache=false` and an encoder SHA-256 match. Red denotes
higher positive patch sensitivity on the shared scale and blue lower
sensitivity; neither color is a spill probability or a validated spill mask.

## Reproduce on the two-GPU node

Use a **new** experiment directory for another run. With the `RUN_PYTHON` and
`LD_LIBRARY_PATH` setup in [cluster.md](../cluster.md):

```bash
RUN_DIR=reports/experiments/trajectory_80_sources_REPRO

"$RUN_PYTHON" scripts/prepare_balanced_videos.py \
  --manifest /home/olemou/videos/manifest.jsonl \
  --dataset-root /home/olemou/videos \
  --output-dir "$RUN_DIR/balanced" --seed 239 --per-class 40

"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/extract_causal_video_features.py \
  --manifest "$RUN_DIR/balanced/manifest.jsonl" \
  --dataset-root /home/olemou/videos --checkpoint-dir /home/olemou/hf \
  --thermal-views 8 --clip-length 12 --output "$RUN_DIR/causal_features.pt"

"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/tune_trajectory_lr_batch.py \
  --cache "$RUN_DIR/causal_features.pt" --output "$RUN_DIR/tuning" \
  --validation-per-class 8 --accumulation-steps 2 \
  --coarse-source-budget 512 --full-source-budget 1024 --max-trials 19
```

Read `tuning/report.json` for `selected_checkpoint`, then pass that checkpoint
to `scripts/visualize_trajectory_video.py` with `--target semantic_similarity
--prototype-class 1`. The tuner saves a training-only `representations.pt`
beside its checkpoints for this purpose.

Artifacts:

- [Comparison](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/tuning/comparison.md)
- [Full report](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/tuning/report.json)
- [Selected checkpoint](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/tuning/trial_11_checkpoint.pt)
- [Training-only representations](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/tuning/representations.pt)
- [Spill spatial overlay](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/visualization_spill_folderImage13/spatial_final_block_overlay.mp4)
- [Spill temporal overlay](../reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/visualization_spill_folderImage13/temporal_attention_video.mp4)

For the standard 64-frame protocol, extract a new cache and rerun the search;
the measured 12-frame rates and convergence cannot be assumed to transfer.
