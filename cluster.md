# Two-GPU causal trajectory pipeline on `rg31702`

This is the current V-JEPA 2 + TrajectoryModel workflow. The pretrained
`facebook/vjepa2-vitl-fpc64-256` encoder is **frozen**: it extracts genuine
past-only features from the MP4s. The trajectory adapter and its heads are
trained with DDP on two GPUs. The final video visualization explains a selected
scalar of that trained model; it does not train a spatial localizer. For model,
loss, data, and attribution details, see [README.md](README.md).

`cluster/README.md` describes an older Gaussian-head experiment and is not the
command reference for this pipeline.

## Check the allocated node

Run these commands in the terminal on the allocated GPU node. A shell prompt
containing `rg31702` identifies the host but does not prove that PyTorch can
access its GPUs.

```bash
cd /home/olemou/video-RepLearning
export LD_LIBRARY_PATH=/home/olemou/miniconda3/envs/torchgpu/lib:${LD_LIBRARY_PATH:-}
RUN_PYTHON=/home/olemou/miniconda3/envs/torchgpu/bin/python

nvidia-smi
"$RUN_PYTHON" -c 'import torch; print("CUDA GPUs:", torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

Both checks must show two available CUDA GPUs before the full command. The
runner also checks this and stops before writing a run if fewer than two GPUs
are visible. FFmpeg and FFprobe must be on `PATH`.

## Run from videos to visualization

Choose a **new** `--output-root`; existing experiments are preserved. This
command prepares a byte-unique balanced manifest, pre-encodes identity,
whole-clip mirror, and eight seeded thermal views, trains for 100 full-batch
updates, then visualizes a real video. Source feature extraction and trajectory
training both launch two global ranks with `torch.distributed.run`. One
visualization process uses one GPU because the final attribution is for one
video and needs all of its causal prefixes together.

```bash
"$RUN_PYTHON" scripts/run_trajectory_pipeline.py \
  --manifest /home/olemou/videos/manifest.jsonl \
  --dataset-root /home/olemou/videos \
  --encoder-dir /home/olemou/hf \
  --visualize-video /home/olemou/videos/cold/folderImage6.mp4 \
  --output-root reports/experiments/trajectory_two_gpu_run1 \
  --nproc-per-node 2 \
  --thermal-views 8 \
  --epochs 100 \
  --prototype-class 1
```

The class 1 target is cosine similarity to the **fitted spill prototype** from
`training/representations.pt`; change to `--prototype-class 0` to explain
similarity to the fitted clean prototype. This is a training-fit explanation,
not a held-out accuracy or spill-localization result. The frame overlay uses
blue for lower and red for higher positive patch attribution on one scale
shared across frames. Frozen V-JEPA provides patch positions; it has not
learned spatial spill attention from these labels.

Expected outputs:

```text
reports/experiments/trajectory_two_gpu_run1/
  pipeline_settings.json
  balanced/manifest.jsonl
  balanced/provenance.json
  causal_features.pt
  training/checkpoint.pt
  training/representations.pt
  training/history.csv
  training/report.json
  training/curves.png
  visualization/attribution.json
  visualization/temporal_attention_video.mp4
  visualization/spatial_final_block_overlay.mp4
  visualization/spatial_final_block_maps.pt
```

Only rank 0 writes the shared feature cache, checkpoint, training reports, and
plots. All ranks participate in feature extraction or DDP training and in
global losses and evaluation. The training report records rank source counts
and sampler padding removed per epoch. See the DDP details in [README.md](README.md).

## Tune trajectory learning rate and source batch

After extracting the **full balanced cache**, run a new selection experiment.
This command uses all two GPUs, holds out 16 clean and 16 spill source videos,
and searches global source batches of 32, 64, and the largest feasible batch.
Two views of each source enter the loss, so a global source batch of 32 means
64 trajectories. The default search budget is 19 trials: coarse candidates,
refined learning rates, then three additional seeds for the strongest candidates.

```bash
"$RUN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/tune_trajectory_lr_batch.py \
  --cache reports/experiments/trajectory_two_gpu_run1/causal_features.pt \
  --output reports/experiments/trajectory_lr_search_run1 \
  --accumulation-steps 2 \
  --coarse-source-budget 512 \
  --full-source-budget 1024 \
  --max-trials 19
```

`comparison.md`, `comparison.csv`, and `report.json` record the selected peak
learning rate, effective global source batch, scaling formula, convergence,
GPU configuration, and every trial. `trial_XX_checkpoint.pt` holds each stable
full-budget model. The report's `selected_checkpoint` identifies the selected
trial. Validation is used for selection; it is not a separate final test.
The [I-JEPA source review and measured 12-frame search](docs/trajectory-lr-search.md)
explain why its selected settings do not automatically transfer to the standard
64-frame eight-view cache.
Source exposures define the compute budget, while optimizer steps and warmup
are recomputed for each batch. Sampler padding is removed. Each rank receives
distinct source videos and participates in the global ViCo loss and DDP
gradient synchronization; only rank 0 writes the shared outputs.

The existing `train_trajectory.py` command remains the documented 100-update,
fixed-rate all-data optimization protocol. The tuner is a separate experiment
with a source-held-out selection split and warmup/cosine schedule.
For the measured 40-per-class, ten-view batch-aware test and its spill overlay,
see [the 80-video run](docs/trajectory-80-batch-search.md). The pipeline runner
also accepts `--videos-per-class` and `--learning-rate`; its full-batch training
still differs from the tuner's selected mini-batch schedule.

## Check each stage on the real data

To inspect the commands without changing files or needing GPUs, append
`--dry-run` to the full command. To execute and inspect one stage at a time,
start a new output directory with `--stop-after prepare`. Then repeat the same
command with `--resume --stop-after extract`, then `train`, then `visualize`.
Keep all paths, seed, number of GPU ranks, view count, and prototype class
identical; `--epochs` can increase when resuming training.

After `prepare`, inspect `balanced/provenance.json` for equal `selected_counts`
and removed duplicate IDs. After `extract`, inspect the feature cache metadata
for `past_only_prefix_last_tubelet_v1`, the V-JEPA checkpoint hash, and
`distributed_world_size=2`. After `train`, inspect `training/report.json` for
`distributed_world_size=2`, class counts, `optimizer_updates`, per-rank source
counts, and removed sampler padding. After `visualize`, inspect
`visualization/attribution.json` and the two MP4s.

For example, to stop after real data preparation, use the full command above
with `--stop-after prepare` added. Continue it by adding both `--resume` and
`--stop-after extract` to the same command. The default final stage is
`visualize`.

If training stops after an epoch, rerun the **same** command and output root
with `--resume`. `--epochs 100` means the total target epoch, not 100 more.
The existing checkpoint restores the optimizer, per-rank RNG states, history,
and view exposures. A completed feature cache is reused.

The full 156-source, ten-view causal-prefix extraction is expensive. For a
shorter *new* experiment, use `--thermal-views 0` and a smaller `--epochs`,
then keep those settings fixed when resuming that experiment. `--clip-length`
defaults to 64 sampled frames; a shorter value such as 12 can make a bounded
pipeline smoke test, but it changes the trajectory and is a separate experiment.
`--target-fps` defaults to 4.

## Verification status

The complete **smoke pipeline** ran on `rg31702` with two H100 80 GB GPUs and
four real, byte-unique source videos (two per class), `--clip-length 12`,
`--thermal-views 0`, and `--epochs 1`. The feature cache has shape
`[4,2,6,1024]` and records `distributed_world_size=2` and
`past_only_prefix_last_tubelet_v1`. The training report records two sources
per rank, zero sampler repeats, one optimizer update, and class counts `[2,2]`.
The two visualization MP4s each contain 12 sampled frames at 4 FPS. The
spatial map is 16×16; its final-block replay feature error was `0.05289` in
this bfloat16 run, below the script's `0.08` acceptance limit.

This verifies the executable stages and two-rank participation on real cluster
hardware. It does **not** measure convergence, held-out recognition, or spill
localization. The full 156-source, eight-thermal-view, 100-epoch experiment
above has not been run by this verification.

On this node, `torchcodec` and `matplotlib` were unavailable in `torchgpu`.
The dataset used exact selected-frame FFmpeg decoding, and the trainer saved
`curves.png` using Pillow. `curves.pdf` is produced only when matplotlib is
installed. Both fallbacks were exercised in the real smoke run.
