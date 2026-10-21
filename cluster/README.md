# Real cluster execution from pretrained V-JEPA weights

This guide starts with a fresh cluster workspace where the pretrained
`facebook/vjepa2-vitl-fpc64-256` checkpoint has already been downloaded from
Hugging Face. It runs the complete experiment on real MP4 videos:

The earlier 8-clean/8-spill manifest was invalid as an independent 16-video
experiment: one clean pair and one spill pair were byte-identical. The corrected
workflow uses seven unique clean and seven unique spill videos. Its encoder
and head stages are real training, but held-out video prediction must be
reported separately from the contrastive training loss.

```text
pretrained V-JEPA weights + real videos
    -> real frozen-encoder inference
    -> Gaussian video-head training
    -> video loss curve and checkpoint verification
    -> real spatial-token extraction
    -> human spill-mask annotation
    -> mask-supervised segmentation training
    -> segmentation loss curve and heatmaps
```

No random or synthetic encoder features are used. The pretrained encoder is
frozen, while the Gaussian video heads and segmentation decoder receive real
gradients.

## 1. Required files on shared cluster storage

The compute node must be able to read all of these:

```text
vision-model/                         # this repository
vision-model/dataset/                 # real MP4 files
vision-model/dataset/manifest.jsonl   # base video manifest
Hugging Face cache containing:
  facebook/vjepa2-vitl-fpc64-256
```

The corrected experiment uses only videos in the base manifest. The additional
`cold/scene_4.mp4` is an exact duplicate of `warm/scene_1_hot.mp4` and is
excluded.

Keep the repository, videos, Hugging Face cache, checkpoints, and reports on a
persistent shared filesystem. Do not store final artifacts only in a node's
temporary directory.

## 2. Define cluster paths

Replace the example paths with paths valid on your cluster:

```bash
export PROJECT_DIR=/shared/projects/jepa/vision-model
export HF_CACHE_DIR=/shared/cache/huggingface
export EXP_DIR=$PROJECT_DIR/reports/experiments/unique_7_clean_7_spill

cd "$PROJECT_DIR"
```

`HF_CACHE_DIR` is the directory that contains the Hugging Face `hub/` cache.
The model must be visible from the compute node, not only from your laptop.

## 3. Create the Python environment

Use Python 3.11-3.13, CUDA PyTorch 2.7, FFmpeg, and FFprobe. A cluster commonly
provides CUDA and Python through modules or Conda. Adapt the module names to
your site.

Example using the existing `torchgpu` Conda environment:

```bash
cd "$PROJECT_DIR"
conda activate torchgpu

python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --force-reinstall --no-deps \
  transformers==4.57.6 tokenizers==0.22.2 accelerate==1.15.0
.venv/bin/python -m pip install -e . --no-deps
```

Confirm the runtime before submitting a long job:

```bash
PYTHONPATH=src .venv/bin/python -c \
  "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

ffmpeg -version
ffprobe -version
```

## 4. Verify the downloaded checkpoint offline

Set the Hugging Face cache and force offline loading so a compute job cannot
silently depend on internet access:

```bash
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_OFFLINE=1

PYTHONPATH=src .venv/bin/python scripts/inspect_encoder.py \
  --manifest dataset/manifest.jsonl \
  --model-id facebook/vjepa2-vitl-fpc64-256 \
  --device cuda \
  --dtype bfloat16
```

This command loads and converts all pretrained encoder tensors and checks the
real encoder. A missing-cache error means the checkpoint is not present under
the `HF_HOME` seen by the compute node.

## 5. Validate the real videos

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_videos.py \
  --dataset-root dataset \
  --manifest dataset/manifest.jsonl \
  --source-root source_images \
  --categories clean cold warm \
  --short-threshold 64 \
  --decode-all
```

The correct labels are:

| Class | Categories |
|---:|---|
| 0 | `clean` |
| 1 | `cold`, `warm` |

## 6. Create the balanced experiment

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_unique_experiment.py \
  --base-manifest dataset/manifest.jsonl \
  --dataset-root dataset \
  --output "$EXP_DIR"
```

This selects seven unique clean and seven unique spill videos, validates them, writes the
balanced manifest, records checksums, and creates isolated checkpoint and
report directories.

## 7. Submit real video encoding and head training

Use one GPU for this experiment. The encoder extraction is GPU intensive, but
the cached-feature head optimizer is small. Launching multiple distributed
workers does not speed up this 14-video research runner.

Create `cluster/video_train.sbatch` with site-specific partition, account, and
environment activation:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jepa-video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=cluster/logs/video-%j.out
#SBATCH --error=cluster/logs/video-%j.err

set -euo pipefail

PROJECT_DIR=/shared/projects/jepa/vision-model
HF_CACHE_DIR=/shared/cache/huggingface
EXP_DIR=$PROJECT_DIR/reports/experiments/unique_7_clean_7_spill

cd "$PROJECT_DIR"
mkdir -p cluster/logs
export PYTHONPATH="$PROJECT_DIR/src"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi

# Real MP4 decoding and real pretrained V-JEPA inference.
.venv/bin/python scripts/simulate_loss_training.py \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --dataset-root "$PROJECT_DIR/dataset" \
  --encode-only \
  --feature-cache "$EXP_DIR/checkpoints/video_features.pt" \
  --device cuda

# Real gradient optimization of Gaussian mean and variance heads.
.venv/bin/python scripts/convergence_study.py \
  --cache "$EXP_DIR/checkpoints/video_features.pt" \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --output "$EXP_DIR/video_training" \
  --epochs 100 \
  --calibration-epochs 40 \
  --curriculum-epochs 19 \
  --seeds 239 \
  --variants adaptive_refinement \
  --lr 5.25e-4 \
  --refinement-max-iter 40 \
  --refinement-precision float64 \
  --refinement-threshold 0.001 \
  --refinement-patience 5 \
  --device cuda

.venv/bin/python scripts/plot_experiment_loss.py \
  "$EXP_DIR/video_training/adaptive_refinement_seed239.json" \
  --output "$EXP_DIR/video_training/loss_curve.png"

.venv/bin/python scripts/verify_experiment_artifacts.py \
  --cache "$EXP_DIR/checkpoints/video_features.pt" \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --folders "$EXP_DIR/video_training" \
  --output "$EXP_DIR/video_training/artifact_verification.json"
```

The `simulate_loss_training.py` filename is historical. With `--encode-only`,
the code decodes every real video and runs the actual pretrained V-JEPA encoder.
It does not generate simulated features and performs no optimization in that
command. Optimization begins in `convergence_study.py`.

Submit and monitor:

```bash
mkdir -p cluster/logs
sbatch cluster/video_train.sbatch
squeue -u "$USER"
tail -f cluster/logs/video-JOB_ID.out
```

Do not launch the same output directory from two jobs simultaneously.

## 8. Interpret video convergence

The video objective is:

```text
L_video = L_contrastive + 0.01 * L_uncertainty
```

With seven videos per class, each anchor has six same-class positives. The
multi-positive contrastive floor is about `log(6) = 1.79176`. A final raw loss
near `0.5` is not expected. Check these fields in the JSON report:

```text
contrastive_excess_loss
uncertainty_regret
last_ten_excess_slope
last_ten_uncertainty_regret_slope
training_fit_converged
```

The video run is a training-fit experiment. It does not by itself measure
generalization to new physical scenes. In the earlier 8/8 run, a seven-fold
deduplicated audit found that trained heads underperformed raw pretrained
features on held-out videos. Run `scripts/audit_video_generalization.py` on
the new cache and inspect its held-out report before selecting the heads:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/audit_video_generalization.py \
  --cache "$EXP_DIR/checkpoints/video_features.pt" \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --dataset-root "$PROJECT_DIR/dataset" \
  --output "$EXP_DIR/video_training/generalization_audit.json" \
  --epochs 24 --folds 7 --device cuda
```

## 9. Submit real spatial-token extraction

Create `cluster/spatial_tokens.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jepa-tokens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=cluster/logs/tokens-%j.out
#SBATCH --error=cluster/logs/tokens-%j.err

set -euo pipefail

PROJECT_DIR=/shared/projects/jepa/vision-model
HF_CACHE_DIR=/shared/cache/huggingface
EXP_DIR=$PROJECT_DIR/reports/experiments/unique_7_clean_7_spill

cd "$PROJECT_DIR"
mkdir -p cluster/logs
export PYTHONPATH="$PROJECT_DIR/src"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_OFFLINE=1

.venv/bin/python scripts/train_spill_localizer.py \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --dataset-root "$PROJECT_DIR/dataset" \
  --token-cache "$EXP_DIR/checkpoints/spatial_tokens.pt" \
  --device cuda \
  --token-layer patch_embed \
  --rebuild-cache \
  --encode-only
```

Submit after the experiment manifest exists:

```bash
sbatch cluster/spatial_tokens.sbatch
```

This performs real encoder inference and produces local tokens with shape
`[14,32,16,16,1024]`. It stops without running the rejected video-label MIL
localizer.

## 10. Annotate masks through an SSH tunnel

Annotation is an interactive human step and should not consume a scheduled GPU.
Run it on a permitted login/interactive CPU node that can access the shared
experiment files:

```bash
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

.venv/bin/python -m vision_model.spill annotate \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --dataset-root "$PROJECT_DIR/dataset" \
  --token-cache "$EXP_DIR/checkpoints/spatial_tokens.pt" \
  --path "$EXP_DIR/segmentation/annotations" \
  --host 127.0.0.1 \
  --port 8765
```

From your local computer, forward the port to the node running the editor:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@CLUSTER_NODE
```

Then open `http://127.0.0.1:8765` locally. Brush over each spill and press
**Save + next**. White pixels mean spill and black pixels mean background.
Complete all 138 positive-video masks. The annotations are saved immediately
on shared storage and can be resumed.

Check completeness:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_spill_annotations.py \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --annotations "$EXP_DIR/segmentation/annotations" \
  --output "$EXP_DIR/segmentation/annotation_status.json" \
  --require-complete
```

Continue only when `full_experiment_ready` is `true`.

## 11. Submit segmentation training

Create `cluster/segmentation.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jepa-segment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=cluster/logs/segment-%j.out
#SBATCH --error=cluster/logs/segment-%j.err

set -euo pipefail

PROJECT_DIR=/shared/projects/jepa/vision-model
EXP_DIR=$PROJECT_DIR/reports/experiments/unique_7_clean_7_spill

cd "$PROJECT_DIR"
mkdir -p cluster/logs
export PYTHONPATH="$PROJECT_DIR/src"

.venv/bin/python scripts/check_spill_annotations.py \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --annotations "$EXP_DIR/segmentation/annotations" \
  --require-complete

.venv/bin/python -m vision_model.spill train \
  --manifest "$EXP_DIR/manifest.jsonl" \
  --dataset-root "$PROJECT_DIR/dataset" \
  --token-cache "$EXP_DIR/checkpoints/spatial_tokens.pt" \
  --path "$EXP_DIR/segmentation/annotations" \
  --output "$EXP_DIR/segmentation/output" \
  --epochs 100 \
  --folds 4 \
  --batch-size 2 \
  --device cuda
```

Submit only after annotation is complete:

```bash
sbatch cluster/segmentation.sbatch
```

This trains a real convolutional segmentation decoder with:

```text
weighted pixel binary cross-entropy + soft Dice loss
```

Four-fold cross-validation holds out whole videos. The final model trains on
all annotations. Contrastive loss is not used in segmentation.

## 12. Required output artifacts

After the jobs finish, retain at least:

```text
$EXP_DIR/
├── manifest.jsonl
├── provenance.json
├── checkpoints/
│   ├── video_features.pt
│   └── spatial_tokens.pt
├── video_training/
│   ├── adaptive_refinement_seed239.json
│   ├── adaptive_refinement_seed239.pt
│   ├── artifact_verification.json
│   ├── loss_curve.png
│   └── loss_curve.csv
└── segmentation/
    ├── annotation_status.json
    ├── annotations/
    └── output/
        ├── training_report.json
        ├── spill_segmenter.pt
        ├── loss_curve.png
        ├── loss_curve.csv
        ├── inference_results.json
        └── videos/<video>/
            ├── spill_segmentation.png
            ├── spill_segmentation.mp4
            └── spill_segmentation.npz
```

Red in the segmentation overlays means high spill probability; blue means low
probability. Generic transformer attention is not a spill mask.

## 13. Run tests on the cluster environment

Before long jobs, run:

```bash
cd "$PROJECT_DIR"
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The current suite contains 69 tests. Tests do not launch the expensive
100-epoch experiments.

## Cluster failure checklist

- **Checkpoint not found:** verify `HF_HOME` points to the shared cache visible
  inside the compute job and keep `HF_HUB_OFFLINE=1` only after download.
- **CUDA unavailable:** inspect the Slurm GPU request, CUDA module, environment,
  `nvidia-smi`, and PyTorch CUDA build.
- **Out of memory during encoding:** request a GPU with at least 12 GB and do
  not run two encoders on the same allocated GPU.
- **Stale or incompatible cache:** remove only the affected experiment cache and
  rerun its extraction job. Caches are checked against manifest hashes, label
  schema, token layer, and padding policy.
- **Annotation editor unavailable locally:** keep it bound to `127.0.0.1` and
  correct the SSH tunnel to the node actually running the server.
- **Segmentation refuses to start:** run the annotation completeness checker and
  ensure both frames of each two-frame tubelet have saved masks.
- **Job preempted:** cached encoder outputs remain reusable after they are
  completely written. Do not treat a partially written output as valid.

Edit only the cluster-specific paths, partition, account, module-loading, and
wall-time directives. Keep the manifest, feature cache, annotations, and output
paths consistent across every stage.
