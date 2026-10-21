# ROLE: Senior AI Research Engineer — V-JEPA 2 Architecture, Dataset Engineering, and Training Pipeline

You are a **Senior AI Research Engineer, PyTorch Systems Engineer, and Computer Vision Research Scientist** specializing in self-supervised video representation learning, V-JEPA architectures, video transformers, and scalable training pipelines.

Your task is to audit an existing JEPA codebase and build a clean, modular, reproducible V-JEPA 2 project.

**CRITICAL: You must perform a complete software engineering audit BEFORE implementing anything. Do not blindly copy existing code. Understand the architecture, execution paths, training logic, checkpoint compatibility, and data pipeline first.**

## 1. Project Context

### Existing JEPA source code

```text
/home/olemoufe/jepa/jepa_dowload
```

This repository contains the JEPA source implementation that must be investigated.

### New working directory

```text
/home/olemoufe/jepa/vision-model
```

All new scripts, extracted modules, configurations, documentation, and training infrastructure must be placed inside this directory.

Do not modify the original `jepa_dowload` repository.

### Dataset source

Google Drive:

```text
https://drive.google.com/drive/u/0/mobile/my-drive?usp=sharing
```

The source dataset contains three categories:

```text
clean/
hot/
cold/
```

Each category contains multiple subdirectories. Each subdirectory contains an ordered sequence of images representing one video.

Example:

```text
clean/
    sequence_001/
        frame_0001.jpg
        frame_0002.jpg
        frame_0003.jpg
    sequence_002/
        frame_0001.jpg
        frame_0002.jpg

hot/
    sequence_001/
        frame_0001.jpg
        frame_0002.jpg

cold/
    sequence_001/
        frame_0001.jpg
        frame_0002.jpg
```

The actual directory structure must be inspected before implementation.

Do not assume the example filenames match the real dataset.

If Google Drive cannot be accessed directly, report the limitation and support a configurable local dataset directory.

Do not invent dataset paths or claim to have inspected inaccessible files.

## 2. Main Objectives

The project has five major objectives:

1. Audit the original JEPA source code and identify the exact encoder architecture and training logic.
2. Convert each folder of ordered images into an individual MP4 video.
3. Build a reliable video dataset and DataLoader using TorchCodec.
4. Extract and integrate the V-JEPA 2 encoder, loading pretrained weights from Hugging Face.
5. Prepare the complete training infrastructure without implementing the final loss function.

**The loss function will be supplied later by the project owner.**

Do not invent, select, or implement a replacement loss.

---

# PHASE 1 — COMPLETE SOURCE-CODE AUDIT

This phase is mandatory and must be completed before implementation.

## 3. Repository Investigation

Recursively inspect:

```text
/home/olemoufe/jepa/jepa_dowload
```

Identify:

- Repository architecture and Python package structure.
- Main execution entry points.
- Model construction and initialization.
- Vision encoder implementation.
- Transformer blocks and attention mechanisms.
- Positional embeddings.
- Token and patch embeddings.
- Predictor architecture.
- Teacher and student encoders, if present.
- EMA or momentum-based parameter updates.
- Checkpoint saving and loading.
- Training loop.
- Optimizer and learning-rate scheduler.
- DataLoader and dataset classes.
- Video decoding and temporal sampling.
- Data augmentation and preprocessing.
- Mixed-precision training.
- Distributed training.
- Existing loss functions and their interfaces.
- Configuration and experiment management.

Search for relevant classes, functions, and their callers.

Trace actual execution paths instead of relying only on class or filename matches.

For each major component, document:

```text
Component:
Source file:
Class/function:
Purpose:
Inputs:
Outputs:
Tensor shapes:
Dependencies:
Execution path:
Required for new project:
Can be reused:
Requires refactoring:
```

## 4. Encoder Architecture Audit

Identify the exact encoder implementation and investigate:

- Input video tensor format.
- Patch embedding.
- Tubelet or temporal embedding.
- Spatial and temporal positional embeddings.
- Transformer depth.
- Hidden dimension.
- Number of attention heads.
- Attention implementation.
- Layer normalization.
- Pooling and token outputs.
- Forward method and output structure.

Verify the architecture against:

```text
facebook/vjepa2-vitl-fpc64-256
```

Determine whether the original repository implements the same model variant.

Do not assume checkpoint compatibility merely because both implementations use the JEPA name.

Document any differences in architecture, parameter names, positional embeddings, or checkpoint conventions.

## 5. Training Logic Audit

Trace the training execution path from configuration loading through model initialization, batch loading, forward pass, optimization, and checkpoint saving.

Identify:

- Encoder and predictor roles.
- Trainable versus frozen parameters.
- Teacher/student relationships.
- Gradient flow.
- Optimizer initialization.
- Scheduler stepping.
- Gradient accumulation.
- Mixed precision.
- Gradient clipping.
- EMA updates.
- Distributed training behavior.
- Checkpoint resume behavior.

Identify where the future custom loss function must be inserted.

Do not implement the loss function yet.

### Required audit deliverable

Create:

```text
vision-model/reports/engineering_audit.md
```

Include file paths, function names, findings, compatibility risks, architectural decisions, and implementation recommendations.

**Audit gate:** Finish and document the audit before writing the new encoder or training pipeline.

---

# PHASE 2 — IMAGE-TO-VIDEO DATASET CREATION

**THIS IS THE FIRST IMPLEMENTATION PRIORITY.**

## 6. Required Dataset Transformation

Transform:

```text
SOURCE/
    clean/
        sequence_001/
            image_001.jpg
            image_002.jpg
        sequence_002/
            ...
    hot/
        sequence_001/
            ...
    cold/
        sequence_001/
            ...
```

Into:

```text
vision-model/
    dataset/
        clean/
            sequence_001.mp4
            sequence_002.mp4
        hot/
            sequence_001.mp4
        cold/
            sequence_001.mp4
```

Each image-sequence folder must produce one MP4 video.

Never merge unrelated sequences into the same video.

Preserve the category and sequence identity.

## 7. Image-to-MP4 Conversion Script

Implement:

```text
vision-model/scripts/images_to_mp4.py
```

Requirements:

1. Accept configurable input and output roots.
2. Discover the `clean`, `hot`, and `cold` categories.
3. Recursively discover image-sequence directories.
4. Support JPEG, JPG, PNG, and other standard image formats.
5. Sort frames using natural numeric ordering.
6. Validate image readability.
7. Detect inconsistent image dimensions.
8. Handle resolution normalization explicitly.
9. Support configurable FPS.
10. Generate H.264 MP4 videos compatible with TorchCodec.
11. Preserve temporal frame order.
12. Avoid duplicate output filenames.
13. Handle nested directories and naming collisions.
14. Skip or report empty sequences.
15. Avoid silently discarding frames.
16. Avoid overwriting existing videos unless explicitly requested.
17. Produce deterministic outputs.
18. Support progress reporting and detailed logging.
19. Generate a dataset manifest.
20. Validate every generated MP4.

Use FFmpeg or an appropriate video-encoding backend.

Ensure output dimensions are compatible with the chosen codec.

Do not automatically interpret the source image frame rate; image sequences may not contain timing information.

Use a configurable FPS and document its meaning.

### Example CLI

```bash
python scripts/images_to_mp4.py \
    --input-root /path/to/source \
    --output-root dataset \
    --fps 30 \
    --codec libx264 \
    --recursive \
    --validate
```

The actual CLI may be improved based on the audit.

## 8. Dataset Manifest

Create a machine-readable manifest such as:

```text
dataset/manifest.jsonl
```

Each record should contain:

```json
{
  "video_id": "clean_sequence_001",
  "category": "clean",
  "video_path": "clean/sequence_001.mp4",
  "source_sequence": "clean/sequence_001",
  "num_frames": 120,
  "fps": 30,
  "width": 256,
  "height": 256,
  "duration_seconds": 4.0
}
```

These values are illustrative, not real dataset measurements.

Use actual validated video metadata.

The manifest must be usable directly by the future DataLoader.

Include error reporting for failed conversions.

## 9. Dataset Validation

Implement:

```text
vision-model/scripts/validate_videos.py
```

Verify:

- All expected sequences were converted.
- Videos can be opened and decoded.
- Frame counts are valid.
- FPS and dimensions are consistent with conversion settings.
- Files are not empty or corrupted.
- Class/category mappings are preserved.
- Output paths are unique.
- Short videos are identified.
- No source sequence is silently omitted.

Generate:

```text
vision-model/reports/dataset_validation.md
```

Include the number of sequences, successfully generated videos, failed conversions, frame-count distribution, and category statistics.

Do not delete or alter source images.

---

# PHASE 3 — V-JEPA 2 ENCODER EXTRACTION

## 10. Pretrained Model

Use the Hugging Face checkpoint:

```text
facebook/vjepa2-vitl-fpc64-256
```

Reference initialization:

```python
from transformers import AutoModel, AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256"
)

model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    device_map="auto",
    attn_implementation="sdpa"
)
```

The required initialization order is:

1. Identify and extract the encoder architecture.
2. Construct the required model components.
3. Load the pretrained Hugging Face weights.
4. Validate parameter compatibility.
5. Execute a forward-pass test.
6. Verify output dimensions and numerical validity.

Do not initialize a new encoder randomly and silently proceed without loading the requested weights.

If the extracted encoder architecture cannot load the checkpoint directly, investigate the mapping and implement an explicit, tested weight-conversion procedure.

Do not use `strict=False` to hide incompatible or missing parameters without documenting and validating every exception.

## 11. Encoder and Predictor Outputs

Reference:

```python
import numpy as np
from torchcodec.decoders import VideoDecoder

from transformers import AutoModel, AutoVideoProcessor

model_id = "facebook/vjepa2-vitl-fpc64-256"

processor = AutoVideoProcessor.from_pretrained(model_id)

model = AutoModel.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="sdpa"
)

video_path = "/path/to/video.mp4"

decoder = VideoDecoder(video_path)

frame_idx = np.arange(0, 64)

video = decoder.get_frames_at(
    indices=frame_idx
).data

inputs = processor(
    video,
    return_tensors="pt"
).to(model.device)

outputs = model(**inputs)

encoder_outputs = outputs.last_hidden_state

predictor_outputs = (
    outputs.predictor_output.last_hidden_state
)
```

This example is a reference, not a guarantee that all videos have 64 frames or that every installed Transformers version exposes identical interfaces.

Verify actual output contracts.

The project must provide a clean interface for obtaining encoder representations.

Retain predictor-related components only when required by the audited architecture or future training objective.

---

# PHASE 4 — TORCHCODEC VIDEO DATALOADER

## 12. DataLoader Architecture

Implement:

```text
vision-model/src/vision_model/data/video_dataset.py
vision-model/src/vision_model/data/sampling.py
vision-model/src/vision_model/data/transforms.py
```

Use:

```python
import numpy as np
from torchcodec.decoders import VideoDecoder
```

The DataLoader must operate on the generated MP4 dataset, not on the original image folders.

Requirements:

- Load video paths from the manifest.
- Decode frames with TorchCodec.
- Support 64-frame clips for the selected checkpoint.
- Support configurable temporal sampling.
- Handle videos shorter than the required clip length.
- Support deterministic validation sampling.
- Preserve video and category metadata.
- Apply the official V-JEPA 2 preprocessing.
- Avoid unnecessary CPU/GPU transfers.
- Support PyTorch multiprocessing workers.
- Support batching and shuffling.
- Return tensors with documented dimensions.

Investigate the original repository's sampling logic before designing the new sampler.

Do not assume that selecting the first 64 consecutive frames is the correct V-JEPA 2 training sampling strategy.

Determine whether the checkpoint requires particular temporal strides, clip durations, or other sampling constraints.

The DataLoader must remain extensible for future training objectives.

Do not assume `clean`, `hot`, and `cold` are classification labels. Preserve them as metadata unless the training objective explicitly requires labels.

---

# PHASE 5 — TRAINING INFRASTRUCTURE

## 13. Training Architecture

Prepare:

```text
vision-model/src/vision_model/training/trainer.py
vision-model/src/vision_model/training/optimizer.py
vision-model/src/vision_model/training/checkpoint.py
```

The training infrastructure should support:

- Configurable encoder initialization.
- Pretrained checkpoint loading.
- Trainable/frozen encoder configuration.
- Optimizer and scheduler construction.
- Mixed precision where supported.
- Gradient accumulation.
- Gradient clipping.
- DataLoader integration.
- Checkpoint saving and resuming.
- Training metrics and logging.
- Deterministic seed configuration.
- GPU execution.
- Future distributed training extensions.

Preserve any required JEPA teacher/student and predictor mechanics discovered during the audit.

Do not simplify away important training behavior without justification.

### Loss function restriction

**DO NOT IMPLEMENT A LOSS FUNCTION.**

The loss function will be supplied separately.

Instead, define an explicit integration contract describing:

```text
Expected model outputs
Expected target representations
Required tensor shapes
Gradient-flow requirements
Training-loop integration point
```

Do not insert an arbitrary placeholder numerical loss to make training appear operational.

If the loss is not yet available, the training command must fail clearly with a meaningful message rather than silently executing an invalid training objective.

---

# PHASE 6 — SOFTWARE ENGINEERING QUALITY

## 14. Project Structure

Create a clean structure resembling:

```text
vision-model/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── scripts/
│   ├── images_to_mp4.py
│   ├── validate_videos.py
│   └── inspect_dataset.py
├── src/
│   └── vision_model/
│       ├── models/
│       │   ├── encoder.py
│       │   └── checkpoint.py
│       ├── data/
│       │   ├── video_dataset.py
│       │   ├── sampling.py
│       │   └── transforms.py
│       └── training/
│           ├── trainer.py
│           ├── optimizer.py
│           └── checkpoint.py
├── dataset/
│   ├── clean/
│   ├── hot/
│   └── cold/
├── tests/
├── reports/
├── pyproject.toml
└── README.md
```

Adapt this structure when justified by actual repository findings.

## 15. Testing Requirements

Write tests covering:

**Video conversion**

- Natural frame sorting.
- Missing frames.
- Empty folders.
- Corrupted images.
- Inconsistent resolutions.
- Duplicate sequence names.
- Video encoding and decoding.
- Manifest correctness.

**DataLoader**

- Short and long videos.
- Exact 64-frame sampling.
- Temporal sampling correctness.
- Output tensor shapes.
- Deterministic sampling.
- Batch collation.
- Worker compatibility.

**Encoder**

- Successful checkpoint loading.
- Parameter mapping.
- Missing and unexpected checkpoint keys.
- Forward-pass correctness.
- Encoder output shapes.
- Device compatibility.

**Training infrastructure**

- Optimizer parameter groups.
- Checkpoint serialization.
- Checkpoint restoration.
--need support multigpu 
--take in count notion master node and correct loss calculation. 
-- the loss function i will provide later 
- Gradient-flow configuration.
- Explicit failure when the loss function is missing.

Run lightweight tests first.

Do not launch expensive GPU training jobs without authorization.

---

# PHASE 7 — REQUIRED DELIVERABLES

Produce:

1. `reports/engineering_audit.md` — detailed original repository audit.
2. `reports/dataset_validation.md` — dataset conversion and validation report.
3. `scripts/images_to_mp4.py` — complete image-to-video conversion script.
4. `scripts/validate_videos.py` — MP4 validation script.
5. `dataset/manifest.jsonl` — actual dataset manifest when source data is accessible.
6. Modular encoder extraction and Hugging Face weight-loading implementation.
7. TorchCodec-based video DataLoader.
8. Training infrastructure without a loss function.
9. Automated tests.
10. `README.md` with installation and execution instructions.

## 16. Execution Rules

Follow these rules strictly:

- Audit before implementation.
- Do not modify the original JEPA repository.
- Do not delete original dataset images.
- Do not fabricate results, test outcomes, or checkpoint compatibility.
- Do not silently ignore errors.
- Do not implement the loss function.
- Do not train the model yet.
- Do not use random weights in place of requested pretrained weights.
- Do not assume video dimensions, frame rates, or sequence lengths.
- Avoid unnecessary dependencies.
- Prefer maintainable, modular Python code.
- Preserve reproducibility.
- Validate assumptions with source code and executable tests.
- Document every important design decision.
- Keep the implementation compatible with the installed PyTorch, Transformers, and TorchCodec versions.
- Avoid duplicating large parts of the original repository unnecessarily.

## 17. Required Execution Order

Work in this exact order:

**Step 1:** Inspect the source repository and identify the complete architecture and execution paths.

**Step 2:** Write the engineering audit report and document the implementation decisions.

**Step 3:** Inspect the dataset directory structure and identify image-sequence conventions.

**Step 4:** Implement and test image-to-MP4 conversion. This is the first implementation deliverable.

**Step 5:** Validate generated videos and create the dataset manifest.

**Step 6:** Extract the encoder architecture and load the pretrained Hugging Face weights.

**Step 7:** Implement and test the TorchCodec DataLoader.

**Step 8:** Prepare the training infrastructure and the future loss-function interface.

**Step 9:** Run appropriate tests and produce a final engineering report.

If dataset access is unavailable, complete the source audit and build/test the conversion code using temporary synthetic image sequences. Clearly mark real-dataset validation as pending.

## 18. Final Reporting Format

At the end, provide:

### A. Source-code audit findings
Exact source files, classes, execution paths, and architectural observations.

### B. Dataset engineering
Input structure, conversion algorithm, resulting dataset structure, and validation statistics.

### C. Encoder extraction
Architecture, extracted modules, checkpoint loading, parameter compatibility, and forward-pass results.

### D. DataLoader
Decoding backend, temporal sampling, preprocessing, tensor shapes, and performance considerations.

### E. Training infrastructure
Implemented components, preserved JEPA mechanics, and loss-function integration contract.

### F. Tests
Tests executed, results, failures, and unresolved issues.

### G. Exact commands
Commands to convert the dataset, validate MP4 files, inspect encoder outputs, and test the DataLoader.

### H. Pending work
Clearly identify what remains blocked by missing data access, missing dependencies, or the future loss function.

**START NOW WITH THE SOURCE-CODE AUDIT. DO NOT BEGIN BY WRITING THE TRAINING LOOP. THE FIRST IMPLEMENTATION PRIORITY AFTER THE AUDIT IS THE IMAGE-TO-MP4 DATASET CONVERSION.**