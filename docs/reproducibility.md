# Reproducibility

Every reported number must trace back to code, data, configuration, and model
artifacts. A run without the fields below cannot support the final report.

## Current state

The project currently writes JSON training histories, PyTorch checkpoints,
Parquet feature stores, split hashes, preprocessing hashes, and audit reports.
It does not yet integrate MLflow. [The roadmap](../ROADMAP.md) makes local
MLflow the next infrastructure task.

## Local tracking decision

Use MLflow with a local SQLite metadata store and a local artifact directory.
Do not require a hosted account for the primary workflow. Keep the existing
JSON, checkpoint, and Parquet files. MLflow indexes them and records their
relationships.

The implementation must keep these local paths outside Git:

- `mlflow.db`
- `mlartifacts/`
- `mlruns/` if a command uses MLflow's fallback file store
- `artifacts/`
- `checkpoints/`
- `runs/`

The server must bind to localhost by default. Remote exposure needs separate
authentication and is outside this project.

## Required run fields

### Identity

- Run ID and experiment group.
- UTC start time.
- Git commit and dirty-worktree flag.
- Python version and platform.
- Torch, CUDA, cuDNN, FFmpeg, and model-library versions.
- CPU, GPU, and available memory.

### Data

- Dataset name and version or retrieval date.
- Manifest hash and row count.
- Split hash and partition role.
- Included manipulation families.
- Source identity count and cue-label counts.
- Preprocessing hash and cache-index hash.
- Failed-view count, coverage, and abstention reasons.

### Training

- Branch or fusion stage.
- Full resolved configuration.
- Random seed.
- Pretrained model identifier and revision.
- Optimizer, scheduler, learning rate, batch size, and maximum steps.
- Early-stopping metric, direction, and patience.
- Epoch losses and validation metrics.
- Best epoch and selection metric.
- Wall-clock time and peak memory.

### Outputs

- Checkpoint path and SHA-256 hash.
- Training history.
- Prediction file and feature-store hashes.
- Calibration model and validation threshold.
- Metrics, confidence intervals, subgroup reports, and corruption reports.
- Error or interruption details for failed runs.

## Naming

Use stable experiment names:

```text
<stage>-<dataset>-<comparison>
```

Use descriptive run names:

```text
<candidate>-seed<seed>-<short-commit>
```

Examples:

```text
detector-fakeavceleb-mtcnn-vs-yunet
yunet-seed20260824-1cbbc83
audio-fakeavceleb-wav2vec2-vs-wavlm
```

Names help navigation. Hashes remain the source of identity.

## Artifact boundaries

- Raw datasets stay outside the repository and MLflow artifact directory.
- Derived face and mouth crops follow the source dataset's license.
- Checkpoints remain local unless their training data permits distribution.
- Split manifests may enter Git only after the protocol is frozen.
- Never overwrite a checkpoint or prediction file used in a report.
- Never reuse one MLflow run for separate training attempts.

## Reproduction levels

### Smoke reproduction

A clean environment runs a tiny fixture through preprocessing, training,
feature export, fusion, and evaluation. This checks software integration.

### Result reproduction

The frozen configuration and dataset recreate the reported result within its
declared seed variation. This requires the original split and model revisions.

### Artifact verification

A reviewer can verify hashes and regenerate every table from stored prediction
files without retraining large models.

The final project must support all three levels.

## Failure rules

- Mark interrupted and non-finite runs as failed. Do not delete them.
- Log configuration errors before retrying with changed settings.
- Start a new run when any input, seed, or hyperparameter changes.
- Do not select a model from incomplete seed sets.
- Do not silently rerun a failed seed until it produces a favorable result.
