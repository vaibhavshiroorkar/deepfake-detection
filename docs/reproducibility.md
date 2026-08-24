# Reproducibility

Every reported number must trace back to code, data, configuration, and model
artifacts. A run without the fields below cannot support the final report.

## Current state

The project writes JSON training histories, PyTorch checkpoints, Parquet
feature stores, split hashes, preprocessing hashes, audit reports, and local
MLflow evidence. `ddf run` resolves layered YAML configuration files from an
explicit project root. With tracking enabled, it records the resolved
configuration, runtime snapshot, metrics, and artifacts in a local MLflow run.

## Tracking data and redaction

Redaction uses sensitive configuration and tag key names. Sensitive values are
excluded from parameters and tags. They are replaced in `resolved-config.yaml`
and in `failure.json` messages. This protects configured sensitive values. It
cannot detect an arbitrary secret that is not present under a sensitive key in
the configuration.

`configuration_sha256` identifies the validated, unredacted resolved
configuration. The tracked `resolved-config.yaml` artifact is redacted, so it
does not reproduce that digest by itself.

The smoke run logs byte SHA-256 hashes for payload artifacts as
`smoke.payload.<name>.sha256`. It logs the complete `smoke-report.json` byte
hash separately as `smoke.report_sha256`.

## Local tracking decision

The default tracked workflow uses MLflow with a local SQLite metadata store and
a local artifact directory. It does not require a hosted account. Existing
JSON, checkpoint, and Parquet files remain the primary outputs. MLflow indexes
their relationships. See [ADR-001](decisions/ADR-001-local-mlflow.md) for the
decision and its collaboration review trigger.

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
- Resolved configuration SHA-256 and a redacted configuration artifact.
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

A clean environment runs this local fixture command:

```powershell
uv sync --extra media --extra tracking
uv run --extra media --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
```

It creates a tracked CPU fusion smoke run and local artifacts. Its metrics
prove software integration only. They are not research findings.

### Result reproduction

The frozen configuration and dataset recreate the reported result within its
declared seed variation. This requires the original split and model revisions.

### Artifact verification

A reviewer can verify hashes and regenerate every table from stored prediction
files without retraining large models.

The final project must support all three levels.

## Failure rules

- After an MLflow run starts, an exception logs redacted `failure.json`, ends
  the run as `FAILED`, and then propagates the original exception.
- Invalid YAML, schema validation, or configured-command dispatch parsing
  occurs before tracking starts. These errors create no MLflow run.
- A nonzero configured handler becomes a failed tracked lifecycle. Its original
  exit code is returned to the caller.
- Mark interrupted and non-finite runs as failed. Do not delete them.
- Start a new run when any input, seed, or hyperparameter changes.
- Do not select a model from incomplete seed sets.
- Do not silently rerun a failed seed until it produces a favorable result.
