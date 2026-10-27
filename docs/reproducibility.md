# Reproducibility

Every reported number must trace back to code, data, configuration, and model
artifacts. A run without the fields below cannot support the final report.

## Current state

The project writes JSON training histories, PyTorch checkpoints, Parquet
feature stores, split hashes, preprocessing hashes, audit reports, and local
MLflow evidence. `ddf run` resolves layered YAML configuration files from an
explicit project root. With tracking enabled, it records the resolved
configuration, runtime snapshot, metrics, and artifacts in a local MLflow run.

## Corrected experiment contracts

Cross-fitting uses three source-disjoint sets inside each fold. Optimizer updates
use training rows. Epoch selection uses inner validation rows. OOF export reads
the outer holdout only after selection. Checkpoints record both training and
selection sources. OOF export refuses missing provenance or overlap with either set.
Legacy checkpoints remain usable for scoring, but cannot establish a new OOF claim.

`run_program.ps1` writes corrected caches, manifests, final branches and evaluation
into separate `v2` paths. Its `-SkipCache` option explicitly reuses the original
cache. The nested runner also accepts `-CacheIndex`, `-CacheRoot`, `-CacheAudit`
and `-TrainingManifest` when corrected inputs live elsewhere.

```powershell
& ./scripts/run_crossfit_seeds.ps1 -Seeds @(17, 29, 43)
```

The runner reads the original cache and split, then writes separate nested
artifacts under `runs/nested-crossfit-v2-20261024`. Rebuild old audio caches first:
new audio training defaults to `valid_samples` masking and requires real sample
counts. Historical checkpoints default to `legacy_none`; rebuilt caches cannot
silently change their masking behavior. Use `--masking-mode legacy_none` only
for an explicit historical reproduction.

An unspecified sync start is `None`. Explicit `0.0` means the window starts at
zero. Rebuild LAV-DF caches with a new code version and new index before training.
The changed zero-start fingerprint prevents `--skip-cached` from reusing the old
window. Existing indexes still point at historical entries until rebuilt.

```bash
bash scripts/run_encoder_comparison.sh
bash scripts/train_encoder_streams.sh
```

The encoder comparison re-extracts DINOv3, PE and CLIP into a new run directory.
It records normalization, input and feature digests, source assignments, probe
arguments and per-clip scores. Three seeds vary the source split. They do not
measure fine-tuning seed variation because the encoders remain frozen.

Completion receipts bind commands, implementation files, inputs and outputs.
A file without a receipt is incomplete evidence. A mismatched receipt requires a
new run path. A failed child stops its runner. Receipts for training and export also hash
the cache NPZ payloads selected by their input manifests. Historical artifacts are retained.
The encoder comparison validates existing features before reuse.
OOF export explicitly permits reported unavailable rows with
`features export --allow-unavailable`. The head's `--drop-incomplete` option
records how many clips it omits from training. Evaluation coverage stays visible.

New cross-modal streams use temporal positions. History files record the mode,
attention heads, encoders and query steps. Legacy histories restore `none`, which
preserves their old order-blind computation. A changed architecture needs retraining.

Deep fusion records training checkpoint hashes and separate final serving hashes.
OOF deep-fusion training requires `--stream-checkpoints` pointing to a JSON mapping
of every stream name to its final checkpoint path. Holdout fusion binds the exact
checkpoints that produced its training features. Loaders reject replacements and
legacy heads with no complete binding. Fit a new head rather than adding a binding
to an old artifact after training.

## Local quality checks

```powershell
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m deepfake_detection.documentation
node --test tests/test_webapp_behavior.js
uv lock --check
```

CI runs Python 3.11, 3.12 and 3.13. Its full install includes web, dashboard and
probe extras. Separate base installs check CLI help and documentation without Torch.

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

Detector runs log aggregate reports, hashes, path-free prediction JSONL, and
resolved configurations. They do not log raw media, review images, raw
annotation JSONL, derived crops, or model binaries. Keep private input paths
relative in tracked configuration layers. Fixture detector reports use
`software_fixture_only` and cannot select a real detector.

Research detector reports require a clean Git worktree. Each report records its
source run ID and the byte SHA-256 of the pinned `uv.lock`. Research comparison
requires paired MTCNN and YuNet reports from the same environment and evidence.
The saved decision records both exact report byte hashes, both run IDs, and all
common split, sample, audit, evaluation-set, and lock hashes. A detector without
tracked frames is ineligible. An exact speed tie stays undecided because bare
downstream scalar scores are not accepted. Each report is read into one byte
buffer. Parsing and hashing both use that buffer, and callers cannot supply a
detached digest.

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
- `models/`

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

## Research training hardware

Visual, audio, and synchronization research training requires CUDA. The CLI
stops before loading a manifest when the requested device is not CUDA or CUDA
is unavailable. CPU remains valid for unit tests and software fixture smoke
runs.

The current research machine has an NVIDIA GeForce RTX 5070 Ti with 16,303 MiB
of VRAM. Each branch run records its GPU identity in the runtime snapshot. It
also logs peak allocated GPU memory, completed training examples, samples per
second, and elapsed time. Use a short measured branch run to forecast the full
experiment matrix. Do not use a hardware-only estimate as a schedule claim.

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
- Detector review images and raw annotation files stay outside Git and MLflow.
- Detector model binaries stay outside Git and MLflow. Record only their
  expected and observed SHA-256 hashes.
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
