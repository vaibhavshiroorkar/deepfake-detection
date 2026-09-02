# Deepfake Generalization

This repository tests whether three cue-specific detectors generalize better together than a visual detector alone.

The branches inspect visual artifacts, audio spoofing, and mouth-audio alignment. The final model uses calibrated late fusion. It does not issue a full verdict when required evidence is missing.

## Project documents

- [Research design](docs/research-design.md): question, protocol, ablations, and exit criteria.
- [Data card](docs/data-card.md): data contract, split policy, risks, and handling rules.
- [Roadmap](ROADMAP.md): implementation order and phase gates.
- [Model selection](docs/model-selection.md): controlled component comparisons and selection rules.
- [Reproducibility](docs/reproducibility.md): required run metadata and local tracking.
- [Threat model](docs/threat-model.md): supported threats, failure modes, and claim limits.
- [Changelog](CHANGELOG.md): material software and protocol changes.

Read the research design before changing the model or evaluation protocol. Read
the data card before adding a dataset. Update the changelog with each material
change.

## Quick start

Install Python 3.11 through 3.13, FFmpeg, and `uv`. Then run:

```powershell
uv sync --extra cpu --extra ml --extra media --extra dashboard --group dev
uv run pytest
uv run ddf --help
```

Use the `cu130` extra instead of `cpu` on a compatible NVIDIA system. Do not install both extras together.

## Local tracked smoke

From the repository root, run the local smoke fixture and then start the local
MLflow UI. Use the installed executable so `uv` does not change the CUDA
environment while starting the server:

```powershell
uv sync --extra media --extra tracking
uv run --extra media --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
$researchRoot = (Get-Location).Path.Replace('\', '/')
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri "sqlite:///$researchRoot/mlflow.db" `
  --default-artifact-root "file:///$researchRoot/mlartifacts" `
  --host 127.0.0.1 `
  --port 5000
```

The smoke metrics are software fixture evidence. They are not research
findings. Open `http://127.0.0.1:5000`, then select an experiment and run.

## Workflow

Normalize and audit source metadata:

```powershell
uv run ddf manifest build `
  --input data\FakeAVCeleb_v1.2\full_manifest.csv `
  --output artifacts\manifest.csv `
  --audit artifacts\manifest-audit.json `
  --dataset FakeAVCeleb
```

Create the frozen source-disjoint protocol:

```powershell
uv run ddf split build `
  --manifest artifacts\manifest.csv `
  --output-dir artifacts\splits `
  --dataset FakeAVCeleb `
  --seed 20260824
```

The command writes full train, validation, and test manifests. It also writes identity-strict stress subsets and `audit.json`. Commit the split manifests only after the team freezes the research protocol.

Build all branch views:

```powershell
uv run ddf cache build `
  --manifest artifacts\manifest.csv `
  --dataset-root data\FakeAVCeleb_v1.2 `
  --cache-root C:\deepfake-cache `
  --index artifacts\cache-index.csv `
  --audit artifacts\cache-audit.json `
  --dataset FakeAVCeleb `
  --device cuda `
  --code-version v1
```

The command returns exit code 2 when any clip fails. It still writes the successful index and a failure record. The audit records fusion-ready coverage, each quality blocker, and the global preprocessing hash. Pass that hash to every branch training command.

The default remains MTCNN, greedy IoU tracking, and box-relative mouth crops.
The landmark and YuNet variants are explicit configurations:

```powershell
uv run --extra media --extra tracking ddf run --root . `
  --config configs/local.yaml `
  --config configs/detectors/mtcnn-landmark.yaml
uv run ddf detector fetch-yunet `
  --report runs/detector/yunet-asset.json
uv run --extra media --extra tracking ddf run --root . `
  --config configs/local.yaml `
  --config configs/detectors/yunet-landmark.yaml
```

Edit private data and cache paths locally before running these configurations.
The reviewed detector workflow is documented in
[the CLI reference](docs/reference/cli.md). Fixture smoke evidence tests the
software path only. It cannot select a detector, tracker, or crop mode.

Train the visual and audio branches with `ddf train visual` and `ddf train audio`. Train the alignment model with `ddf train sync`. Run `uv run ddf train <branch> --help` for the complete arguments.

Build source-grouped cross-fitting manifests from the frozen training partition only. Never cross-fit over validation or test identities. Train each branch on every fold's training manifest. Export only its held-out rows to one dedicated out-of-fold store:

```powershell
uv run ddf features export `
  --manifest artifacts\fold-0-holdout.csv `
  --cache-index artifacts\cache-index.csv `
  --cache-root C:\deepfake-cache `
  --feature-store artifacts\oof-features.parquet `
  --report artifacts\fold-0-features.json `
  --dataset FakeAVCeleb `
  --run-id fold-0 `
  --partition-role oof `
  --visual-checkpoint checkpoints\fold-0-visual.pt `
  --audio-checkpoint checkpoints\fold-0-audio.pt `
  --sync-checkpoint checkpoints\fold-0-sync.pt `
  --device cuda
```

Repeat this for each fold. Do not put training-set predictions into the out-of-fold store.

Fit late fusion from out-of-fold feature rows:

```powershell
uv run ddf train fusion `
  --feature-store artifacts\oof-features.parquet `
  --output checkpoints\fusion.joblib `
  --metadata artifacts\fusion.json
```

Use `--branches` for branch and pair ablations. Use `--model mlp` for the small MLP ablation. Logistic fusion over all three branches is the primary model.

Export the locked validation or test features to a separate store. Use the matching partition role. Then score every row:

```powershell
uv run ddf features score `
  --feature-store artifacts\test-features.parquet `
  --fusion-model checkpoints\fusion.joblib `
  --output artifacts\test-predictions.csv
```

Missing evidence produces a blank fusion probability. The visual probability remains available when its branch succeeded.

Choose the decision threshold from validation predictions:

```powershell
uv run ddf threshold `
  --predictions artifacts\validation-predictions.csv `
  --output artifacts\threshold.json
```

Freeze that output before test evaluation. Then evaluate a locked prediction CSV:

```powershell
uv run ddf evaluate `
  --predictions artifacts\test-predictions.csv `
  --output artifacts\test-metrics.json `
  --threshold <validation-threshold>
```

Blank probability values count as abstentions. They remain in the coverage denominator. Add a `visual_probability` column to run the paired fusion comparison.

Run one video after all four model artifacts exist:

```powershell
uv run ddf predict video.mp4 `
  --visual-checkpoint checkpoints\visual.pt `
  --audio-checkpoint checkpoints\audio.pt `
  --sync-checkpoint checkpoints\sync.pt `
  --fusion-model checkpoints\fusion.joblib `
  --output artifacts\prediction.json `
  --threshold <validation-threshold> `
  --code-version v1 `
  --device cuda
```

Only load fusion files created by this project. Joblib files can execute code during loading.

## Dashboard

The dashboard defaults to the trained visual development baseline when its
local checkpoint is present:

```powershell
.\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The dashboard is restricted to the local host.
It uses `runs\initial-20260902\visual-initial.pt`, preprocessing version
`2689577`, and the fixed evaluation threshold of `0.5`. It verifies the exact
checkpoint hash and training provenance before loading the model.

Visual-only results state that the model has only been validated on the
source-disjoint FakeAVCeleb development split. The dashboard does not expose
multimodal artifact loading until a compatible research artifact set exists.
It shows the evidence coverage gate before any verdict.

## Repository rules

- Keep raw media, caches, checkpoints, and run outputs outside Git.
- Keep detector review images, annotations, and model binaries outside Git and
  MLflow.
- Use cue-specific labels for branch training.
- Use the global clip label only for fusion.
- Never balance validation or test data.
- Freeze thresholds before running the final test set.
- Train fusion only from out-of-fold branch predictions.
- Record every seed, split hash, preprocessing hash, and checkpoint hash.
- Treat a negative fusion result as a valid result.

## Tests

Run the complete suite:

```powershell
uv run pytest
uv run ddf-docs
```

The media integration test needs FFmpeg. It skips when FFmpeg is unavailable.
