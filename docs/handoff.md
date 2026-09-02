# Project handoff

## Current outcome

The repository has a trained visual development baseline, a matching
visual-only dashboard mode, GPU prototypes for the implemented branches,
local MLflow history, four dataset directories, and a verified MNW checkout.
FaceForensics++ is paused and incomplete.

The visual baseline is not a final generalization result. It has only been
evaluated on a source-disjoint FakeAVCeleb validation split. The full
multimodal research matrix is not complete.

## Workspace and Git

- Primary checkout: `C:\Users\vaibh\Documents\GitHub\deepfake-generalization`
- Active worktree: `.worktrees\datasets-mnw-research-history`
- Feature branch: `feat/datasets-mnw-research-history`
- Implementation checkpoint: `268957796d366a81b5ab897dd1a4f523f1dc4b11`
- Remote YuNet archive branch: `old`

The feature branch contains the dataset, tracking, CUDA, cache-index, and sync
determinism changes. It has not been merged into `main`. Raw data, checkpoints,
caches, MLflow storage, and run outputs are ignored by Git.

The primary checkout has an untracked
`src/deepfake_detection/data/download.py`. Preserve it until it is compared
with the tracked downloader on the feature branch.

## Dataset state

All datasets are under the ignored top-level `data` directory.

| Directory | State | Verified content |
|---|---|---:|
| `data/Celeb-DF-v2` | Complete | 6,529 videos |
| `data/FakeAVCeleb_v1.2` | Complete | 21,544 videos |
| `data/MNW` | Complete | 67,521 Git LFS objects |
| `data/FaceForensics++` | Paused | 1,039 of 9,431 videos |

MNW is pinned at `df66c459dd8b043cc7a8aeab30de8f8126710c7f`.
`git lfs fsck` passes. MNW is evaluation-only and must not be used for
training, validation, threshold selection, or commercial work.

FaceForensics++ uses c23 videos from the EU2 mirror. The retained files are
1,000 YouTube originals and 39 actor originals. The downloader and supervisor
are stopped. No partial transfer file remains.

## Trained artifacts

The main development checkpoint is local and ignored:

`runs/initial-20260902/visual-initial.pt`

Its training record is:

`runs/initial-20260902/visual-initial-history.json`

| Field | Value |
|---|---:|
| Architecture | EfficientNet-B0 plus GRU |
| Training rows | 1,595 |
| Validation rows | 400 |
| Source overlap | 0 |
| Epochs | 5 |
| Best epoch | 4 |
| Final training loss | 0.0403575 |
| Final validation loss | 0.0194314 |
| Training throughput | 11.3568 samples/sec |
| Peak allocated GPU memory | 11,013.91 MiB |

Checkpoint SHA-256:

`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`

Training MLflow run:

`4243b35e64c743b89cc33000cc9d3d3e`

One-epoch prototype artifacts also exist under `runs/prototype-20260902`:

- `visual.pt`
- `audio.pt`
- `sync.pt`
- `fusion/fusion.joblib`

The fusion artifact is a software fixture. It is not a trained research fusion
model and cannot support a final multimodal claim.

## Development validation result

The visual checkpoint was evaluated at a fixed threshold of 0.5 on all 400
source-disjoint FakeAVCeleb validation rows. The report and row-level
predictions are local:

- `runs/initial-20260902/visual-validation-metrics.json`
- `runs/initial-20260902/visual-validation-predictions.csv`

Evaluation MLflow run:

`56182266f70a424581f763b2d3b41989`

| Metric | Value |
|---|---:|
| ROC AUC | 0.999175 |
| PR AUC | 0.999292 |
| Balanced accuracy | 0.9975 |
| Precision | 1.0 |
| Recall | 0.995 |
| F1 | 0.997494 |
| False-positive rate | 0.0 |
| False-negative rate | 0.005 |
| Equal-error rate | 0.0025 |
| Brier score | 0.002651 |
| Expected calibration error | 0.003411 |

The confusion counts are 200 true negatives, 199 true positives, zero false
positives, and one false negative.

These values show strong in-dataset development performance. They do not show
cross-dataset generalization. Celeb-DF-v2, FaceForensics++, and MNW have not
been evaluated with this checkpoint.

## MLflow history

MLflow uses:

- Database: `mlflow.db`
- Artifacts: `mlartifacts`
- UI: `http://127.0.0.1:5000`

Start the UI from the feature worktree without changing the installed CUDA
environment:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlflow.db `
  --default-artifact-root file:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlartifacts `
  --host 127.0.0.1 `
  --port 5000
```

Experiments currently include:

- `initial-baseline-20260902`
- `prototype-gpu-20260902`
- `smoke-fixture-fusion`

The prototype experiment retains two failed attempts and five finished runs.
The failed runs preserve the CUDA and cache-path debugging history.

## Dashboard state

The Streamlit dashboard exists at
[`src/deepfake_detection/dashboard/app.py`](../src/deepfake_detection/dashboard/app.py).
Its presentation view model has unit coverage in
[`tests/test_dashboard_view.py`](../tests/test_dashboard_view.py).

The dashboard now runs a provenance-checked visual-only mode. Its local
configuration points to `runs/initial-20260902/visual-initial.pt`,
preprocessing version `2689577`, and the fixed threshold `0.5`. The loader
checks the checkpoint SHA-256, MLflow run ID, split hash, training commit,
seed, and preprocessing hash before loading the model.

Each visual-only result names its limited evidence scope. It states that the
reported score has only been validated on a source-disjoint FakeAVCeleb
development split and does not establish cross-dataset generalization.

The dashboard does not expose multimodal artifact loading. The current fusion
fixture is not a research model and must not be used for a multimodal claim.

Start the dashboard from the feature worktree:

```powershell
.\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`.

## Verification state

The feature worktree passes:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync ddf-docs
git diff --check
```

Pytest collects 369 tests. The latest run had no failures and one skip.

The visual-only loader was also run on the RTX 5070 Ti with the saved
checkpoint and one held-out validation clip. It returned probability
`0.006948`; the stored batch evaluation contains `0.006941` for the same clip.

The feature branch is suitable to push for review. It is not suitable to label
as a finished research release.

## Remaining research work

1. Finish FaceForensics++.
2. Evaluate the visual model on Celeb-DF-v2 and FaceForensics++.
3. Train full audio and synchronization baselines.
4. Generate genuine out-of-fold branch features.
5. Train logistic and MLP fusion candidates.
6. Freeze a threshold using validation data only.
7. Run seeds 17, 29, and 43.
8. Run method-holdout and subgroup analysis.
9. Run the locked MNW evaluation once model selection is complete.

ConvNeXt, WavLM, AASIST, and SyncNet-style candidates remain planned and are
not implemented.

## Resume FaceForensics++

Run the ignored supervisor from the feature worktree:

```powershell
$script = Resolve-Path runs\prototype-20260902\download-supervisor.ps1

Start-Process powershell.exe `
  -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $script,
    "-ActiveLfsPid",
    "0"
  ) `
  -WorkingDirectory (Get-Location) `
  -WindowStyle Hidden
```

The downloader skips completed videos and resumes from the 1,039 retained
files.
