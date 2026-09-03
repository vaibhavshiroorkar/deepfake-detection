# Project handoff

## Current state

The repository has a visual-only dashboard and one saved visual development
baseline. The baseline is not a generalization result. Its recorded evaluation
uses 400 source-disjoint FakeAVCeleb development-validation rows.

The primary checkout at
`C:\Users\vaibh\Documents\GitHub\deepfake-generalization` contains ignored
`mlflow.db` and `runs\initial-20260902` evidence. Its ignored `data` directory
is empty. Earlier handoffs recorded four dataset directories and their counts,
but those directories are not present now and cannot be verified. This blocks
the real-video CUDA smoke. Do not substitute a fixture or report a real
inference pass while the raw data is absent.

## Historical dataset inventory

The following table comes from the prior verified handoff. It records the
dataset state reported at that time. It does not describe the present
filesystem.

| Dataset directory | Recorded state | Recorded detail |
|---|---|---|
| `data/Celeb-DF-v2` | Complete | 6,529 videos |
| `data/FakeAVCeleb_v1.2` | Complete | 21,544 videos |
| `data/MNW` | Complete | 67,521 Git LFS objects; pinned at `df66c459dd8b043cc7a8aeab30de8f8126710c7f` |
| `data/FaceForensics++` | Paused | 1,039 of 9,431 videos; c23 EU2; 1,000 YouTube originals and 39 actor originals |

The present `data` directory is empty. None of these raw dataset states can be
reverified now.

The active dashboard work is on `feat/multipage-teaching-dashboard` in
`.worktrees\multipage-teaching-dashboard`. Raw data, checkpoints, run output,
MLflow storage, and model artifacts remain ignored by Git.

## Evidence record

The evidence below came from the current primary-checkout files
`runs\initial-20260902\visual-initial-history.json`,
`runs\initial-20260902\visual-validation-metrics.json`,
`runs\initial-20260902\cache-audit.json`, and `mlflow.db`.

| Field | Recorded value |
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
| Fixed threshold | 0.5 |
| ROC AUC | 0.999175 |
| PR AUC | 0.999292 |
| Balanced accuracy | 0.9975 |
| F1 | 0.997494 |
| True negatives | 200 |
| True positives | 199 |
| False positives | 0 |
| False negatives | 1 |

The checkpoint SHA-256 is
`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`.
The training run ID is `4243b35e64c743b89cc33000cc9d3d3e`. The evaluation run
ID is `56182266f70a424581f763b2d3b41989`. Both records use preprocessing hash
`fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96` and
split hash
`3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c`.

These figures apply only to FakeAVCeleb development validation. They do not
measure Celeb-DF-v2, FaceForensics++, MNW, or cross-dataset performance.

## Recorded chronology

On 2026-09-02, the initial model supervisor logged a cache-build failure with
exit code 2. It restarted later that day, used the completed initial-model
cache, reported 1,595 usable training rows and 400 validation rows, then
recorded completion at 18:50:21 +05:30. The cache audit records one failed
clip, 12 unstable-face-track blockers, four audio-video-duration blockers,
and two low-face-coverage blockers across the cache attempt.

The SQLite MLflow record has three named experiments: `smoke-fixture-fusion`,
`prototype-gpu-20260902`, and `initial-baseline-20260902`. The initial baseline
has the finished training run and the finished fixed-threshold evaluation run
listed above. The historical training runtime artifact records an NVIDIA
GeForce RTX 5070 Ti, 16,302 MiB GPU memory, and CUDA package versions for
torch and torchvision. This is evidence from the recorded training run. It
does not describe the current Python environment.

The prototype experiment has two failed runs. Run
`52da7def729f415fbb43eddbad77a1b1` saved a `FileNotFoundError` for a cache
path that repeated `runs\prototype-20260902`. Run
`a8aaf6cc31144b36997dd7c3e30e607a` saved a CUDA deterministic-algorithm error
for `upsample_linear1d_backward_out_cuda`. Commit
`268957796d366a81b5ab897dd1a4f523f1dc4b11` changed sync token resizing to
deterministic nearest timestamp selection. The SQLite record also shows the
later sync prototype run `73915f8d22fc4b3eb31bf303f307cbc4` as finished. The
evidence does not record a separate narrative cause for either failure beyond
the saved exception messages.

Prototype visual, audio, sync, and fusion runs are marked `prototype_only` in
MLflow. The fusion run `7b799a76d4a74305b02742ded2033118` has dataset tag
`software_fixture`. It is not a trained research fusion model.

## Current execution environment

The primary checkout currently detects an NVIDIA GeForce RTX 5070 Ti through
`nvidia-smi`. The driver is 596.49 and the reported memory is 16,303 MiB. The
CUDA toolkit environment points to 13.2.

The primary checkout `.venv` is not a CUDA runtime now. It has
`torch 2.12.1+cpu`; `torch.version.cuda` is `None`; and
`torch.cuda.is_available()` is `False`. GPU hardware and a toolkit environment
do not make this CPU-only Torch install capable of CUDA inference.

The ignored visual checkpoint exists in the primary checkout. Its SHA-256
matches
`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`.

For a compatible NVIDIA system, use the README environment setup with the
`cu130` extra instead of `cpu`:

```powershell
uv sync --extra cu130 --extra ml --extra media --extra dashboard --group dev
```

Do not install both the `cpu` and `cu130` extras together. This handoff does
not change the current primary environment.

## Dashboard flow

The feature worktree does not have its own `.venv`. From that worktree, use
the primary checkout environment and set `PYTHONPATH` to the worktree source:

```powershell
$env:PYTHONPATH = "src"
C:\Users\vaibh\Documents\GitHub\deepfake-generalization\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

For a checkout with its own `.venv`, use the relative command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

The dashboard reads ignored `runs` artifacts from the active checkout. Launch
from the primary checkout when its saved artifacts are present. If you launch
from a worktree, place the required checkpoint, history, and metrics files in
that worktree first.

Open `http://127.0.0.1:8501`. The page order is Overview, Video input,
Preprocessing, Visual model, Prediction, Experiments, Audio branch, Sync
branch, Fusion, and Documentation.

Video input accepts one local clip. Preprocessing builds the visual view after
the user starts it. Prediction loads the frozen visual engine only after its
checkpoint hash, run ID, split hash, commit, seed, and preprocessing hash
match the dashboard defaults. Experiments reads local history and metrics JSON
only. It cross-checks shared provenance, requires the FakeAVCeleb
development-validation scope, and labels every result with that scope. It
shows the local MLflow URL `http://127.0.0.1:5000` and the two run IDs.

Audio and sync are prototype teaching pages. Full training is incomplete.
They do not load checkpoints or calculate probabilities. Fusion is locked. Its
current artifact is a software fixture, so the page does not load it or return
a fusion probability. The documentation page links only to tracked project
documents that exist.

The dashboard reports missing local evidence as an error. It does not query a
remote service, load a substitute artifact, or create a metric when the local
record is absent.

## Local MLflow

Use the primary checkout when its database and artifact root are present:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlflow.db `
  --default-artifact-root file:///C:/Users/vaibh/Documents/GitHub/deepfake-generalization/mlartifacts `
  --host 127.0.0.1 `
  --port 5000
```

Open `http://127.0.0.1:5000`. Select `initial-baseline-20260902`, then choose
training run `4243b35e64c743b89cc33000cc9d3d3e` or evaluation run
`56182266f70a424581f763b2d3b41989`.

## Verification limits and next work

The real-video CUDA smoke is blocked for two reasons. The ignored raw data
directory is empty, and the current primary Python environment has CPU-only
Torch with CUDA unavailable. The saved validation CSV and metrics JSON remain
available in the primary checkout, but they do not make a fresh real-video
inference possible.

Next work should restore and verify the raw dataset before running the frozen
manifest row through the provenance-checked CUDA path. Install the documented
CUDA environment only when that change is authorized. Confirm the checkpoint
hash before the run. Compare the authentic-row probability with
`0.006941306870430708` only after the dataset, manifest, and CUDA environment
are available.

After that, finish full audio and sync training, create genuine source-grouped
out-of-fold branch features, train fusion candidates, choose a validation-only
threshold, and run the locked external evaluations. Do not make a multimodal
or cross-dataset claim before those steps have recorded evidence.
