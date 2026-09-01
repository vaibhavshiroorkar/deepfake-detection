# Dataset, MNW, and Research History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Arrange four local datasets, acquire FaceForensics++ and MNW, replace the external evaluation protocol, preserve an auditable MLflow research trail, and prevent research branch training without CUDA.

**Architecture:** Raw data lives under the ignored top-level `data/` directory. MLflow remains the detailed run store, while versioned research documents hold preregistered comparisons and accepted result references. The training CLI rejects CPU branch training before it loads data, while direct low-level training functions remain usable by CPU unit tests.

**Tech Stack:** PowerShell, Python 3.11 to 3.13, Git LFS, PyTorch 2.12.1 with CUDA 13.0, MLflow 3.15.1, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-02-datasets-mnw-and-research-history-design.md`

## Global Constraints

- Use exactly four top-level dataset directories: `Celeb-DF-v2`, `FakeAVCeleb_v1.2`, `FaceForensics++`, and `MNW`.
- Download all FaceForensics++ video families with `c23` compression only.
- Treat MNW as evaluation-only and non-commercial. Never use it for training, validation, or model selection.
- Keep at least 50 GB free after MNW acquisition.
- Use local MLflow. Do not add W&B.
- Require CUDA for research visual, audio, and synchronization training.
- Allow CPU unit tests and software fixture smoke runs.
- Do not claim that any research model is trained until checkpoint and run evidence exists.
- Do not commit raw datasets, model checkpoints, MLflow state, or generated run outputs.

---

### Task 1: Arrange the local dataset root

**Files:**
- Move: `src/deepfake_detection/data/Celeb-DF-v2/` to `data/Celeb-DF-v2/`
- Move: `src/deepfake_detection/data/FakeAVCeleb_v1.2/` to `data/FakeAVCeleb_v1.2/`
- Keep: `src/deepfake_detection/data/download.py`

**Interfaces:**
- Consumes: the two untracked dataset trees currently inside the package.
- Produces: two verified dataset directories under the ignored top-level data root.

- [ ] **Step 1: Record source counts and resolve every move target**

Run:

```powershell
$workspace = (Resolve-Path -LiteralPath '.').Path
$sourceRoot = (Resolve-Path -LiteralPath 'src\deepfake_detection\data').Path
$targetRoot = Join-Path $workspace 'data'
$celebSource = Join-Path $sourceRoot 'Celeb-DF-v2'
$fakeavSource = Join-Path $sourceRoot 'FakeAVCeleb_v1.2'
$celebTarget = Join-Path $targetRoot 'Celeb-DF-v2'
$fakeavTarget = Join-Path $targetRoot 'FakeAVCeleb_v1.2'
$workspacePrefix = $workspace + [System.IO.Path]::DirectorySeparatorChar
$movePaths = @($celebSource, $fakeavSource, $targetRoot, $celebTarget, $fakeavTarget)
foreach ($path in $movePaths) {
  $absolute = [System.IO.Path]::GetFullPath($path)
  if (-not $absolute.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Move path escapes the workspace: $absolute"
  }
  $absolute
}
(Get-ChildItem -LiteralPath $celebSource -Recurse -File -Filter '*.mp4').Count
(Get-ChildItem -LiteralPath $fakeavSource -Recurse -File -Filter '*.mp4').Count
```

Expected: every resolved path remains inside the workspace. The source counts are 6,253 Celeb-DF-v2 MP4 files and 21,544 FakeAVCeleb MP4 files.

- [ ] **Step 2: Create the target root and move the two trees**

Run only after checking the paths printed in Step 1:

```powershell
New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
Move-Item -LiteralPath $celebSource -Destination $celebTarget
Move-Item -LiteralPath $fakeavSource -Destination $fakeavTarget
```

Expected: both moves complete on the same volume without copying media bytes.

- [ ] **Step 3: Verify counts and required metadata after the move**

Run:

```powershell
(Get-ChildItem -LiteralPath 'data\Celeb-DF-v2' -Recurse -File -Filter '*.mp4').Count
(Get-ChildItem -LiteralPath 'data\FakeAVCeleb_v1.2' -Recurse -File -Filter '*.mp4').Count
Test-Path -LiteralPath 'data\Celeb-DF-v2\List_of_testing_videos.txt'
Test-Path -LiteralPath 'data\FakeAVCeleb_v1.2\meta_data.csv'
Get-ChildItem -LiteralPath 'data\FakeAVCeleb_v1.2' -Directory | Select-Object -ExpandProperty Name
```

Expected: counts remain 6,253 and 21,544. Both metadata checks return `True`. FakeAVCeleb has the four cue folders named in its README.

- [ ] **Step 4: Confirm raw data remains ignored and package code remains visible**

Run:

```powershell
git status --short
git check-ignore -v data\Celeb-DF-v2 data\FakeAVCeleb_v1.2
Test-Path -LiteralPath 'src\deepfake_detection\data\download.py'
```

Expected: the moved dataset trees do not appear in Git status. `download.py` still exists and remains untracked until the later source commit.

### Task 2: Acquire all FaceForensics++ c23 videos

**Files:**
- Create: `data/FaceForensics++/`
- Modify: `src/deepfake_detection/data/download.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_faceforensics_download.py`

**Interfaces:**
- Consumes: the user's confirmed FaceForensics++ terms acceptance and the upstream server file lists.
- Produces: all original and manipulated `c23` video families.

- [ ] **Step 1: Confirm free space and downloader arguments**

Run:

```powershell
[math]::Round([System.IO.DriveInfo]::new('C').AvailableFreeSpace / 1GB, 2)
.venv\Scripts\python.exe src\deepfake_detection\data\download.py -h
```

Expected: at least 50 GB remains available. Help lists `all`, `c23`, `videos`, and the three server choices.

- [ ] **Step 2: Write failing transfer cleanup tests**

Create `tests/test_faceforensics_download.py`:

```python
from pathlib import Path

import pytest

from deepfake_detection.data import download


def test_download_file_removes_partial_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise OSError("transfer failed")

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fail)

    with pytest.raises(OSError, match="transfer failed"):
        download.download_file("https://example.test/video.mp4", tmp_path / "video.mp4")

    assert tuple(tmp_path.iterdir()) == ()


def test_download_file_preserves_existing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"complete")

    def unexpected(*_: object, **__: object) -> None:
        raise AssertionError("existing payload must not be downloaded")

    monkeypatch.setattr(download.urllib.request, "urlretrieve", unexpected)

    download.download_file("https://example.test/video.mp4", target)

    assert target.read_bytes() == b"complete"
```

Run:

```powershell
uv run pytest tests\test_faceforensics_download.py -v
```

Expected: the cleanup test fails because the current script leaves its temporary file.

- [ ] **Step 3: Normalize and harden the downloader**

Make `tqdm==4.67.1` a direct project dependency and run `uv lock`. Sort imports, remove unused `random` and `urllib`, replace tabs, and apply Ruff's safe formatting fixes. Keep the upstream server paths and CLI arguments unchanged.

Replace the temporary-file block in `download_file()` with:

```python
file_handle, temporary_name = tempfile.mkstemp(dir=out_dir)
os.close(file_handle)
try:
    urllib.request.urlretrieve(  # noqa: S310
        url,
        temporary_name,
        reporthook=reporthook if report_progress else None,
    )
    os.replace(temporary_name, out_file)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
```

Mark the three fixed-server `urlopen` calls with `# noqa: S310`. Fix the unused third argument in `dataset_mask_url`. Then run:

```powershell
uv run pytest tests\test_faceforensics_download.py -v
uv run ruff check src\deepfake_detection\data\download.py tests\test_faceforensics_download.py
uv run ruff format --check src\deepfake_detection\data\download.py tests\test_faceforensics_download.py
```

Expected: both tests and both static checks pass.

- [ ] **Step 4: Start the resumable download**

Run in a PTY so the terms prompt is visible:

```powershell
.venv\Scripts\python.exe src\deepfake_detection\data\download.py data\FaceForensics++ --dataset all --compression c23 --type videos --server EU
```

At the prompt, send one newline. If the EU server fails before any file completes, retry with `--server EU2`, then `--server CA`. Do not run two mirrors at once.

- [ ] **Step 5: Verify every requested family**

Run:

```powershell
$ff = 'data\FaceForensics++'
$required = @(
  'original_sequences\youtube\c23\videos',
  'original_sequences\actors\c23\videos',
  'manipulated_sequences\Deepfakes\c23\videos',
  'manipulated_sequences\DeepFakeDetection\c23\videos',
  'manipulated_sequences\Face2Face\c23\videos',
  'manipulated_sequences\FaceShifter\c23\videos',
  'manipulated_sequences\FaceSwap\c23\videos',
  'manipulated_sequences\NeuralTextures\c23\videos'
)
$required | ForEach-Object {
  $path = Join-Path $ff $_
  [pscustomobject]@{
    Path = $_
    Exists = Test-Path -LiteralPath $path
    Videos = @(Get-ChildItem -LiteralPath $path -File -Filter '*.mp4' -ErrorAction SilentlyContinue).Count
  }
}
```

Expected: all eight directories exist and every video count is greater than zero.

- [ ] **Step 6: Check for incomplete transfer files and record disk use**

Run:

```powershell
Get-ChildItem -LiteralPath 'data\FaceForensics++' -Recurse -File |
  Where-Object Extension -ne '.mp4' |
  Select-Object FullName,Length
[math]::Round((Get-ChildItem -LiteralPath 'data\FaceForensics++' -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 2)
```

Expected: no unnamed temporary files remain. If a failed download left a partial file, remove only that verified temporary file, then rerun Step 4.

- [ ] **Step 7: Commit the reproducible acquisition tool**

Run:

```powershell
git add pyproject.toml uv.lock src\deepfake_detection\data\download.py tests\test_faceforensics_download.py
git commit -m "Add resumable FaceForensics downloader"
```

### Task 3: Acquire the full MNW benchmark safely

**Files:**
- Create: `data/MNW/`

**Interfaces:**
- Consumes: the official `microsoft/MNW` Git repository and Git LFS objects.
- Produces: one pinned, complete external evaluation benchmark.

- [ ] **Step 1: Clone metadata without downloading LFS payloads**

Run:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/microsoft/MNW.git data\MNW
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
git -C data\MNW rev-parse HEAD
git -C data\MNW lfs version
```

Expected: the clone completes, prints a commit hash, and finds Git LFS.

- [ ] **Step 2: Calculate the full LFS size and enforce the 50 GB reserve**

Run:

```powershell
$lfsReport = git -C data\MNW lfs ls-files --all --json | ConvertFrom-Json
$lfsObjects = @($lfsReport.files)
$lfsBytes = ($lfsObjects | Measure-Object -Property size -Sum).Sum
$freeBytes = [System.IO.DriveInfo]::new('C').AvailableFreeSpace
$reserveBytes = 50GB
[pscustomobject]@{
  Objects = @($lfsObjects).Count
  PayloadGB = [math]::Round($lfsBytes / 1GB, 2)
  FreeGB = [math]::Round($freeBytes / 1GB, 2)
  FreeAfterGB = [math]::Round(($freeBytes - $lfsBytes) / 1GB, 2)
  Fits = ($freeBytes - $lfsBytes) -ge $reserveBytes
}
```

Expected: `Fits` is `True`. If it is `False`, stop before pulling LFS data and report the exact shortfall.

- [ ] **Step 3: Pull every MNW LFS object**

Run only when Step 2 reports `Fits = True`:

```powershell
git -C data\MNW lfs pull
```

Expected: Git LFS exits with code 0.

- [ ] **Step 4: Verify the checkout has no unresolved LFS pointers**

Run:

```powershell
git -C data\MNW lfs status
git -C data\MNW status --short
git -C data\MNW lfs fsck
Get-ChildItem -LiteralPath 'data\MNW' -Directory | Select-Object -ExpandProperty Name
```

Expected: the checkout is clean, `git lfs fsck` succeeds, and the official media directories are present.

### Task 4: Replace the external benchmark and create durable research records

**Files:**
- Create: `docs/research/experiment-matrix.md`
- Create: `docs/research/result-traceability.md`
- Create: `docs/research/findings.md`
- Modify: `docs/data-card.md`
- Modify: `docs/research-design.md`
- Modify: `docs/README.md`
- Modify: `docs/handbook/01-problem-and-research-question.md`
- Modify: `README.md`
- Modify: `ROADMAP.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: the MNW evaluation-only rule, existing model-selection protocol, and local MLflow ADR.
- Produces: one live external benchmark protocol and versioned evidence ledgers that cannot imply unrun results.

- [ ] **Step 1: Write failing documentation contract tests**

Add these constants and tests to `tests/test_documentation.py`:

```python
RESEARCH_EVIDENCE_HEADINGS = {
    "docs/research/experiment-matrix.md": (
        "Controls",
        "Fixed seeds",
        "Experiment stages",
        "Status rules",
    ),
    "docs/research/result-traceability.md": (
        "Traceability contract",
        "Result registry",
        "Acceptance rules",
    ),
    "docs/research/findings.md": (
        "Finding contract",
        "Accepted findings",
        "Superseded findings",
    ),
}


@pytest.mark.parametrize(("relative", "headings"), RESEARCH_EVIDENCE_HEADINGS.items())
def test_research_evidence_contracts(
    relative: str, headings: tuple[str, ...]
) -> None:
    assert_markdown_headings(Path.cwd(), relative, headings)


def test_live_protocol_uses_mnw_and_rejects_deepfake_eval() -> None:
    protocol_paths = (
        Path("README.md"),
        Path("ROADMAP.md"),
        Path("docs/data-card.md"),
        Path("docs/research-design.md"),
        Path("docs/handbook/01-problem-and-research-question.md"),
    )
    protocol = "\n".join(path.read_text(encoding="utf-8") for path in protocol_paths)
    assert "Deepfake-Eval" not in protocol
    assert "Microsoft-Northwestern-WITNESS" in protocol
    assert "evaluation-only" in protocol
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
uv run pytest tests\test_documentation.py -k 'research_evidence or live_protocol' -v
```

Expected: failures for the three missing files and the old Deepfake-Eval text.

- [ ] **Step 3: Write the experiment matrix**

Create `docs/research/experiment-matrix.md` with these rules:

```markdown
# Experiment matrix

## Controls

All comparisons use one frozen source split, preprocessing hash, cache index,
training budget, early-stopping rule, and evaluation implementation. MNW is
evaluation-only and cannot select a model or threshold.

## Fixed seeds

Research branch comparisons use seeds 17, 29, and 43. A candidate remains
incomplete until all three runs finish or retain an explained failure.

## Experiment stages

| ID | Stage | Candidates | Selection evidence | Status |
| --- | --- | --- | --- | --- |
| DET-01 | Detector | MTCNN, YuNet | Reviewed training-only benchmark | planned |
| VIS-01 | Visual | EfficientNet-B0 plus GRU, ConvNeXt-Tiny | Validation and method-holdout metrics | planned |
| AUD-01 | Audio | Wav2Vec2 Base, WavLM, AASIST | Validation and method-holdout metrics | planned |
| SYN-01 | Sync | Current temporal branch, SyncNet-style baseline | Offset and mismatch metrics | planned |
| FUS-01 | Fusion | Logistic regression, small MLP | Out-of-fold validation metrics | planned |
| EXT-01 | External | Frozen selected system on MNW | Locked zero-shot metrics | planned |

## Status rules

Use only `planned`, `running`, `failed`, `accepted`, or `superseded`. Add MLflow
run IDs only after runs exist. Never convert smoke fixture metrics into a
research result.
```

- [ ] **Step 4: Write traceability and findings contracts**

Create `docs/research/result-traceability.md` with an empty registry table. Use columns `Result ID`, `Paper location`, `Analysis command`, `Report SHA-256`, `Prediction SHA-256`, `MLflow run IDs`, `Decision`, and `Status`. State that a row is accepted only after all hashes resolve and the full seed set is present.

Create `docs/research/findings.md` with `Finding contract`, `Accepted findings`, and `Superseded findings` headings. State that both finding sections remain empty until an accepted registry row exists. Do not add current status prose or fixture results.

- [ ] **Step 5: Replace protocol references and update live links**

Use `Microsoft-Northwestern-WITNESS (MNW)` on first mention. Link to `https://github.com/microsoft/MNW`. State that MNW is evaluation-only and non-commercial in the data card and research design. Replace the locked Deepfake-Eval roadmap item with a locked MNW evaluation item. Change README data examples to `data\FakeAVCeleb_v1.2`.

Turn the three research evidence entries in `docs/README.md` from planned names into live links. Add one changelog entry that records the benchmark replacement, four-folder data layout, and unchanged MLflow decision.

- [ ] **Step 6: Run documentation checks**

Run:

```powershell
uv run pytest tests\test_documentation.py -v
uv run ddf-docs
rg -n -i 'Deepfake-Eval|deepfake_eval|deepfake eval' README.md ROADMAP.md CHANGELOG.md docs configs src tests -g '!docs/superpowers/**' -g '!src/deepfake_detection/data/**'
```

Expected: all documentation tests and `ddf-docs` pass. The final search returns no live protocol reference.

- [ ] **Step 7: Commit the protocol and evidence documents**

Run:

```powershell
git add README.md ROADMAP.md CHANGELOG.md docs tests\test_documentation.py
git commit -m "Replace external benchmark with MNW"
```

### Task 5: Enforce CUDA for research training and log GPU cost

**Files:**
- Modify: `src/deepfake_detection/experiments/runtime.py`
- Modify: `src/deepfake_detection/experiments/__init__.py`
- Modify: `src/deepfake_detection/experiments/training_log.py`
- Modify: `src/deepfake_detection/cli.py`
- Modify: `docs/reproducibility.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_experiment_runtime.py`
- Test: `tests/test_training_log.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: a branch-training device string and the installed PyTorch CUDA runtime.
- Produces: `require_research_cuda(device: str) -> None`, GPU-only CLI branch training, and MLflow throughput and peak-memory metrics.

- [ ] **Step 1: Write failing CUDA guard tests**

Add to `tests/test_experiment_runtime.py`:

```python
def test_research_cuda_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        runtime.require_research_cuda("cpu")


def test_research_cuda_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runtime.require_research_cuda("cuda")
```

Add the required `sys` and `SimpleNamespace` imports.

- [ ] **Step 2: Run the CUDA tests and confirm they fail**

Run:

```powershell
uv run pytest tests\test_experiment_runtime.py -k research_cuda -v
```

Expected: failure because `require_research_cuda` does not exist.

- [ ] **Step 3: Implement the CUDA guard**

Add to `src/deepfake_detection/experiments/runtime.py`:

```python
def require_research_cuda(device: str) -> None:
    import torch

    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("Research branch training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Research branch training requested CUDA, but CUDA is unavailable")
    try:
        torch.cuda.get_device_properties(requested)
    except (AssertionError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"CUDA device is unavailable: {device}") from error
```

Export it from `experiments/__init__.py`. Call it in `_binary_branch_train()` and `_sync_branch_train()` immediately after deterministic seeding and before manifest loading.

- [ ] **Step 4: Prove both branch handlers call the guard**

Extend the existing parameterized branch training test in `tests/test_cli.py`. Monkeypatch `runtime.require_research_cuda` with a function that records the device and raises a private stop exception. Assert that visual and sync handlers pass the parser default `cuda`. Keep direct `fit_binary_branch(..., device="cpu")` and `fit_sync_branch(..., device="cpu")` unit tests unchanged.

Run:

```powershell
uv run pytest tests\test_cli.py -k 'branch_training and cuda' -v
```

Expected: both branch handlers pass `cuda` to the guard.

- [ ] **Step 5: Write failing GPU cost logging tests**

Extend binary and sync log tests to pass `samples_per_second=12.5` and `peak_gpu_memory_mib=4096.0`. Expect these MLflow metrics:

```python
{
    "training.samples_per_second": 12.5,
    "training.peak_gpu_memory_mib": 4096.0,
}
```

Run:

```powershell
uv run pytest tests\test_training_log.py -k 'binary_training or sync_training' -v
```

Expected: failures because the logging functions do not accept the two values.

- [ ] **Step 6: Measure and log throughput and peak memory**

Before each branch fit, call `torch.cuda.reset_peak_memory_stats(arguments.device)`. Measure training elapsed time once. Compute:

```python
training_examples = len(train_dataset) * len(history.epochs)
samples_per_second = training_examples / elapsed_seconds
peak_gpu_memory_mib = torch.cuda.max_memory_allocated(arguments.device) / (1024**2)
```

Add these fields to the history JSON under `hardware`. Pass them to `log_binary_training()` or `log_sync_training()`. In `_log_training()`, validate both values as finite and positive, then log them as final metrics without an epoch step.

- [ ] **Step 7: Document the GPU rule and run focused tests**

Add a `Research training hardware` subsection to `docs/reproducibility.md`. State that branch training requires CUDA, CPU is limited to tests and smoke fixtures, and the current machine is an RTX 5070 Ti with 16,303 MiB VRAM. Explain that the full time forecast must use measured samples per second.

Add a changelog entry for the guard and GPU cost metrics.

Run:

```powershell
uv run pytest tests\test_experiment_runtime.py tests\test_training_log.py tests\test_cli.py -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: all focused tests and static checks pass.

- [ ] **Step 8: Commit the GPU evidence change**

Run:

```powershell
git add CHANGELOG.md docs\reproducibility.md src\deepfake_detection\cli.py src\deepfake_detection\experiments tests\test_cli.py tests\test_experiment_runtime.py tests\test_training_log.py
git commit -m "Require GPU research training"
```

### Task 6: Verify data, MLflow history, and repository state

**Files:**
- Verify: `data/`
- Verify: `mlflow.db`
- Verify: `mlartifacts/`
- Verify: the full tracked repository

**Interfaces:**
- Consumes: every prior task.
- Produces: current evidence for the final handoff and no unsupported training claim.

- [ ] **Step 1: Verify exactly four dataset directories**

Run:

```powershell
Get-ChildItem -LiteralPath 'data' -Directory | Sort-Object Name | Select-Object -ExpandProperty Name
```

Expected:

```text
Celeb-DF-v2
FaceForensics++
FakeAVCeleb_v1.2
MNW
```

- [ ] **Step 2: Verify dataset payloads and pinned MNW commit**

Repeat the count checks from Tasks 1 and 2. Run `git -C data\MNW rev-parse HEAD`, `git -C data\MNW lfs fsck`, and `git -C data\MNW status --short`. Record FaceForensics++ and MNW disk sizes in the final handoff only. Do not add a point-in-time report to Git.

- [ ] **Step 3: Verify local MLflow history without changing it**

Run a read-only SQLite query through the project environment. Report experiment names, run counts by status, and run names. Confirm that the existing runs are smoke fixtures and that no visual, audio, or sync research run exists.

- [ ] **Step 4: Run the complete verification suite**

Run:

```powershell
uv run pytest -v
uv run ruff check src tests
uv run ruff format --check src tests
uv run ddf-docs
git diff --check
git status --short
```

Expected: tests and checks pass. Git status shows no raw data. Any tracked changes are limited to the current task.

- [ ] **Step 5: Report the measured time forecast**

Report download sizes and observed transfer times. Keep the full training estimate at 8 to 24 days until branch throughput benchmarks exist. State clearly that all research models remain untrained. Do not turn fixture metrics into a research finding.
