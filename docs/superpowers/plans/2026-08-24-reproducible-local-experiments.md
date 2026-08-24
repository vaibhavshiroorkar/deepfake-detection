# Reproducible Local Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one local command resolve a versioned configuration, capture the run environment, execute a deterministic CPU smoke experiment, and record its evidence in local MLflow.

**Architecture:** A small configuration package merges versioned YAML files and hashes the resolved mapping. A configured-run adapter reuses the existing argparse commands, wraps them in an optional MLflow lifecycle, and exposes the tracker to training handlers. A synthetic fusion smoke command proves the complete config, tracking, metrics, artifact, and failure path without downloading pretrained models.

**Tech Stack:** Python 3.11 through 3.13, PyYAML 6.0.3, MLflow 3.15.1, SQLite, scikit-learn, joblib, pytest, GitHub Actions, uv, Ruff

**Spec:** `docs/reproducibility.md`

## Global Constraints

- Work directly on `main` and leave a clean commit after each task.
- Keep MLflow optional under the `tracking` extra. Existing commands and imports must work without it.
- Use `sqlite:///mlflow.db` for local metadata and `mlartifacts/` for local artifacts.
- Resolve relative tracking and output paths from an explicit project root.
- Keep `mlflow.db`, `mlartifacts/`, `mlruns/`, `artifacts/`, `checkpoints/`, and `runs/` outside Git.
- Use YAML `schema_version: 1`.
- Merge mapping layers recursively. Later scalar and list values replace earlier values.
- Hash the resolved configuration as canonical JSON with sorted keys and compact separators.
- Reject unsafe YAML tags, recursive `run` dispatch, nonmapping files, unsupported schema versions, and invalid argument shapes.
- An enabled tracking run must fail clearly if MLflow is not installed. A disabled tracking run must not import MLflow.
- Record failed and interrupted runs as failed. Never delete or silently reuse them.
- Record the Git dirty flag. Do not reject a dirty tree because capture and enforcement are separate concerns.
- Keep raw media and derived face crops outside MLflow artifacts.
- Use explicit manual logging. Do not enable MLflow autologging.
- Preserve direct CLI commands. `ddf run` is an additional reproducible entry point.
- The CPU smoke command must avoid network downloads and finish in seconds.
- Update code, tests, essential operational documentation, `ROADMAP.md`, and `CHANGELOG.md` with the implementation they describe.
- Run `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv lock --check`, `uv run ddf-docs`, and `uv run pytest` before the phase gate.

---

### Task 1: Layered versioned YAML configuration

**Files:**
- Create: `src/deepfake_detection/experiments/__init__.py`
- Create: `src/deepfake_detection/experiments/configuration.py`
- Create: `configs/local.yaml`
- Create: `configs/smoke.yaml`
- Create: `tests/test_configuration.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: one or more YAML paths in lowest-to-highest precedence order.
- Produces: `ResolvedConfiguration`, `load_configuration()`, and `configuration_argv()`.

- [ ] **Step 1: Write failing merge, validation, and argv tests**

Create `tests/test_configuration.py` with focused contracts:

```python
from pathlib import Path

import pytest

from deepfake_detection.experiments.configuration import (
    configuration_argv,
    load_configuration,
)


def test_configuration_layers_merge_and_hash_deterministically(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        """
schema_version: 1
command: [smoke]
arguments:
  output-dir: runs/base
  seed: 17
tracking:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  artifact_root: mlartifacts
  experiment_name: smoke-base
  run_name: base-seed17
""".lstrip(),
        encoding="utf-8",
    )
    override.write_text(
        """
schema_version: 1
arguments:
  output-dir: runs/override
tracking:
  run_name: override-seed17
""".lstrip(),
        encoding="utf-8",
    )

    first = load_configuration((base, override))
    second = load_configuration((base, override))

    assert first.values["arguments"]["seed"] == 17
    assert first.values["arguments"]["output-dir"] == "runs/override"
    assert first.values["tracking"]["experiment_name"] == "smoke-base"
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_configuration_argv_handles_flags_lists_and_false_values(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
schema_version: 1
command: [cache, build]
arguments:
  manifest: data/manifest.csv
  methods: [faceswap, wav2lip]
  keep-leading-silence: true
  external: false
tracking:
  enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    resolved = load_configuration((path,))

    assert configuration_argv(resolved) == (
        "cache",
        "build",
        "--keep-leading-silence",
        "--manifest",
        "data/manifest.csv",
        "--methods",
        "faceswap",
        "wav2lip",
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ("- not-a-mapping\n", "mapping"),
        ("schema_version: 2\ncommand: [smoke]\narguments: {}\n", "schema"),
        ("schema_version: 1\ncommand: run\narguments: {}\n", "command"),
    ],
)
def test_configuration_rejects_invalid_contracts(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_configuration((path,))
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```powershell
uv run pytest tests\test_configuration.py -v
```

Expected: collection fails because `deepfake_detection.experiments.configuration` does not exist.

- [ ] **Step 3: Implement the deterministic configuration package**

Implement these public types and functions in `configuration.py`:

```python
@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    values: dict[str, Any]
    sources: tuple[Path, ...]
    sha256: str

    def write_yaml(self, path: Path) -> None: ...


def load_configuration(paths: Sequence[Path]) -> ResolvedConfiguration: ...
def configuration_argv(configuration: ResolvedConfiguration) -> tuple[str, ...]: ...
```

Use `yaml.safe_load()`. Copy every mapping during recursive merge. Require a nonempty sequence of configuration paths. Require `schema_version == 1`, a nonempty list of command strings, a mapping named `arguments`, and a mapping named `tracking` after all layers merge. Reject `command[0] == "run"`. Permit argument values that are strings, integers, finite floats, booleans, lists of scalar values, or `None`. Sort argument keys when building argv so the result is stable.

Compute `sha256` from:

```python
json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
```

Export the three public symbols from `experiments/__init__.py`.

- [ ] **Step 4: Add pinned YAML support and versioned local configurations**

Add `PyYAML==6.0.3` to core dependencies. Add `configs/local.yaml`:

```yaml
schema_version: 1
tracking:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  artifact_root: mlartifacts
  experiment_name: local-smoke
  run_name: fusion-smoke-seed17
  tags:
    project: deepfake-generalization
    environment: local
```

Add `configs/smoke.yaml`:

```yaml
schema_version: 1
command: [smoke]
arguments:
  output-dir: runs/smoke
  seed: 17
  samples: 32
tracking:
  experiment_name: smoke-fixture-fusion
  run_name: logistic-seed17
```

Run `uv lock`.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
uv run pytest tests\test_configuration.py -v
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
```

Add a changelog entry for versioned layered YAML configuration. Commit with:

```powershell
git add pyproject.toml uv.lock CHANGELOG.md configs src/deepfake_detection/experiments tests/test_configuration.py
git commit -m "Add versioned experiment configuration"
```

---

### Task 2: Deterministic runtime and hardware capture

**Files:**
- Create: `src/deepfake_detection/experiments/runtime.py`
- Create: `tests/test_experiment_runtime.py`
- Modify: `src/deepfake_detection/experiments/__init__.py`
- Modify: `src/deepfake_detection/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: a project root and a seed.
- Produces: `RuntimeSnapshot`, `capture_runtime()`, and `seed_everything()`.

- [ ] **Step 1: Write failing runtime and repeatability tests**

Add tests that assert:

```python
def test_runtime_snapshot_captures_identity_and_software() -> None:
    snapshot = capture_runtime(Path.cwd())

    assert snapshot.started_at_utc.endswith("+00:00")
    assert snapshot.git_commit
    assert isinstance(snapshot.git_dirty, bool)
    assert snapshot.python_version
    assert snapshot.platform
    assert "scikit-learn" in snapshot.packages
    assert snapshot.cpu


def test_seed_everything_repeats_numpy_values() -> None:
    seed_everything(23, deterministic=True)
    first = np.random.random(4)
    seed_everything(23, deterministic=True)
    second = np.random.random(4)

    np.testing.assert_array_equal(first, second)
```

Add a CLI test that monkeypatches `experiments.runtime.seed_everything` and proves visual and sync training call the shared function instead of a private CLI implementation.

- [ ] **Step 2: Run the focused tests and confirm the missing interface failure**

Run:

```powershell
uv run pytest tests\test_experiment_runtime.py tests\test_cli.py -v
```

Expected: imports fail for `RuntimeSnapshot`, `capture_runtime`, and `seed_everything`.

- [ ] **Step 3: Implement runtime capture**

Define:

```python
@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    started_at_utc: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    packages: dict[str, str | None]
    cpu: str
    gpu: str | None
    gpu_memory_mib: int | None
    available_memory_mib: int | None
    ffmpeg_version: str | None

    def as_dict(self) -> dict[str, Any]: ...


def capture_runtime(root: Path) -> RuntimeSnapshot: ...
def seed_everything(seed: int, *, deterministic: bool) -> None: ...
```

Capture the Git commit with `git rev-parse HEAD` and the dirty flag with `git status --porcelain`. Use fixed command arguments, no shell, and `check=False`. Capture versions for `torch`, `torchvision`, `mlflow`, `timm`, `transformers`, `av`, `opencv-python`, `facenet-pytorch`, and `scikit-learn` with `importlib.metadata.version()`, returning `None` for absent optional packages.

Use `platform.processor()` with `platform.machine()` as fallback. If Torch is installed and CUDA is available, use `torch.cuda.get_device_name(0)` and device properties for VRAM. Capture available memory with `GlobalMemoryStatusEx` on Windows and `os.sysconf` on POSIX. Return `None` when the platform does not expose it. Read only the first line from `ffmpeg -version`.

Seed Python, NumPy, CPU Torch, and all CUDA devices. When deterministic is true, call `torch.use_deterministic_algorithms(True)` and set cuDNN deterministic true and benchmark false. Missing Torch must not break configuration-only commands.

- [ ] **Step 4: Replace the CLI seed helper**

Delete `_seed_everything()` from `cli.py`. Import and call the shared `seed_everything(arguments.seed, deterministic=True)` in binary and sync training. Preserve the direct CLI behavior and defaults.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
uv run pytest tests\test_experiment_runtime.py tests\test_cli.py -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Update the changelog and commit:

```powershell
git add CHANGELOG.md src/deepfake_detection/experiments src/deepfake_detection/cli.py tests/test_experiment_runtime.py tests/test_cli.py
git commit -m "Capture deterministic run environments"
```

---

### Task 3: Optional local MLflow tracking adapter

**Files:**
- Create: `src/deepfake_detection/experiments/tracking.py`
- Create: `tests/test_tracking.py`
- Modify: `src/deepfake_detection/experiments/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: resolved tracking settings, a resolved configuration, and a runtime snapshot.
- Produces: `TrackingSettings`, `RunLogger`, `NullRunLogger`, and `start_tracked_run()`.

- [ ] **Step 1: Write failing disabled, success, and failure lifecycle tests**

Use a small fake MLflow module in `tests/test_tracking.py`. It must record calls to `set_tracking_uri`, `set_experiment`, `start_run`, `log_params`, `set_tags`, `log_dict`, `log_metrics`, `log_artifact`, and `end_run`.

Test these contracts:

```python
def test_disabled_tracking_does_not_import_mlflow(...) -> None: ...
def test_tracked_run_logs_config_runtime_metrics_and_finishes(...) -> None: ...
def test_tracked_run_records_failure_and_marks_run_failed(...) -> None: ...
def test_relative_sqlite_and_artifact_paths_resolve_from_project_root(...) -> None: ...
```

The failure test must raise a sentinel exception inside the context, then assert a `failure.json` artifact was logged and `end_run(status="FAILED")` was called.

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```powershell
uv run pytest tests\test_tracking.py -v
```

Expected: collection fails because `experiments.tracking` does not exist.

- [ ] **Step 3: Implement the tracking contract**

Define:

```python
@dataclass(frozen=True, slots=True)
class TrackingSettings:
    enabled: bool
    tracking_uri: str
    artifact_root: Path
    experiment_name: str
    run_name: str
    tags: dict[str, str]

    @classmethod
    def from_configuration(
        cls,
        values: Mapping[str, Any],
        *,
        root: Path,
    ) -> TrackingSettings: ...


class RunLogger(Protocol):
    @property
    def run_id(self) -> str: ...
    def log_params(self, values: Mapping[str, Any]) -> None: ...
    def log_metrics(self, values: Mapping[str, float], *, step: int | None = None) -> None: ...
    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None: ...
    def log_dict(self, values: Mapping[str, Any], artifact_file: str) -> None: ...


@contextmanager
def start_tracked_run(
    settings: TrackingSettings,
    *,
    configuration: ResolvedConfiguration,
    runtime: RuntimeSnapshot,
    mlflow_module: Any | None = None,
) -> Iterator[RunLogger]: ...
```

When disabled, yield `NullRunLogger` and do not import MLflow. When enabled and no module is injected, import `mlflow` lazily. Raise an actionable `RuntimeError` that names `uv sync --extra tracking` when the import fails.

Resolve `sqlite:///relative.db` from the project root. Resolve the artifact root, create it, and create or select the experiment with its file URI. Log the configuration hash, scalar configuration values, runtime identity tags, `resolved-config.yaml`, and `runtime.json`. Log `failure.json` with exception type and message before marking a failed run. Never log environment variables.

- [ ] **Step 4: Add the pinned optional dependency**

Add:

```toml
tracking = [
    "mlflow==3.15.1",
]
```

Run `uv lock`. Do not add MLflow to core dependencies.

- [ ] **Step 5: Verify the fake backend and a real local backend**

Run:

```powershell
uv run pytest tests\test_tracking.py -v
uv run --extra tracking pytest tests\test_tracking.py -v
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
```

Update the changelog and commit:

```powershell
git add pyproject.toml uv.lock CHANGELOG.md src/deepfake_detection/experiments tests/test_tracking.py
git commit -m "Add local MLflow run tracking"
```

---

### Task 4: Configured command execution and training evidence

**Files:**
- Create: `src/deepfake_detection/experiments/runner.py`
- Create: `src/deepfake_detection/experiments/training_log.py`
- Create: `tests/test_experiment_runner.py`
- Create: `tests/test_training_log.py`
- Modify: `src/deepfake_detection/experiments/__init__.py`
- Modify: `src/deepfake_detection/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: configuration paths, a project root, and the existing argparse parser.
- Produces: `execute_configured_run()`, the `ddf run` command, and stage-specific metric and artifact logging.

- [ ] **Step 1: Write failing configured-run tests**

Test that `execute_configured_run()`:

- loads layers in order;
- parses the derived argv with the existing parser;
- runs from the explicit project root;
- attaches a logger and resolved configuration to the target namespace;
- replaces a configured `run_id` with the MLflow run ID when the namespace has that field;
- marks nonzero handler results as failed while returning the same code;
- restores the original working directory;
- rejects a target command of `run`.

Add a parser test for:

```powershell
ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
```

- [ ] **Step 2: Write failing training logging tests**

Create pure fake histories and a fake `RunLogger`. Assert:

```python
log_binary_training(...)
log_sync_training(...)
log_fusion_training(...)
```

log stable parameter names, one metric step per epoch, best epoch, wall-clock seconds, checkpoint hash, and the expected output artifacts. Reject nonfinite metric values before passing them to MLflow.

- [ ] **Step 3: Run tests and confirm missing runner and logger failures**

Run:

```powershell
uv run pytest tests\test_experiment_runner.py tests\test_training_log.py tests\test_cli.py -v
```

Expected: imports fail for the new runner and training logging functions.

- [ ] **Step 4: Implement configured dispatch**

Define:

```python
def execute_configured_run(
    configuration_paths: Sequence[Path],
    *,
    root: Path,
    parser_factory: Callable[[], argparse.ArgumentParser],
    disable_tracking: bool = False,
) -> int: ...
```

Parse `configuration_argv(resolved)` with the existing parser. Reject a parsed handler that is the configured-run handler. Capture runtime before starting tracking. Change to `root` only for the target handler call and restore the previous directory in `finally`. Add private namespace attributes `_run_logger`, `_resolved_configuration`, and `_config_hash`. When tracking is enabled and the target namespace has `run_id`, replace it with the MLflow run ID.

Add a root `run` parser with repeatable `--config`, `--root`, and `--no-tracking`. Direct commands remain unchanged.

- [ ] **Step 5: Implement stage logging and connect handlers**

Implement:

```python
def log_binary_training(...) -> None: ...
def log_sync_training(...) -> None: ...
def log_fusion_training(...) -> None: ...
```

The binary and sync functions log training and validation loss per epoch, optimizer steps, stage flags, best epoch, checkpoint hash, elapsed wall time, resolved configuration hash, checkpoint, and history JSON. Fusion logs sample count, branch names, model kind, split hash, preprocessing hash, model artifact, and metadata JSON.

Modify the three training handlers to use the resolved configuration hash when present and otherwise preserve `hash_config(run_config)`. Direct commands receive `NullRunLogger`.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
uv run pytest tests\test_experiment_runner.py tests\test_training_log.py tests\test_cli.py tests\test_training_recipes.py -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Update the changelog and commit:

```powershell
git add CHANGELOG.md src/deepfake_detection/experiments src/deepfake_detection/cli.py tests
git commit -m "Run configured tracked experiments"
```

---

### Task 5: Tracked CPU fusion smoke run

**Files:**
- Create: `src/deepfake_detection/experiments/smoke.py`
- Create: `tests/test_smoke_experiment.py`
- Create: `tests/test_mlflow_smoke.py`
- Modify: `src/deepfake_detection/experiments/__init__.py`
- Modify: `src/deepfake_detection/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: output directory, seed, sample count, and the active run logger.
- Produces: `SmokeReport`, `run_fusion_smoke()`, and the `ddf smoke` command.

- [ ] **Step 1: Write failing deterministic smoke tests**

Require at least 16 samples and a multiple of 8. Test:

```python
def test_fusion_smoke_is_deterministic_and_writes_evidence(tmp_path: Path) -> None:
    first = run_fusion_smoke(tmp_path / "first", seed=17, samples=32)
    second = run_fusion_smoke(tmp_path / "second", seed=17, samples=32)

    assert first.metrics == second.metrics
    assert first.threshold == second.threshold
    assert first.samples == 32
    assert first.train_samples == 24
    assert first.validation_samples == 8
    assert (tmp_path / "first" / "fusion.joblib").is_file()
    assert (tmp_path / "first" / "smoke-report.json").is_file()
    assert (tmp_path / "first" / "predictions.csv").is_file()
```

Assert the synthetic rows use eight source identities, balanced labels, three branch logits, and the three current quality features. Assert the smoke report labels every metric as fixture evidence, never a research finding.

- [ ] **Step 2: Write a failing real MLflow integration test**

In `tests/test_mlflow_smoke.py`, skip only when `mlflow` is not installed. With the tracking extra installed, create temporary local and smoke YAML layers, then call:

```python
exit_code = main(
    [
        "run",
        "--root",
        str(tmp_path),
        "--config",
        str(local),
        "--config",
        str(smoke),
    ]
)
```

Use `MlflowClient` against the temporary SQLite URI. Assert one finished run, the configuration hash parameter, smoke metrics, `resolved-config.yaml`, `runtime.json`, `fusion.joblib`, `smoke-report.json`, and `predictions.csv`.

- [ ] **Step 3: Run the focused tests and confirm missing smoke failures**

Run:

```powershell
uv run pytest tests\test_smoke_experiment.py -v
uv run --extra tracking pytest tests\test_mlflow_smoke.py -v
```

Expected: imports or parser lookup fail because the smoke implementation does not exist.

- [ ] **Step 4: Implement the deterministic smoke experiment**

Define:

```python
@dataclass(frozen=True, slots=True)
class SmokeReport:
    seed: int
    samples: int
    train_samples: int
    validation_samples: int
    threshold: float
    metrics: dict[str, float]
    artifact_hashes: dict[str, str]


def run_fusion_smoke(
    output_dir: Path,
    *,
    seed: int,
    samples: int,
    logger: RunLogger | None = None,
) -> SmokeReport: ...
```

Generate eight source groups. Use six source groups for fitting and two for validation. Balance both classes inside every source group. Generate deterministic visual, audio, and sync logits plus face coverage, audio clipped, and audio-video duration delta. Fit the existing logistic `LateFusion`. Select the threshold on the fit partition with `select_balanced_accuracy_threshold()`. Evaluate the held-out fixture partition with `binary_metrics()`.

Write `fusion.joblib`, `predictions.csv`, and `smoke-report.json` atomically where practical. Hash every artifact. Log all metrics and artifacts through the active logger. Include `evidence_scope: software_fixture_only` in the report.

- [ ] **Step 5: Add the smoke command**

Add:

```text
ddf smoke --output-dir PATH --seed 17 --samples 32
```

The direct command uses `NullRunLogger`. A configured run receives the active MLflow logger. Keep `configs/smoke.yaml` unchanged from Task 1.

- [ ] **Step 6: Verify the one-command local run and commit**

Run:

```powershell
uv run pytest tests\test_smoke_experiment.py tests\test_cli.py -v
uv run --extra tracking pytest tests\test_mlflow_smoke.py -v
uv run --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
```

Delete no run evidence. The generated files are ignored and must remain available for local inspection. Update the changelog and commit:

```powershell
git add CHANGELOG.md src/deepfake_detection/experiments src/deepfake_detection/cli.py tests
git commit -m "Add tracked CPU smoke experiments"
```

---

### Task 6: CI and essential operational documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/decisions/ADR-001-local-mlflow.md`
- Create: `tests/test_ci.py`
- Modify: `README.md`
- Modify: `ROADMAP.md`
- Modify: `docs/reproducibility.md`
- Modify: `docs/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the accepted local and smoke configurations and repository commands.
- Produces: protected Windows CI, an actual local smoke recipe, and a concise record of the MLflow decision.

- [ ] **Step 1: Write failing CI contract tests**

Parse `.github/workflows/ci.yml` with `yaml.safe_load()`. Assert:

- permissions are read-only;
- the main job runs on `windows-latest`;
- checkout uses `actions/checkout@v6`;
- setup-uv uses `astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9` with uv `0.12.1`;
- setup-python uses `actions/setup-python@v6` with Python `3.11`;
- install includes `cpu`, `media`, `ml`, and `tracking`;
- the job runs lint, format, lock, documentation, tests, and the configured smoke command;
- pull requests run the documentation change contract.

- [ ] **Step 2: Run the CI test and confirm the missing workflow failure**

Run:

```powershell
uv run pytest tests\test_ci.py -v
```

Expected: failure because `.github/workflows/ci.yml` does not exist.

- [ ] **Step 3: Add the Windows CI workflow**

Use:

```yaml
name: ci

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9
        with:
          enable-cache: true
          version: "0.12.1"
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install
        run: uv sync --extra cpu --extra media --extra ml --extra tracking --group dev
      - name: Lint
        run: uv run ruff check src tests
      - name: Format
        run: uv run ruff format --check src tests
      - name: Lock
        run: uv lock --check
      - name: Documentation
        run: uv run ddf-docs
      - name: Pull request change contract
        if: github.event_name == 'pull_request'
        run: uv run ddf-docs --changed-from ${{ github.event.pull_request.base.sha }}
      - name: Tests
        run: uv run pytest
      - name: Tracked smoke
        run: uv run --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
```

- [ ] **Step 4: Update only essential docs**

Add a short root README quickstart:

```powershell
uv sync --extra tracking
uv run --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
uv run --extra tracking mlflow server --host 127.0.0.1 --backend-store-uri sqlite:///mlflow.db
```

Update `docs/reproducibility.md` from future tense to the exact implemented behavior. State that the smoke metrics prove software integration only. Add the local MLflow ADR with status `Accepted`, context, decision, alternatives, consequences, and the W&B review trigger for supervisor-hosted collaboration. Add the ADR link to `docs/README.md`.

Check the implemented Phase 1 roadmap items for MLflow, local storage, run contract, versioned configs, CPU smoke, and CI. Leave research evidence and the paused handbook items unchecked.

- [ ] **Step 5: Run the complete phase gate**

Run:

```powershell
uv sync --extra cpu --extra media --extra ml --extra tracking --group dev
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
uv run ddf-docs
uv run pytest
uv run --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml
git diff --check
git status --short
```

Expected:

- lint, format, lock, documentation, and tests pass;
- the smoke command exits zero;
- `mlflow.db`, `mlartifacts/`, and `runs/smoke/` exist locally but do not appear in Git status;
- only Task 6 product files appear before commit.

- [ ] **Step 6: Commit the implementation gate**

Add the changelog entry and commit:

```powershell
git add .github/workflows/ci.yml README.md ROADMAP.md CHANGELOG.md docs/README.md docs/reproducibility.md docs/decisions/ADR-001-local-mlflow.md tests/test_ci.py
git commit -m "Complete reproducible local experiments"
```

## Phase gate review

Before planning detector and landmark work, verify:

- `uv run --extra tracking ddf run --root . --config configs/local.yaml --config configs/smoke.yaml` succeeds from a clean checkout.
- MLflow has one finished smoke run with config and runtime artifacts.
- A failing configured handler produces a failed MLflow run.
- Direct CLI commands still work without the tracking extra.
- The resolved configuration hash reaches checkpoint metadata for configured branch runs.
- Training handlers log epoch metrics and their checkpoint and history artifacts.
- CI runs the same local quality and smoke commands.
- No local database, run directory, checkpoint, or artifact is tracked.
- Smoke metrics are labeled as fixture evidence and are not copied into research findings.
