# Multipage Teaching Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single prediction screen with a safe ten-page dashboard that teaches the current deepfake pipeline in order.

**Architecture:** A small navigation manifest defines page order and status. Focused page modules read and write one validated session-state contract. Shared runtime helpers call the existing preprocessor and provenance-checked visual inference engine, while presentation helpers keep Streamlit code consistent.

**Tech Stack:** Python 3.13, Streamlit 1.60, PyTorch 2.12 with CUDA 13.0, NumPy, pytest, Streamlit AppTest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-03-multipage-teaching-dashboard-design.md`

## Global Constraints

- Bind Streamlit to `127.0.0.1`.
- Accept no client-controlled checkpoint, model, fusion, threshold, or device values.
- Use only the frozen visual checkpoint declared by `dashboard_defaults()`.
- Keep the threshold fixed at `0.5` and show it with each result.
- Do not expose the fixture fusion artifact as research evidence.
- Store uploaded bytes only in Streamlit session state and temporary files.
- Delete every temporary file in a `finally` block.
- Use MLflow, not W&B, in current documentation and interface copy.
- Use ASCII punctuation in code, comments, documentation, and UI copy.

---

### Task 1: Navigation contract

**Files:**
- Create: `src/deepfake_detection/dashboard/navigation.py`
- Create: `tests/test_dashboard_navigation.py`

**Interfaces:**
- Produces: `PageSpec(slug: str, module: str, title: str, state: PageState)`
- Produces: `PAGES: tuple[PageSpec, ...]`
- Produces: `page_by_slug(slug: str) -> PageSpec`

- [ ] **Step 1: Write the failing navigation tests**

```python
from deepfake_detection.dashboard.navigation import PAGES, PageState, page_by_slug


def test_dashboard_pages_follow_the_pipeline_order() -> None:
    assert [page.slug for page in PAGES] == [
        "overview",
        "video-input",
        "preprocessing",
        "visual-model",
        "prediction",
        "experiments",
        "audio-branch",
        "sync-branch",
        "fusion",
        "documentation",
    ]


def test_unfinished_research_pages_have_honest_states() -> None:
    assert page_by_slug("audio-branch").state is PageState.PROTOTYPE
    assert page_by_slug("sync-branch").state is PageState.PROTOTYPE
    assert page_by_slug("fusion").state is PageState.LOCKED
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_navigation.py -q`

Expected: collection fails because `dashboard.navigation` does not exist.

- [ ] **Step 3: Implement the immutable navigation manifest**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PageState(StrEnum):
    READY = "ready"
    PROTOTYPE = "prototype"
    LOCKED = "locked"


@dataclass(frozen=True, slots=True)
class PageSpec:
    slug: str
    module: str
    title: str
    state: PageState

    @property
    def navigation_label(self) -> str:
        if self.state is PageState.READY:
            return self.title
        return f"{self.title} ({self.state.value})"


PAGES = (
    PageSpec("overview", "pages/overview.py", "Overview", PageState.READY),
    PageSpec("video-input", "pages/video_input.py", "1. Video input", PageState.READY),
    PageSpec("preprocessing", "pages/preprocessing.py", "2. Preprocessing", PageState.READY),
    PageSpec("visual-model", "pages/visual_model.py", "3. Visual model", PageState.READY),
    PageSpec("prediction", "pages/prediction.py", "4. Prediction", PageState.READY),
    PageSpec("experiments", "pages/experiments.py", "Experiments", PageState.READY),
    PageSpec("audio-branch", "pages/audio_branch.py", "Audio branch", PageState.PROTOTYPE),
    PageSpec("sync-branch", "pages/sync_branch.py", "Sync branch", PageState.PROTOTYPE),
    PageSpec("fusion", "pages/fusion.py", "Fusion", PageState.LOCKED),
    PageSpec("documentation", "pages/documentation.py", "Documentation", PageState.READY),
)


def page_by_slug(slug: str) -> PageSpec:
    for page in PAGES:
        if page.slug == slug:
            return page
    raise KeyError(slug)
```

- [ ] **Step 4: Run the navigation tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_navigation.py -q`

Expected: both tests pass.

- [ ] **Step 5: Commit the navigation contract**

```powershell
git add src/deepfake_detection/dashboard/navigation.py tests/test_dashboard_navigation.py
git commit -m "Add dashboard navigation contract"
```

### Task 2: Uploaded clip and session-state contract

**Files:**
- Create: `src/deepfake_detection/dashboard/state.py`
- Create: `tests/test_dashboard_state.py`

**Interfaces:**
- Produces: `UploadedClip(name: str, suffix: str, content: bytes, sha256: str)`
- Produces: `uploaded_clip(values: MutableMapping[str, object]) -> UploadedClip | None`
- Produces: `store_upload(values: MutableMapping[str, object], *, name: str, content: bytes) -> UploadedClip`
- Produces: `temporary_video(clip: UploadedClip) -> Iterator[Path]`
- Produces: `store_prepared(values: MutableMapping[str, object], clip_sha256: str, prepared: PreparedClip) -> None`
- Produces: `prepared_for_upload(values: Mapping[str, object], clip_sha256: str) -> PreparedClip | None`
- Produces: `store_prediction(values: MutableMapping[str, object], clip_sha256: str, result: PredictionResult) -> None`
- Produces: `prediction_for_upload(values: Mapping[str, object], clip_sha256: str) -> PredictionResult | None`

- [ ] **Step 1: Write failing state and cleanup tests**

```python
from pathlib import Path

from deepfake_detection.dashboard.state import store_upload, temporary_video


def test_store_upload_hashes_bytes_and_normalizes_the_suffix() -> None:
    values: dict[str, object] = {}
    clip = store_upload(values, name="sample.MP4", content=b"video")
    assert clip.name == "sample.MP4"
    assert clip.suffix == ".mp4"
    assert len(clip.sha256) == 64


def test_temporary_video_removes_the_file_after_use() -> None:
    clip = store_upload({}, name="sample.mp4", content=b"video")
    with temporary_video(clip) as path:
        assert path.read_bytes() == b"video"
        retained = Path(path)
    assert not retained.exists()
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_state.py -q`

Expected: collection fails because `dashboard.state` does not exist.

- [ ] **Step 3: Implement the upload and temporary-file contract**

```python
from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip


@dataclass(frozen=True, slots=True)
class UploadedClip:
    name: str
    suffix: str
    content: bytes
    sha256: str


def store_upload(
    values: MutableMapping[str, object], *, name: str, content: bytes
) -> UploadedClip:
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError("Unsupported video format")
    clip = UploadedClip(
        name=name,
        suffix=suffix,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    values["dashboard.upload"] = clip
    return clip


@contextmanager
def temporary_video(clip: UploadedClip) -> Iterator[Path]:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=clip.suffix) as handle:
            path = Path(handle.name)
            handle.write(clip.content)
        yield path
    finally:
        if path is not None and path.exists():
            path.unlink()
```

Implement the prepared and prediction accessors with keys that include the
upload hash. Replacing an upload must make old derived state unreachable.

```python
def uploaded_clip(values: Mapping[str, object]) -> UploadedClip | None:
    value = values.get("dashboard.upload")
    return value if isinstance(value, UploadedClip) else None


def store_prepared(
    values: MutableMapping[str, object],
    clip_sha256: str,
    prepared: PreparedClip,
) -> None:
    values["dashboard.prepared"] = (clip_sha256, prepared)


def prepared_for_upload(
    values: Mapping[str, object], clip_sha256: str
) -> PreparedClip | None:
    value = values.get("dashboard.prepared")
    if not isinstance(value, tuple) or len(value) != 2 or value[0] != clip_sha256:
        return None
    return value[1] if isinstance(value[1], PreparedClip) else None


def store_prediction(
    values: MutableMapping[str, object],
    clip_sha256: str,
    result: PredictionResult,
) -> None:
    values["dashboard.prediction"] = (clip_sha256, result)


def prediction_for_upload(
    values: Mapping[str, object], clip_sha256: str
) -> PredictionResult | None:
    value = values.get("dashboard.prediction")
    if not isinstance(value, tuple) or len(value) != 2 or value[0] != clip_sha256:
        return None
    return value[1] if isinstance(value[1], PredictionResult) else None
```

- [ ] **Step 4: Add and pass derived-state invalidation tests**

```python
def test_a_new_upload_does_not_reuse_the_previous_prediction() -> None:
    values: dict[str, object] = {}
    first = store_upload(values, name="one.mp4", content=b"one")
    result = PredictionResult(
        clip_id="one",
        verdict="real",
        probability=0.1,
        branch_logits={"visual": -2.2},
        blockers=(),
        preprocessing_fingerprint="fixture",
    )
    store_prediction(values, first.sha256, result)
    second = store_upload(values, name="two.mp4", content=b"two")
    assert prediction_for_upload(values, second.sha256) is None
```

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_state.py -q`

Expected: all state tests pass.

- [ ] **Step 5: Commit the state contract**

```powershell
git add src/deepfake_detection/dashboard/state.py tests/test_dashboard_state.py
git commit -m "Add dashboard clip state"
```

### Task 3: Multipage shell and shared components

**Files:**
- Modify: `src/deepfake_detection/dashboard/app.py`
- Create: `src/deepfake_detection/dashboard/components.py`
- Create: `src/deepfake_detection/dashboard/pages/__init__.py`
- Create: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: `PAGES`, `PageState`, and page files from later tasks
- Produces: `render_page_header(step: str, title: str, summary: str) -> None`
- Produces: `require_upload() -> UploadedClip | None`
- Produces: `render_status(state: PageState) -> None`

- [ ] **Step 1: Create minimal honest page bodies and write the failing shell test**

Create the ten page files named in Task 1. Each page must render its approved
title, one-sentence purpose, and current state. Audio and sync say `prototype`.
Fusion says `locked`. Tasks 4 through 7 add the full teaching content and
runtime behavior.

```python
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.navigation import PAGES


def test_sidebar_lists_every_page_in_pipeline_order() -> None:
    app = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    assert not app.exception
    links = app.get("page_link")
    assert [link.label for link in links] == [
        page.navigation_label for page in PAGES
    ]


def test_shell_exposes_no_artifact_or_threshold_controls() -> None:
    app = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    assert not app.exception
    assert not app.text_input
    assert not app.slider
```

- [ ] **Step 2: Run the shell tests and verify they fail against the single page**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_app.py -q`

Expected: navigation assertions fail because the current app has no page links.

- [ ] **Step 3: Build the hidden Streamlit navigation and visible pipeline index**

Use `st.Page` and `st.navigation(..., position="hidden")`. Render one
`st.page_link` per `PageSpec`. Append `prototype` and `locked` to the visible
labels. Locked pages remain routable so their body can explain the gate.
Pass the selected page slug to the sidebar renderer. Mark the selected link as
`current` in text and with `aria-current="page"`. Mark stages backed by the
current session state as `complete`. Do not imply that prototype pages are
complete.

Keep the existing laboratory palette and font roles. Replace the oversized
hero with a compact project masthead so each teaching page begins above the
fold. Keep visible keyboard focus and the narrow-screen media query.

- [ ] **Step 4: Implement shared prerequisite and stage components**

```python
def require_upload() -> UploadedClip | None:
    clip = uploaded_clip(st.session_state)
    if clip is None:
        st.info("Start with 1. Video input, then return to this page.")
        st.page_link("pages/video_input.py", label="Go to Video input")
    return clip


def render_page_header(step: str, title: str, summary: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.write(summary)
```

- [ ] **Step 5: Run the shell tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_app.py -q`

Expected: shell tests pass with no model loads.

- [ ] **Step 6: Commit the shell**

```powershell
git add src/deepfake_detection/dashboard/app.py src/deepfake_detection/dashboard/components.py src/deepfake_detection/dashboard/pages tests/test_dashboard_app.py
git commit -m "Build multipage dashboard shell"
```

### Task 4: Overview and video input pages

**Files:**
- Modify: `src/deepfake_detection/dashboard/pages/overview.py`
- Modify: `src/deepfake_detection/dashboard/pages/video_input.py`
- Create: `tests/test_dashboard_overview.py`
- Create: `tests/test_dashboard_video_input.py`

**Interfaces:**
- Consumes: `store_upload()` and shared page components
- Produces: session key `dashboard.upload`

- [ ] **Step 1: Write failing overview content tests**

```python
def test_overview_names_the_current_system_and_limits() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/overview.py"
    ).run()
    assert not page.exception
    body = " ".join(item.value for item in page.markdown)
    assert "FakeAVCeleb" in body
    assert "FaceForensics++" in body
    assert "MNW" in body
    assert "visual development baseline" in body.lower()
```

- [ ] **Step 2: Implement the overview as a bounded reading page**

Show the question, the visual/audio/sync/fusion pipeline, four dataset states,
and the current research boundary. Use `st.container(border=True)` for pipeline
stages. Do not copy obsolete metrics or architecture claims from `origin/old`.

- [ ] **Step 3: Write failing upload conversion tests at the page boundary**

Test `store_upload()` with a Streamlit-compatible byte fixture. Confirm the
page has one file uploader, explains local handling, and contains no dataset
write action.

- [ ] **Step 4: Implement the video input page**

Use one `st.file_uploader`. When a file is present, call `store_upload`, show
`st.video(content)`, filename, byte size, and the first twelve hash characters.
Show a `Continue to Preprocessing` page link only after the upload is stored.

- [ ] **Step 5: Run both page test files**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_overview.py tests/test_dashboard_video_input.py -q`

Expected: all tests pass without CUDA or model loading.

- [ ] **Step 6: Commit the first two teaching pages**

```powershell
git add src/deepfake_detection/dashboard/pages/overview.py src/deepfake_detection/dashboard/pages/video_input.py tests/test_dashboard_overview.py tests/test_dashboard_video_input.py
git commit -m "Add overview and video input pages"
```

### Task 5: Real preprocessing teaching page

**Files:**
- Create: `src/deepfake_detection/dashboard/runtime.py`
- Modify: `src/deepfake_detection/dashboard/configuration.py`
- Modify: `src/deepfake_detection/dashboard/pages/preprocessing.py`
- Create: `tests/test_dashboard_runtime.py`
- Create: `tests/test_dashboard_preprocessing.py`
- Modify: `tests/test_dashboard_config.py`

**Interfaces:**
- Consumes: `UploadedClip`, `dashboard_defaults()`, and `build_preprocessor()`
- Produces: `prepare_uploaded_visual(clip: UploadedClip, *, device: str = "cuda") -> PreparedClip`
- Produces: `display_face_frames(view: np.ndarray) -> tuple[np.ndarray, ...]`

- [ ] **Step 1: Write failing frame display and runtime tests**

```python
from types import SimpleNamespace


def test_display_face_frames_reverses_imagenet_normalization() -> None:
    normalized = np.zeros((1, 3, 2, 2), dtype=np.float32)
    frames = display_face_frames(normalized)
    assert len(frames) == 1
    assert frames[0].shape == (2, 2, 3)
    assert frames[0].dtype == np.uint8


def test_prepare_uploaded_visual_uses_the_frozen_preprocessing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakePreprocessor:
        def prepare_visual(self, record, path):
            calls["prepared_path"] = path
            return SimpleNamespace(
                preprocessing_config_hash=(
                    "fd372dbe6bb64f359db4d57b05c3b5cd"
                    "27ed6660f2bb8bdc50567224e0928c96"
                )
            )

    def fake_factory(**values):
        calls.update(values)
        return FakePreprocessor()

    monkeypatch.setattr(runtime, "build_preprocessor", fake_factory)
    clip = UploadedClip("sample.mp4", ".mp4", b"video", "a" * 64)
    prepare_uploaded_visual(clip, device="cpu")
    assert calls["code_version"] == "2689577"
    assert calls["device"] == "cpu"
```

- [ ] **Step 2: Run the runtime tests and verify missing interfaces**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_runtime.py -q`

Expected: collection fails because `dashboard.runtime` does not exist.

- [ ] **Step 3: Bind and test the frozen preprocessing identity**

Add `preprocessing_hash` to `DashboardDefaults`. Set it to:

```text
fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96
```

Extend `tests/test_dashboard_config.py` to assert this value. The runtime must
read the value from `dashboard_defaults()` instead of copying it into page
code.

- [ ] **Step 4: Implement the runtime adapter**

Build the existing preprocessor with MTCNN, greedy IoU, box crops, and the
frozen code version. Materialize the upload through `temporary_video()` and
call `prepare_visual()`. Verify `prepared.preprocessing_config_hash` equals the
checkpoint metadata hash before returning it.

Reverse ImageNet normalization with the literal means and standard deviations
used by `views.preprocessor._normalize_image`. Clip to `[0, 255]`, transpose to
HWC, and return `uint8` frames.

- [ ] **Step 5: Write the failing page prerequisite and stage tests**

```python
def test_preprocessing_page_requires_video_input() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/preprocessing.py"
    ).run()
    assert not page.exception
    assert any("Video input" in item.value for item in page.info)


def test_preprocessing_stage_names_match_the_real_pipeline() -> None:
    assert PREPROCESSING_STAGES == (
        "Media probe",
        "Timestamp sampling",
        "Face detection and tracking",
        "Face crop",
        "Resize and normalization",
        "Model tensor",
    )
```

- [ ] **Step 6: Implement the numbered preprocessing page**

Render the six stages in bordered containers. The first page visit with an
upload shows a `Run preprocessing` button. After it runs, store the
`PreparedClip`, show the sixteen denormalized face crops, tensor shape, face
coverage, stable-track status, and preprocessing hash. If the visual view is
missing, show the quality blockers and no tensor preview.

Catch `OSError`, `RuntimeError`, and `ValueError` at the page action boundary.
Show direct guidance for a decode failure, missing face view, provenance
mismatch, or unavailable CUDA. Add an AppTest case that injects a runtime
failure and confirms that the page shows an error without storing output.

- [ ] **Step 7: Run preprocessing tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_config.py tests/test_dashboard_runtime.py tests/test_dashboard_preprocessing.py tests/test_preprocessor.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit preprocessing teaching flow**

```powershell
git add src/deepfake_detection/dashboard/configuration.py src/deepfake_detection/dashboard/runtime.py src/deepfake_detection/dashboard/pages/preprocessing.py tests/test_dashboard_config.py tests/test_dashboard_runtime.py tests/test_dashboard_preprocessing.py
git commit -m "Add preprocessing teaching page"
```

### Task 6: Visual model and prediction pages

**Files:**
- Modify: `src/deepfake_detection/dashboard/runtime.py`
- Modify: `src/deepfake_detection/dashboard/pages/visual_model.py`
- Modify: `src/deepfake_detection/dashboard/pages/prediction.py`
- Modify: `src/deepfake_detection/dashboard/view_model.py`
- Create: `tests/test_dashboard_visual_model.py`
- Create: `tests/test_dashboard_prediction.py`
- Modify: `tests/test_dashboard_view.py`

**Interfaces:**
- Produces: `load_frozen_visual_engine() -> VisualPredictionEngine`
- Produces: `predict_upload(clip: UploadedClip) -> PredictionResult`
- Consumes: `build_view_model(result, threshold=0.5)`

- [ ] **Step 1: Write a failing frozen-engine configuration test**

Test that `load_frozen_visual_engine()` passes the checkpoint SHA-256, run ID,
split hash, training commit, seed, code version, threshold `0.5`, and device
`cuda` from `dashboard_defaults()` into `VisualInferenceConfig`.

- [ ] **Step 2: Implement the cached frozen-engine adapter**

Keep Streamlit caching at this server-owned boundary. Do not accept function
arguments that a browser can control.

```python
@st.cache_resource
def load_frozen_visual_engine() -> VisualPredictionEngine:
    defaults = dashboard_defaults(root=Path.cwd())
    return load_visual_prediction_engine(
        VisualInferenceConfig(
            visual_checkpoint=defaults.visual_checkpoint,
            code_version=defaults.code_version,
            expected_checkpoint_sha256=defaults.checkpoint_sha256,
            expected_run_id=defaults.run_id,
            expected_split_hash=defaults.split_hash,
            expected_git_commit=defaults.git_commit,
            expected_seed=defaults.seed,
            threshold=0.5,
            device="cuda",
        )
    )
```

- [ ] **Step 3: Write failing visual model page tests**

Assert the page shows the literal shape ladder:

```text
[B, 16, 3, 224, 224]
[B*16, 1280]
[B, 16, 1280]
[B, 256]
[B]
```

Assert it explains EfficientNet-B0, GRU, logit, and sigmoid without claiming
that intermediate values are explanations.

- [ ] **Step 4: Implement the visual model page**

Show five numbered model stages. If preprocessing output exists, show its real
input shape. If a stored prediction exists, show its real visual logit and
probability. Otherwise explain that Prediction runs the classifier.

- [ ] **Step 5: Write failing prediction page tests**

Test missing-upload guidance, the single `Analyze video` action, fixed
threshold copy, development-scope copy, and the absence of text inputs,
sliders, and select boxes.

- [ ] **Step 6: Implement prediction with persistent session results**

Run `predict_upload()` only when the button is pressed. Store the result under
the current upload hash. Render the existing result card, probability, fixed
threshold, visual coverage, blockers, and limitations. Put logit, run ID,
checkpoint hash, split hash, and preprocessing fingerprint in a `Technical
details` expander.

Catch `OSError`, `RuntimeError`, and `ValueError` at the action boundary. Keep
the last valid result only for its matching upload hash. Test that a failed run
shows an error and does not create a result.

- [ ] **Step 7: Run visual and prediction tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_runtime.py tests/test_dashboard_visual_model.py tests/test_dashboard_prediction.py tests/test_dashboard_view.py tests/test_inference.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit model and prediction pages**

```powershell
git add src/deepfake_detection/dashboard/runtime.py src/deepfake_detection/dashboard/pages/visual_model.py src/deepfake_detection/dashboard/pages/prediction.py src/deepfake_detection/dashboard/view_model.py tests/test_dashboard_runtime.py tests/test_dashboard_visual_model.py tests/test_dashboard_prediction.py tests/test_dashboard_view.py
git commit -m "Add visual model and prediction pages"
```

### Task 7: Experiments, branch status, fusion, and documentation pages

**Files:**
- Create: `src/deepfake_detection/dashboard/evidence.py`
- Modify: `src/deepfake_detection/dashboard/pages/experiments.py`
- Modify: `src/deepfake_detection/dashboard/pages/audio_branch.py`
- Modify: `src/deepfake_detection/dashboard/pages/sync_branch.py`
- Modify: `src/deepfake_detection/dashboard/pages/fusion.py`
- Modify: `src/deepfake_detection/dashboard/pages/documentation.py`
- Create: `tests/test_dashboard_evidence.py`
- Create: `tests/test_dashboard_status_pages.py`

**Interfaces:**
- Produces: `ValidationEvidence` with dataset, rows, threshold, metrics, confusion, epochs, run IDs, and hashes
- Produces: `load_validation_evidence(metrics_path: Path, history_path: Path) -> ValidationEvidence`

- [ ] **Step 1: Write failing evidence parsing tests**

```python
def test_validation_evidence_reads_the_tracked_metric_contract(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "FakeAVCeleb",
                "rows": 400,
                "fixed_threshold": 0.5,
                "evidence_scope": "development_validation",
                "checkpoint_run_id": "4243b35e64c743b89cc33000cc9d3d3e",
                "evaluation_run_id": "56182266f70a424581f763b2d3b41989",
                "checkpoint_sha256": "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0",
                "metrics": {
                    "roc_auc": 0.999175,
                    "pr_auc": 0.999292,
                    "balanced_accuracy": 0.9975,
                    "f1": 0.997494,
                },
                "confusion": {
                    "true_positive": 199,
                    "true_negative": 200,
                    "false_positive": 0,
                    "false_negative": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "best_epoch": 4,
                "epochs": [{"epoch": value} for value in range(1, 6)],
                "metadata": {
                    "run_id": "4243b35e64c743b89cc33000cc9d3d3e",
                    "preprocessing_hash": (
                        "fd372dbe6bb64f359db4d57b05c3b5cd"
                        "27ed6660f2bb8bdc50567224e0928c96"
                    ),
                    "split_hash": (
                        "3255ae334536336c73058941285925f3d"
                        "d5b094c02b1037e19f379c6f45db30c"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = load_validation_evidence(path, history_path)
    assert evidence.dataset == "FakeAVCeleb"
    assert evidence.rows == 400
    assert evidence.best_epoch == 4
    assert len(evidence.epochs) == 5
    assert evidence.metrics["roc_auc"] == 0.999175
    assert evidence.confusion["false_negative"] == 1
```

Add rejection tests for a missing file, wrong evidence scope, missing metric,
and a run ID that differs from `dashboard_defaults().run_id`.

- [ ] **Step 2: Implement strict local evidence parsing**

Read `runs/initial-20260902/visual-validation-metrics.json` and
`runs/initial-20260902/visual-initial-history.json`. Validate
`evidence_scope == "development_validation"`, dataset, checkpoint hash,
training run ID, threshold, split hash, preprocessing hash, best epoch, epoch
count, and required metric names. Cross-check shared fields between both files.
Return a frozen data class. Do not query remote services from page rendering.

- [ ] **Step 3: Implement the experiments page**

Show training run, evaluation run, five epochs, best epoch four, validation
rows, ROC AUC, PR AUC, balanced accuracy, F1, and confusion counts. Label every
table `FakeAVCeleb development validation`. Provide
`http://127.0.0.1:5000` as the local MLflow link and explain how to choose the
experiment and runs.

- [ ] **Step 4: Write failing status-page tests**

Use AppTest to require:

- Audio page contains `prototype` and `full training is incomplete`.
- Sync page contains `prototype` and `full training is incomplete`.
- Fusion page contains `locked` and `software fixture`.
- Documentation page links only to existing files.

- [ ] **Step 5: Implement audio, sync, and fusion teaching pages**

Use the architecture names and current status from the handoff and handbook.
Show input, processing stages, output, current artifact status, and unlock
condition. Do not load prototype checkpoints or calculate a probability.

- [ ] **Step 6: Implement the documentation page**

Link to these existing files:

```text
docs/handoff.md
docs/research-design.md
docs/data-card.md
docs/reproducibility.md
docs/model-selection.md
docs/reference/cli.md
```

Use repository-relative text and verify each path before rendering the link.

- [ ] **Step 7: Run evidence and status tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_dashboard_evidence.py tests/test_dashboard_status_pages.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the research evidence pages**

```powershell
git add src/deepfake_detection/dashboard/evidence.py src/deepfake_detection/dashboard/pages/experiments.py src/deepfake_detection/dashboard/pages/audio_branch.py src/deepfake_detection/dashboard/pages/sync_branch.py src/deepfake_detection/dashboard/pages/fusion.py src/deepfake_detection/dashboard/pages/documentation.py tests/test_dashboard_evidence.py tests/test_dashboard_status_pages.py
git commit -m "Add dashboard research evidence pages"
```

### Task 8: Documentation, runtime smoke, and release verification

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/handoff.md`

**Interfaces:**
- Consumes: all completed dashboard pages and the frozen local artifacts
- Produces: verified local launch instructions and final handoff state

- [ ] **Step 1: Update user documentation**

Document the ten-page flow and this launch command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

State which pages execute code and which are teaching or status pages. Keep the
research limitations beside the usage instructions.

- [ ] **Step 2: Run all automated verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
.\.venv\Scripts\ddf-docs.exe
uv lock --check
git diff --check
```

Expected: every command exits zero. Pytest has one expected skip when the
optional pinned YuNet file is absent.

- [ ] **Step 3: Run the Streamlit component smoke**

```powershell
.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; app=AppTest.from_file('src/deepfake_detection/dashboard/app.py', default_timeout=60).run(); assert not app.exception; print([link.label for link in app.get('page_link')])"
```

Expected: all ten page labels, including prototype and locked suffixes, print
in the approved order and no exception is reported.

- [ ] **Step 4: Run one real CUDA prediction**

Verify the FakeAVCeleb dataset root exists. Resolve the previously verified
authentic validation row through the frozen validation manifest. Do not put a
machine-specific raw-data path in tracked documentation. Use the
provenance-checked dashboard runtime and confirm CUDA is available. Confirm
the checkpoint hash equals
`ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0`,
and the known authentic clip probability remains within `0.001` of
`0.006941306870430708`.

If the ignored raw dataset is absent, report this verification step as blocked.
Do not replace it with a fixture or claim that the real prediction passed.

- [ ] **Step 5: Review the final diff for safety**

Confirm the dashboard contains no `joblib` import, artifact path control,
threshold control, device control, W&B copy, dataset write, or public bind.
Confirm no file under `data`, `runs`, `mlartifacts`, or `models` is tracked.

- [ ] **Step 6: Commit the release documentation**

```powershell
git add README.md CHANGELOG.md docs/handoff.md
git commit -m "Document multipage teaching dashboard"
```

- [ ] **Step 7: Push verified main**

```powershell
git push origin main
```

Expected: local `main` and `origin/main` resolve to the same commit.
