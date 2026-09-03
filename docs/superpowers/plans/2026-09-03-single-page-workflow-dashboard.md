# Single-page Workflow Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the original Evidence Gate interface and add the completed teaching features as a four-step, single-page workflow.

**Architecture:** Merge the tested dashboard foundation from `feat/multipage-teaching-dashboard` into `main`, then replace its navigation shell. Page modules will expose reusable render functions. The restored original shell will call those functions inside four workflow expanders and five research tabs, while shared state and runtime modules keep upload-derived values safe.

**Tech Stack:** Python 3.13, Streamlit 1.60, PyTorch 2.12 with CUDA 13.0, NumPy, pytest, Streamlit AppTest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-03-single-page-workflow-dashboard-design.md`

## Global Constraints

- Implement directly on `main`.
- Preserve the original `Verdict follows coverage.` hero, palette, typography, result card, evidence gate, and frozen-baseline sidebar.
- Bind Streamlit to `127.0.0.1`.
- Accept no client-controlled checkpoint, model, fusion, threshold, or device values.
- Keep the threshold fixed at `0.5` and the device fixed at `cuda`.
- Store uploaded bytes only in Streamlit session state and temporary files.
- Delete every temporary file in a `finally` block.
- Keep audio and sync marked `prototype` and fusion marked `locked`.
- Use ASCII punctuation in code, comments, documentation, and interface copy.
- Do not change model artifacts, preprocessing behavior, or research evidence.

---

### Task 1: Integrate the tested dashboard foundation

**Files:**
- Merge: `feat/multipage-teaching-dashboard` into `main`
- Verify: `src/deepfake_detection/dashboard/state.py`
- Verify: `src/deepfake_detection/dashboard/runtime.py`
- Verify: `src/deepfake_detection/dashboard/evidence.py`
- Verify: `src/deepfake_detection/dashboard/pages/*.py`
- Verify: `tests/test_dashboard_*.py`

**Interfaces:**
- Produces: `UploadedClip`, `store_upload()`, `clear_upload()`, and hash-bound prepared and prediction state
- Produces: `prepare_uploaded_visual()`, `load_frozen_visual_engine()`, and `predict_upload()`
- Produces: strict local validation-evidence parsing
- Produces: the ten tested teaching page bodies that later tasks will embed

- [ ] **Step 1: Confirm both branches are clean and have the expected tips**

```powershell
git status --short --branch
git log -1 --oneline main
git log -1 --oneline feat/multipage-teaching-dashboard
git -C .worktrees/multipage-teaching-dashboard status --short --branch
```

Expected: `main` has no uncommitted files. The feature worktree is clean and its tip is `4b08d24` or a direct verified successor.

- [ ] **Step 2: Merge the existing feature work into main**

```powershell
git merge --no-ff feat/multipage-teaching-dashboard -m "Merge dashboard teaching features"
```

Expected: the merge completes without conflicts. Do not resolve unexpected conflicts by choosing one whole side. Stop and inspect each conflict.

- [ ] **Step 3: Run the foundation tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dashboard_state.py `
  tests/test_dashboard_runtime.py `
  tests/test_dashboard_evidence.py `
  tests/test_dashboard_preprocessing.py `
  tests/test_dashboard_prediction.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Confirm the merge contains no data or model artifacts**

```powershell
git diff --name-only HEAD^1..HEAD | rg '^(data|runs|mlartifacts|models)/'
```

Expected: `rg` prints nothing and exits `1` because no forbidden path is present.

### Task 2: Make teaching pages reusable

**Files:**
- Modify: `src/deepfake_detection/dashboard/pages/video_input.py`
- Modify: `src/deepfake_detection/dashboard/pages/preprocessing.py`
- Modify: `src/deepfake_detection/dashboard/pages/visual_model.py`
- Modify: `src/deepfake_detection/dashboard/pages/prediction.py`
- Modify: `src/deepfake_detection/dashboard/pages/experiments.py`
- Modify: `src/deepfake_detection/dashboard/pages/audio_branch.py`
- Modify: `src/deepfake_detection/dashboard/pages/sync_branch.py`
- Modify: `src/deepfake_detection/dashboard/pages/fusion.py`
- Modify: `src/deepfake_detection/dashboard/pages/documentation.py`
- Modify: `src/deepfake_detection/dashboard/components.py`
- Test: `tests/test_dashboard_sections.py`

**Interfaces:**
- Produces: `render_video_input(*, embedded: bool = False) -> None`
- Produces: `render_preprocessing(*, embedded: bool = False) -> None`
- Produces: `render_visual_model(*, embedded: bool = False) -> None`
- Produces: `render_prediction(*, embedded: bool = False) -> None`
- Produces: `render_experiments(*, embedded: bool = False) -> None`
- Produces: `render_audio_branch(*, embedded: bool = False) -> None`
- Produces: `render_sync_branch(*, embedded: bool = False) -> None`
- Produces: `render_fusion(*, embedded: bool = False) -> None`
- Produces: `render_documentation(*, embedded: bool = False) -> None`
- Produces: `require_upload(*, show_page_link: bool = True) -> UploadedClip | None`
- Preserves: direct execution of each page module through an `if __name__ == "__main__"` guard

- [ ] **Step 1: Write failing import-safety tests**

Create `tests/test_dashboard_sections.py`:

```python
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("streamlit")


SECTION_FUNCTIONS = (
    ("video_input", "render_video_input"),
    ("preprocessing", "render_preprocessing"),
    ("visual_model", "render_visual_model"),
    ("prediction", "render_prediction"),
    ("experiments", "render_experiments"),
    ("audio_branch", "render_audio_branch"),
    ("sync_branch", "render_sync_branch"),
    ("fusion", "render_fusion"),
    ("documentation", "render_documentation"),
)


@pytest.mark.parametrize(("module_name", "function_name"), SECTION_FUNCTIONS)
def test_page_module_exposes_one_render_function(
    module_name: str, function_name: str
) -> None:
    module = importlib.import_module(
        f"deepfake_detection.dashboard.pages.{module_name}"
    )
    assert callable(getattr(module, function_name))
```

- [ ] **Step 2: Run the test and verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_sections.py -q
```

Expected: all parameter cases fail because the render functions do not exist.

- [ ] **Step 3: Wrap each workflow page body in its named function**

Use this exact module shape. Keep existing helper functions at module scope.

```python
def render_video_input(*, embedded: bool = False) -> None:
    if not embedded:
        render_page_header(
            "Step 1 of 4",
            "Video input",
            "Choose one talking-head video for local analysis.",
        )


if __name__ == "__main__":
    render_video_input()
```

Place the current uploader, preview, metadata, and remove action after the
conditional header. Apply the same shape with the interfaces listed above. In
the four workflow sections, skip `render_page_header()` and `render_status()`
when `embedded` is true. In the five research sections, skip only the page
header so prototype and locked status text remains visible. Pass
`show_page_link=not embedded` to `require_upload()`. Hide `Continue` page links
in embedded mode. Move each existing page body without changing its runtime
calls, error handling, or state behavior. Do not nest existing module helpers
inside the render function.

Change the prerequisite helper to:

```python
def require_upload(*, show_page_link: bool = True) -> UploadedClip | None:
    clip = uploaded_clip(st.session_state)
    if clip is None:
        st.info("Start with 1. Video input before using this section.")
        if show_page_link:
            st.page_link("pages/video_input.py", label="Go to Video input")
    return clip
```

- [ ] **Step 4: Keep standalone page tests passing**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dashboard_sections.py `
  tests/test_dashboard_video_input.py `
  tests/test_dashboard_preprocessing.py `
  tests/test_dashboard_visual_model.py `
  tests/test_dashboard_prediction.py `
  tests/test_dashboard_status_pages.py
```

Expected: all tests pass. The `__main__` guards preserve existing AppTest behavior.

- [ ] **Step 5: Commit the reusable sections**

```powershell
git add src/deepfake_detection/dashboard/pages tests/test_dashboard_sections.py
git commit -m "Refactor dashboard pages into reusable sections"
```

### Task 3: Replace page navigation with a workflow contract

**Files:**
- Delete: `src/deepfake_detection/dashboard/navigation.py`
- Create: `src/deepfake_detection/dashboard/status.py`
- Create: `src/deepfake_detection/dashboard/workflow.py`
- Delete: `tests/test_dashboard_navigation.py`
- Create: `tests/test_dashboard_workflow.py`
- Modify: `src/deepfake_detection/dashboard/components.py`

**Interfaces:**
- Produces: `StepState(StrEnum)` with `WAITING`, `READY`, and `COMPLETE`
- Produces: `PageState(StrEnum)` with `READY`, `PROTOTYPE`, and `LOCKED` for standalone section labels
- Produces: `WorkflowState(video: StepState, preprocessing: StepState, visual_model: StepState, prediction: StepState)`
- Produces: `workflow_state(*, has_upload: bool, has_prepared: bool, has_prediction: bool) -> WorkflowState`
- Produces: `render_step_status(state: StepState) -> None`
- Removes: `PAGES`, `PageSpec`, `page_by_slug()`, and `pipeline_stage_status()`

- [ ] **Step 1: Write failing workflow-state tests**

Create `tests/test_dashboard_workflow.py`:

```python
from deepfake_detection.dashboard.workflow import StepState, workflow_state


def test_empty_workflow_waits_for_video() -> None:
    state = workflow_state(
        has_upload=False, has_prepared=False, has_prediction=False
    )
    assert state.video is StepState.READY
    assert state.preprocessing is StepState.WAITING
    assert state.visual_model is StepState.READY
    assert state.prediction is StepState.WAITING


def test_prepared_video_unlocks_prediction() -> None:
    state = workflow_state(
        has_upload=True, has_prepared=True, has_prediction=False
    )
    assert state.video is StepState.COMPLETE
    assert state.preprocessing is StepState.COMPLETE
    assert state.visual_model is StepState.READY
    assert state.prediction is StepState.READY


def test_prediction_completes_the_workflow() -> None:
    state = workflow_state(
        has_upload=True, has_prepared=True, has_prediction=True
    )
    assert state.prediction is StepState.COMPLETE
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_workflow.py -q
```

Expected: collection fails because `dashboard.workflow` does not exist.

- [ ] **Step 3: Implement the pure workflow-state contract**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StepState(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class WorkflowState:
    video: StepState
    preprocessing: StepState
    visual_model: StepState
    prediction: StepState


def workflow_state(
    *, has_upload: bool, has_prepared: bool, has_prediction: bool
) -> WorkflowState:
    return WorkflowState(
        video=StepState.COMPLETE if has_upload else StepState.READY,
        preprocessing=(
            StepState.COMPLETE
            if has_prepared
            else StepState.READY
            if has_upload
            else StepState.WAITING
        ),
        visual_model=StepState.READY,
        prediction=(
            StepState.COMPLETE
            if has_prediction
            else StepState.READY
            if has_prepared
            else StepState.WAITING
        ),
    )
```

- [ ] **Step 4: Replace navigation components with status text**

Move `PageState` into `dashboard/status.py` and update all section imports.
Keep `render_status()`, `render_page_header()`, and the revised
`require_upload()` for standalone section tests. Remove only navigation-specific
component code.

```python
def render_step_status(state: StepState) -> None:
    st.markdown(
        f'<div class="step-state {state.value}">{state.value}</div>',
        unsafe_allow_html=True,
    )
```

Delete the navigation manifest and page-index status logic. Page links remain
available only when a section runs directly with `embedded=False`.

- [ ] **Step 5: Run the workflow and component tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dashboard_workflow.py `
  tests/test_dashboard_sections.py `
  tests/test_dashboard_status_pages.py
```

Expected: all tests pass and no test imports `dashboard.navigation`.

- [ ] **Step 6: Commit the workflow contract**

```powershell
git add src/deepfake_detection/dashboard tests/test_dashboard_navigation.py tests/test_dashboard_workflow.py
git commit -m "Replace dashboard navigation with workflow state"
```

### Task 4: Restore the original shell and embed the workflow

**Files:**
- Modify: `src/deepfake_detection/dashboard/app.py`
- Modify: `tests/test_dashboard_app.py`
- Modify: `tests/test_dashboard_view.py`

**Interfaces:**
- Consumes: all nine `render_*()` functions from Task 2
- Consumes: `uploaded_clip()`, `prepared_for_upload()`, and `prediction_for_upload()`
- Consumes: `workflow_state()` and `render_step_status()`
- Produces: one Streamlit page with four workflow expanders and five research tabs

- [ ] **Step 1: Replace multipage assertions with failing single-page tests**

Update `tests/test_dashboard_app.py`:

```python
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = Path("src/deepfake_detection/dashboard/app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_original_identity_wraps_the_four_step_workflow() -> None:
    app = run_app()
    assert not app.exception
    body = " ".join(item.value for item in app.markdown)
    assert "Verdict follows" in body
    assert "coverage." in body
    assert any(item.value == "Frozen baseline" for item in app.header)
    labels = [item.label for item in app.expander]
    assert labels[:4] == [
        "1. Video input",
        "2. Preprocessing",
        "3. Visual model",
        "4. Prediction",
    ]
    assert labels[-1] == "Research and methodology"


def test_research_content_uses_five_tabs() -> None:
    app = run_app()
    labels = [item.label for item in app.get("tab")]
    assert labels == [
        "Experiments",
        "Audio branch",
        "Sync branch",
        "Fusion",
        "Documentation",
    ]
    body = " ".join(item.value for item in app.markdown).lower()
    assert "prototype" in body
    assert "locked" in body


def test_single_page_has_no_navigation_links() -> None:
    app = run_app()
    assert not app.get("page_link")
```

- [ ] **Step 2: Run the app tests and verify they fail against the multipage shell**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_app.py -q
```

Expected: tests fail because the current app has page links and no workflow expanders.

- [ ] **Step 3: Restore the original shell CSS and sidebar**

Use the pre-merge `main` version as the source of truth:

```powershell
git show HEAD^1:src/deepfake_detection/dashboard/app.py
```

Restore these selectors and tokens in `app.py`: `--paper`, `--ink`, `--cobalt`, `--amber`, `--evidence`, `--teal`, `--line`, `.thesis`, `.gate`, `.channel`, `.scope`, `.limits`, and `.result`. Keep the narrow-screen `.gate` rule and visible focus outline. Add the reduced-motion rule from the feature shell.

Add text-backed workflow status styling:

```css
.step-state {
    font: 700 .76rem/1 "Cascadia Mono", Consolas, monospace;
    letter-spacing: .06em;
    margin-bottom: .75rem;
    text-transform: uppercase;
}
.step-state.waiting { color: var(--amber); }
.step-state.ready { color: var(--cobalt); }
.step-state.complete { color: var(--teal); }
```

Render the original sidebar with checkpoint name, run ID, threshold `0.50`, and device `cuda`. Do not put workflow navigation in the sidebar.

- [ ] **Step 4: Implement one-pass workflow state and expanders**

```python
def _current_workflow() -> tuple[UploadedClip | None, WorkflowState]:
    clip = uploaded_clip(st.session_state)
    prepared = prepared_for_upload(st.session_state, clip.sha256) if clip else None
    prediction = prediction_for_upload(st.session_state, clip.sha256) if clip else None
    return clip, workflow_state(
        has_upload=clip is not None,
        has_prepared=prepared is not None,
        has_prediction=prediction is not None,
    )


clip, flow = _current_workflow()

with st.expander("1. Video input", expanded=clip is None):
    render_step_status(flow.video)
    render_video_input(embedded=True)

clip, flow = _current_workflow()
with st.expander(
    "2. Preprocessing",
    expanded=clip is not None and flow.preprocessing is not StepState.COMPLETE,
):
    render_step_status(flow.preprocessing)
    render_preprocessing(embedded=True)

clip, flow = _current_workflow()
with st.expander("3. Visual model"):
    render_step_status(flow.visual_model)
    render_visual_model(embedded=True)

clip, flow = _current_workflow()
with st.expander(
    "4. Prediction",
    expanded=flow.prediction is StepState.READY,
):
    render_step_status(flow.prediction)
    render_prediction(embedded=True)
```

The page functions already hide actions when prerequisites are missing. Keep their direct guidance inside waiting sections. Re-read state after each section so a stored upload or prepared clip unlocks later sections in the same rerun.

- [ ] **Step 5: Embed research content in one expander and five tabs**

```python
with st.expander("Research and methodology"):
    experiments, audio, sync, fusion, documentation = st.tabs(
        (
            "Experiments",
            "Audio branch",
            "Sync branch",
            "Fusion",
            "Documentation",
        )
    )
    with experiments:
        render_experiments(embedded=True)
    with audio:
        render_audio_branch(embedded=True)
    with sync:
        render_sync_branch(embedded=True)
    with fusion:
        render_fusion(embedded=True)
    with documentation:
        render_documentation(embedded=True)
```

- [ ] **Step 6: Keep the original result card in prediction rendering**

Move the original `.scope`, `.result`, `.gate`, and `.limits` HTML contract into `render_prediction()`. Build its values from `build_view_model(stored_result, threshold=0.5)`. Escape all upload-derived text before inserting it into HTML. Keep blockers in normal Streamlit text and technical values in the existing details expander.

- [ ] **Step 7: Run app, view-model, and prediction tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dashboard_app.py `
  tests/test_dashboard_view.py `
  tests/test_dashboard_prediction.py `
  tests/test_dashboard_workflow.py
```

Expected: all tests pass. AppTest finds five expanders, five tabs, and no page links.

- [ ] **Step 8: Commit the single-page interface**

```powershell
git add src/deepfake_detection/dashboard/app.py src/deepfake_detection/dashboard/pages/prediction.py tests/test_dashboard_app.py tests/test_dashboard_view.py tests/test_dashboard_prediction.py
git commit -m "Build single-page dashboard workflow"
```

### Task 5: Verify state clearing and failure boundaries

**Files:**
- Modify: `src/deepfake_detection/dashboard/pages/video_input.py`
- Modify: `src/deepfake_detection/dashboard/pages/preprocessing.py`
- Modify: `src/deepfake_detection/dashboard/pages/prediction.py`
- Modify: `tests/test_dashboard_state.py`
- Modify: `tests/test_dashboard_preprocessing.py`
- Modify: `tests/test_dashboard_prediction.py`

**Interfaces:**
- Consumes: hash-bound upload, prepared, and prediction session keys
- Consumes: `clear_prepared_for_upload(values: MutableMapping[str, object], clip_sha256: str) -> None`
- Consumes: `clear_prediction_for_upload(values: MutableMapping[str, object], clip_sha256: str) -> None`
- Preserves: upload state when preprocessing or prediction fails

- [ ] **Step 1: Verify the existing hash-bound clear helpers**

```python
def test_failed_preprocessing_clears_derived_state_but_keeps_upload() -> None:
    values = state_with_upload_prepared_and_prediction()
    clip = uploaded_clip(values)
    assert clip is not None
    clear_prepared_for_upload(values, clip.sha256)
    clear_prediction_for_upload(values, clip.sha256)
    assert uploaded_clip(values) == clip
    assert "dashboard.prepared" not in values
    assert "dashboard.prediction" not in values


def test_failed_prediction_clears_only_prediction() -> None:
    values = state_with_upload_prepared_and_prediction()
    clip = uploaded_clip(values)
    assert clip is not None
    clear_prediction_for_upload(values, clip.sha256)
    assert uploaded_clip(values) == clip
    assert "dashboard.prepared" in values
    assert "dashboard.prediction" not in values
```

Build `state_with_upload_prepared_and_prediction()` from the existing test fixtures in `tests/test_dashboard_state.py`. Do not use untyped sentinel objects where accessors require `PreparedClip` or `PredictionResult`.

- [ ] **Step 2: Run the state tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_state.py -q
```

Expected: tests pass for the existing hash-bound clear helpers.

- [ ] **Step 3: Add a failing preprocessing action-boundary test**

Extend the existing cached-output failure test in
`tests/test_dashboard_preprocessing.py`. Store a matching
`PredictionResult` under `dashboard.prediction` before clicking the button,
then add these assertions:

```python
assert page.session_state["dashboard.upload"] == clip
assert "dashboard.prepared" not in page.session_state.filtered_state
assert "dashboard.prediction" not in page.session_state.filtered_state
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_preprocessing.py -q
```

Expected: the new assertion fails because preprocessing clears its cached
prepared value but leaves the matching prediction.

- [ ] **Step 4: Clear both derived values before preprocessing**

```python
if st.button("Run preprocessing", key="run_preprocessing", type="primary"):
    clear_prepared_for_upload(st.session_state, clip.sha256)
    clear_prediction_for_upload(st.session_state, clip.sha256)
```

Import `clear_prediction_for_upload` in `pages/preprocessing.py`. Keep the
existing `clear_prediction_for_upload()` call before prediction starts.
Existing `store_*()` calls write fresh values only after successful completion.
Do not replace the hash-bound helpers with broad session-key deletion.

- [ ] **Step 5: Verify the prediction action boundary remains correct**

In preprocessing tests, inject `RuntimeError("CUDA unavailable")`. Assert the upload remains and prepared and prediction values are absent. In prediction tests, inject `ValueError("provenance mismatch")`. Assert the upload and prepared values remain and the prediction value is absent.

- [ ] **Step 6: Run state and action tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dashboard_state.py `
  tests/test_dashboard_video_input.py `
  tests/test_dashboard_preprocessing.py `
  tests/test_dashboard_prediction.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit failure-safe state handling**

```powershell
git add src/deepfake_detection/dashboard/pages tests/test_dashboard_state.py tests/test_dashboard_preprocessing.py tests/test_dashboard_prediction.py
git commit -m "Clear stale dashboard results on failed actions"
```

### Task 6: Update documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/handoff.md`

**Interfaces:**
- Consumes: the finished single-page workflow
- Produces: accurate launch instructions and current dashboard description

- [ ] **Step 1: Replace multipage documentation with the single-page flow**

Document this launch command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run `
  src\deepfake_detection\dashboard\app.py `
  --server.address 127.0.0.1
```

Name the four expanders and the Research and methodology section. State that preprocessing and prediction require CUDA. Keep the current baseline limitations, prototype labels, and locked fusion status.

- [ ] **Step 2: Run all dashboard tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests -k dashboard
```

Expected: all dashboard tests pass.

- [ ] **Step 3: Run the complete project checks**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
.\.venv\Scripts\ddf-docs.exe
uv lock --check
git diff --check
```

Expected: every command exits zero. Pytest may report the existing optional YuNet skip.

- [ ] **Step 4: Run the Streamlit component smoke**

```powershell
.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; app=AppTest.from_file('src/deepfake_detection/dashboard/app.py', default_timeout=60).run(); assert not app.exception; assert [item.label for item in app.expander][:4] == ['1. Video input', '2. Preprocessing', '3. Visual model', '4. Prediction']; assert not app.get('page_link'); print('single-page workflow smoke passed')"
```

Expected: `single-page workflow smoke passed`.

- [ ] **Step 5: Restart the local frontend from main and verify health**

Stop only the process whose command line serves the feature worktree dashboard. Start the command documented in Step 1 from the repository root with a hidden window. Verify `http://127.0.0.1:8501/_stcore/health` returns HTTP `200`. Confirm the listener command points to `src\deepfake_detection\dashboard\app.py` in the main checkout.

- [ ] **Step 6: Review the final diff for safety**

```powershell
rg -n "joblib|0\.0\.0\.0|wandb|W&B" src/deepfake_detection/dashboard
git status --short
git log --oneline --decorate -8
```

Expected: no unsafe loader, public bind, or W&B copy appears. Only planned documentation changes remain before the final commit.

- [ ] **Step 7: Commit the documentation**

```powershell
git add README.md CHANGELOG.md docs/handoff.md
git commit -m "Document single-page dashboard workflow"
```

Do not push unless the user asks.
