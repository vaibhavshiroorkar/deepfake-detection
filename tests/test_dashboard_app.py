import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.components import pipeline_stage_status
from deepfake_detection.dashboard.navigation import PAGES, page_by_slug

_PAGE_BODIES = (
    (
        "pages/overview.py",
        "Overview",
        "Follow the detector from an uploaded clip to an evidence-limited prediction.",
        "Status: ready",
    ),
    (
        "pages/video_input.py",
        "1. Video input",
        "Select the local talking-head clip used by every later pipeline stage.",
        "Status: ready",
    ),
    (
        "pages/preprocessing.py",
        "2. Preprocessing",
        "See how one clip becomes a tracked and normalized face sequence.",
        "Status: ready",
    ),
    (
        "pages/visual_model.py",
        "3. Visual model",
        "Trace frame features through the frozen visual classifier.",
        "Status: ready",
    ),
    (
        "pages/prediction.py",
        "4. Prediction",
        "Run the frozen visual baseline and read its evidence limits.",
        "Status: ready",
    ),
    (
        "pages/experiments.py",
        "Experiments",
        "Review the saved development-validation record behind the visual baseline.",
        "Status: ready",
    ),
    (
        "pages/audio_branch.py",
        "Audio branch",
        "See what remains before audio spoof evidence can support a research claim.",
        "Status: prototype",
    ),
    (
        "pages/sync_branch.py",
        "Sync branch",
        "See how mouth-audio alignment will form a temporal evidence stream.",
        "Status: prototype",
    ),
    (
        "pages/fusion.py",
        "Fusion",
        "See how calibrated visual, audio, and sync evidence will enter late fusion.",
        "Status: locked",
    ),
    (
        "pages/documentation.py",
        "Documentation",
        "Open the repository records that define scope, data, runs, and reproducibility.",
        "Status: ready",
    ),
)


def _run_shell() -> AppTest:
    return AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()


def test_sidebar_lists_every_page_in_pipeline_order() -> None:
    app = _run_shell()
    assert not app.exception
    links = app.get("page_link")
    assert [link.label for link in links] == [page.navigation_label for page in PAGES]


def test_initial_shell_names_the_current_page() -> None:
    app = _run_shell()

    assert not app.exception
    assert [caption.value for caption in app.sidebar.caption] == ["Current: Overview"]


def test_shell_exposes_no_model_configuration_controls() -> None:
    app = _run_shell()

    assert not app.exception
    assert not app.text_input
    assert not app.slider
    assert not app.selectbox
    assert not app.radio
    assert not app.number_input


@pytest.mark.parametrize(
    ("page_path", "title", "summary", "status"),
    _PAGE_BODIES,
)
def test_registered_static_page_renders_its_body(
    page_path: str,
    title: str,
    summary: str,
    status: str,
) -> None:
    app = _run_shell()
    app.switch_page(page_path).run()

    assert not app.exception
    assert [heading.value for heading in app.title] == [title]
    assert summary in [item.value for item in app.markdown]
    assert status in [caption.value for caption in app.caption]


def test_locked_fusion_page_remains_routable() -> None:
    app = _run_shell()
    app.switch_page("pages/fusion.py").run()

    assert not app.exception
    assert [heading.value for heading in app.title] == ["Fusion"]
    assert "Status: locked" in [caption.value for caption in app.caption]


@pytest.mark.parametrize(
    (
        "page_slug",
        "selected_slug",
        "has_upload",
        "has_prepared",
        "has_prediction",
        "expected",
    ),
    (
        ("overview", "overview", False, False, False, "current"),
        ("video-input", "overview", True, False, False, "complete"),
        ("preprocessing", "overview", True, True, False, "complete"),
        ("visual-model", "overview", True, True, True, "complete"),
        ("prediction", "overview", True, True, True, "complete"),
        ("experiments", "overview", True, True, True, "ready"),
        ("audio-branch", "overview", True, True, True, "prototype"),
        ("sync-branch", "overview", True, True, True, "prototype"),
        ("fusion", "overview", True, True, True, "locked"),
    ),
)
def test_pipeline_stage_status_preserves_progress_and_research_state(
    page_slug: str,
    selected_slug: str,
    has_upload: bool,
    has_prepared: bool,
    has_prediction: bool,
    expected: str,
) -> None:
    assert (
        pipeline_stage_status(
            page_by_slug(page_slug),
            selected_slug=selected_slug,
            has_upload=has_upload,
            has_prepared=has_prepared,
            has_prediction=has_prediction,
        )
        == expected
    )


def test_missing_upload_shows_the_required_next_step() -> None:
    app = _run_shell()
    app.switch_page("pages/preprocessing.py").run()
    assert not app.exception
    assert [item.value for item in app.info] == [
        "Start with 1. Video input, then return to this page."
    ]
    assert [link.label for link in app.get("page_link")] == ["Go to Video input"]


def test_status_names_the_page_state() -> None:
    app = AppTest.from_string(
        """
from deepfake_detection.dashboard.components import render_status
from deepfake_detection.dashboard.navigation import PageState

render_status(PageState.PROTOTYPE)
""",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert [caption.value for caption in app.caption] == ["Status: prototype"]
