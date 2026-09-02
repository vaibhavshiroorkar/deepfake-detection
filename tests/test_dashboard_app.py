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
    assert [link.label for link in links] == [page.navigation_label for page in PAGES]


def test_shell_exposes_no_artifact_or_threshold_controls() -> None:
    app = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    assert not app.exception
    assert not app.text_input
    assert not app.slider


def test_missing_upload_shows_the_required_next_step() -> None:
    app = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
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
