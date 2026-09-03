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
