from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.runtime.pages_manager import PagesManager
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


def test_app_does_not_register_legacy_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages_directory = APP.parent / "pages"
    monkeypatch.setattr(PagesManager, "uses_pages_directory", None)

    manager = PagesManager(str(APP.resolve()))

    assert not pages_directory.exists()
    assert PagesManager.uses_pages_directory is False
    assert [page["script_path"] for page in manager.get_pages().values()] == [
        str(APP.resolve())
    ]
