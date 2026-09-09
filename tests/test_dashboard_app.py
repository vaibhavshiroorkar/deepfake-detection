from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path("src/deepfake_detection/dashboard/app.py")
PAGES = APP.parent / "pages"

EXPECTED_NAVIGATION = [
    ("Overview", False),
    ("Evidence gate", False),
    ("Preprocessing", False),
    ("Streams", False),
    ("Visual", False),
    ("Lip-Sync", False),
    ("Emotion", False),
    ("Audio branch", False),
    ("Sync branch", False),
    ("Experiments", False),
    ("Fusion", True),
    ("Explainability", True),
    ("Documentation", False),
]


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def run_page(name: str) -> AppTest:
    return AppTest.from_file(PAGES / name, default_timeout=180).run()


ALL_PAGES = sorted(p.name for p in PAGES.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_page_renders_without_raising(name: str) -> None:
    """Every page has to survive a bare first render.

    The rest of this file checks a handful of pages by name, so a page nobody
    named could crash on import and the suite would still pass.
    """
    page = run_page(name)
    assert not page.exception, [e.message for e in page.exception]


def test_shell_draws_the_pipeline_ordered_sidebar() -> None:
    app = run_app()
    assert not app.exception
    links = app.get("page_link")
    assert [(link.label, link.disabled) for link in links] == EXPECTED_NAVIGATION


def test_shell_shows_the_frozen_baseline_provenance() -> None:
    app = run_app()
    assert any(item.value == "Frozen baseline" for item in app.header)
    captions = " ".join(item.value for item in app.caption)
    assert "Decision threshold: 0.50" in captions
    assert "Compute device: cuda" in captions


def test_shell_opens_on_the_overview_page() -> None:
    app = run_app()
    headers = [item.value for item in app.header]
    assert "The problem" in headers
    assert "Architecture" in headers


def test_every_registered_page_exists() -> None:
    app = run_app()
    for link in app.get("page_link"):
        assert (PAGES / f"{link.page}.py").is_file() or link.label == "Overview"


def test_gate_page_keeps_the_four_step_workflow() -> None:
    page = run_page("gate.py")
    assert not page.exception
    body = " ".join(item.value for item in page.markdown)
    assert "Verdict follows" in body
    assert "coverage." in body
    assert [item.label for item in page.expander] == [
        "1. Video input",
        "2. Preprocessing",
        "3. Visual model",
        "4. Prediction",
    ]


def test_documentation_page_links_the_repository_records() -> None:
    page = run_page("documentation.py")
    assert not page.exception
    labels = [item.label for item in page.get("tab")]
    assert labels[0] == "Problem & architecture"
    assert labels[-1] == "Repository records"


def test_locked_pages_state_what_unlocks_them() -> None:
    for name in ("fusion.py", "explainability.py"):
        page = run_page(name)
        assert not page.exception
        body = " ".join(
            item.value for item in list(page.info) + list(page.error)
        ).lower()
        assert "not built" in body or "locked" in body


def test_prototype_pages_report_prototype_status() -> None:
    for name in ("audio_branch.py", "sync_branch.py"):
        page = run_page(name)
        assert not page.exception
        body = " ".join(item.value for item in page.markdown).lower()
        assert "prototype" in body
