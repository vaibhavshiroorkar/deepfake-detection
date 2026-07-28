from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.lib import locked

# Streams, Fusion and Explainability are locked, and locked here means genuinely
# unreachable: they are not registered as pages and no file for them exists under
# dashboard/pages/. They appear in the sidebar as inert text only.

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = REPO_ROOT / "dashboard" / "pages"

OPEN_PAGES = ["overview.py", "preprocess.py", "documentation.py"]


def test_only_the_working_pages_exist_under_pages():
    """A file in pages/ is servable by URL, so a locked section must not have one."""
    present = sorted(p.name for p in PAGES_DIR.glob("*.py") if p.name != "__init__.py")
    assert present == sorted(OPEN_PAGES)


def test_locked_sections_have_no_page_file():
    for title in locked.locked_titles():
        assert not (PAGES_DIR / f"{title.lower()}.py").exists(), title


def test_app_runs_without_exception():
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception


def test_locked_sections_are_listed_in_the_sidebar():
    """Visible, so the shape of the system is legible, but not as links."""
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    for title in locked.locked_titles():
        assert title in sidebar_text, title
    assert any("Locked" in c.value for c in at.sidebar.caption)


def test_every_locked_section_states_why_and_what_lands_there():
    """The reason becomes the tooltip, so it has to be non-empty in the spec."""
    for spec in locked.LOCKED:
        assert spec["status"].strip(), spec["title"]
        assert spec["views"], spec["title"]
