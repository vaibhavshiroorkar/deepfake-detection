from streamlit.testing.v1 import AppTest

from dashboard.lib import locked

# Streams, Fusion and Explainability stay in their pipeline positions in the
# sidebar, dimmed with a lock icon and not clickable. app.py hides Streamlit's
# built-in nav and draws the list with st.page_link so it can pass disabled=True,
# which st.navigation has no way to express.

LOCKED_PAGES = [
    "dashboard/pages/streams.py",
    "dashboard/pages/fusion.py",
    "dashboard/pages/explainability.py",
]

EXPECTED_NAV_ORDER = ["Overview", "Preprocessing", "Streams", "Fusion",
                      "Explainability", "Documentation"]


def _nav_links():
    """The sidebar nav entries. AppTest has no .page_link accessor, hence .get()."""
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception
    return at.get("page_link")


def test_nav_keeps_every_section_in_pipeline_order():
    """The locked sections are not moved or hidden, just disabled in place."""
    labels = [link.label for link in _nav_links()]
    assert labels == EXPECTED_NAV_ORDER


def test_exactly_the_locked_sections_are_disabled():
    links = {link.label: link for link in _nav_links()}
    for title in locked.locked_titles():
        assert links[title].disabled, title
    for title in ["Overview", "Preprocessing", "Documentation"]:
        assert not links[title].disabled, title


def test_disabled_entries_explain_themselves_on_hover():
    """Dimmed with no explanation is just broken-looking, so each carries a tooltip."""
    links = {link.label: link for link in _nav_links()}
    for spec in locked.LOCKED:
        help_text = links[spec["title"]].help
        assert spec["status"] in help_text
        assert "Will contain" in help_text


def test_locked_page_bodies_still_render():
    """Their routes stay alive, so a direct visit must land on a real explanation."""
    for page in LOCKED_PAGES:
        at = AppTest.from_file(page, default_timeout=120).run()
        assert not at.exception, (page, at.exception)
        assert at.title[0].value.startswith(":material/lock:"), page
        assert at.info and at.info[0].value.strip(), page


def test_every_locked_section_states_why_and_what_lands_there():
    for spec in locked.LOCKED:
        assert spec["status"].strip(), spec["title"]
        assert spec["views"], spec["title"]
