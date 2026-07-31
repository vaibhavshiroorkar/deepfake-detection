"""The locked sections: dimmed, unclickable, but still in the nav where they belong.

Fusion and Explainability sit in their pipeline positions in the sidebar, greyed
out with a lock icon, and clicking them does nothing. app.py hides Streamlit's
built-in nav and draws the list itself with st.page_link, passing disabled=True
for these two, because st.navigation has no notion of a disabled entry.

They keep a page body (render below) for the case where someone reaches the route
directly, and because that body becomes the real page when the stage lands. To
unlock one: drop its spec from LOCKED so app.py stops disabling it, then replace
the body with the real controls. The copy already lives in stream_spec.py.
Streams went through exactly that and is no longer here.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.lib.stream_spec import EXPLAINABILITY, FUSION

# Pipeline order, matching where they sit in the nav.
LOCKED = [FUSION, EXPLAINABILITY]


def locked_titles() -> list[str]:
    """Titles of the sections that are dimmed and unclickable."""
    return [spec["title"] for spec in LOCKED]


def tooltip(spec: dict) -> str:
    """Hover text for a dimmed nav entry: why it is locked, and what lands there."""
    lands = ", ".join(name for name, _ in spec["views"])
    return f"{spec['status']}\n\nWill contain: {lands}."


def render(st, spec: dict):
    """The page body, for a direct visit to a locked route."""
    st.title(f":material/lock: {spec['title']}")
    st.caption(spec["note"])
    st.info(spec["status"])

    st.subheader("What lands here")
    for name, description in spec["views"]:
        st.markdown(f"**{name}.** {description}")
