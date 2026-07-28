"""The locked sections, listed in the sidebar but not openable.

Streams, Fusion and Explainability used to be real pages carrying a lock icon and
a "Locked." notice. That is a label rather than a lock: the page opened, and the
notice was the only thing stopping you. They are now not pages at all, so there is
nothing to navigate to, and they appear here as inert sidebar text.

Kept visible rather than hidden because the shape of the system is worth seeing
from the landing page, and because a section that silently does not exist is
harder to reason about than one that says why it is closed. Hovering an entry
gives the reason.

When a stage lands, add a file under dashboard/pages/, register it in app.py's
st.navigation, and drop its spec from LOCKED below. The "what lands here" copy
already lives in stream_spec.py, so nothing has to be rewritten.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.lib.stream_spec import EXPLAINABILITY, FUSION, STREAMS

# Pipeline order, matching where they would sit in the nav once unlocked.
LOCKED = [STREAMS, FUSION, EXPLAINABILITY]


def locked_titles() -> list[str]:
    """Names of the sections that are not reachable."""
    return [spec["title"] for spec in LOCKED]


def render_sidebar(st):
    """List the locked sections under the nav, as text rather than links."""
    with st.sidebar:
        st.markdown("")
        st.caption("Locked")
        for spec in LOCKED:
            lands = ", ".join(name for name, _ in spec["views"])
            st.markdown(
                f":material/lock: &nbsp;{spec['title']}",
                help=f"{spec['status']}\n\nWill contain: {lands}.",
            )
