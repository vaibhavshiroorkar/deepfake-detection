"""Multi-page preprocessing experiment dashboard (PROJECT_OVERVIEW.md §7).

Run: uv run streamlit run dashboard/app.py
Never trains; never writes data/processed/. See
docs/superpowers/specs/2026-07-23-preprocessing-experiment-dashboard-design.md.
"""
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.lib import locked
from dashboard.lib.stream_spec import EXPLAINABILITY, FUSION

st.set_page_config(page_title="Preprocessing dashboard", layout="wide")

# One flat list in pipeline order, each entry carrying how to draw it. The spec
# marks a page locked: Fusion and Explainability show in place, dimmed with a
# lock icon, and do not respond to a click. `child` indents an entry under the
# section above it.
#
# The nav is drawn by hand because st.navigation has no disabled entry, so the
# built-in one is hidden and st.page_link renders the list instead, with
# disabled=True on the locked two. Registering them still keeps their routes
# alive, which is deliberate: a direct visit lands on a body that says what the
# section will hold and what unlocks it, rather than a dead end.
PAGES = [
    (st.Page("pages/overview.py", title="Overview", default=True), None, False),
    (st.Page("pages/preprocess.py", title="Preprocessing"), None, False),
    (st.Page("pages/streams.py", title="Streams"), None, False),
    (st.Page("pages/stream_visual.py", title="Visual"), None, True),
    (st.Page("pages/stream_lipsync.py", title="Lip-Sync"), None, True),
    (st.Page("pages/stream_emotion.py", title="Emotion"), None, True),
    (st.Page("pages/fusion.py", title="Fusion"), FUSION, False),
    (st.Page("pages/explainability.py", title="Explainability"), EXPLAINABILITY, False),
    (st.Page("pages/documentation.py", title="Documentation"), None, False),
]

nav = st.navigation([page for page, _, _ in PAGES], position="hidden")

with st.sidebar:
    for page, spec, child in PAGES:
        # A narrow spacer column is the indent. st.page_link has no notion of
        # nesting and st.navigation's section headers cannot themselves be pages,
        # which the Streams hub has to be.
        target = st.columns([1, 9])[1] if child else st
        if spec is None:
            target.page_link(page, width="stretch")
        else:
            target.page_link(page, icon=":material/lock:", disabled=True,
                             help=locked.tooltip(spec), width="stretch")

nav.run()
