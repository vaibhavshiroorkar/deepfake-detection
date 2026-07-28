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

st.set_page_config(page_title="Preprocessing dashboard", layout="wide")

# Only the three working pages are registered, in pipeline order: Overview is the
# landing page (static, no compute), Preprocessing is the page that computes, and
# Documentation is the long-form reference.
#
# Streams, Fusion and Explainability are NOT pages. They were previously
# registered with a lock icon, which meant they opened perfectly well and simply
# announced that they were locked; that is a label, not a lock. They are now
# listed in the sidebar as plain text, so the shape of the system stays legible
# while there is genuinely nothing to open. Their page bodies do not exist under
# pages/ either, because Streamlit will serve anything in that directory by URL
# whether or not st.navigation lists it.
nav = st.navigation([
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/preprocess.py", title="Preprocessing"),
    st.Page("pages/documentation.py", title="Documentation"),
])
locked.render_sidebar(st)
nav.run()
