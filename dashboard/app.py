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

st.set_page_config(page_title="Preprocessing dashboard", layout="wide")

# Flat page list in pipeline order — no sidebar sections. The three streams live
# as tabs inside the single Streams page (pages/streams.py).
# Fusion and Explainability are disabled for now — re-add their st.Page lines to
# restore them (the page files are still in pages/).
nav = st.navigation([
    st.Page("pages/preprocess.py", title="Preprocessing"),
    st.Page("pages/streams.py", title="Streams"),
    # st.Page("pages/fusion.py", title="Fusion"),
    # st.Page("pages/explainability.py", title="Explainability"),
])
nav.run()
