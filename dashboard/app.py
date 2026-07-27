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

# Flat page list in pipeline order — no sidebar sections. Overview is the landing
# page (short, static, no compute). The three streams live as tabs inside the
# single Streams page (pages/streams.py). Fusion and Explainability stay visible
# but locked (lock icon + a locked notice in the page body) until Stages 6 and
# 10 build them. Documentation is last: the long-form reference.
nav = st.navigation([
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/preprocess.py", title="Preprocessing"),
    st.Page("pages/streams.py", title="Streams"),
    st.Page("pages/fusion.py", title="Fusion", icon=":material/lock:"),
    st.Page("pages/explainability.py", title="Explainability", icon=":material/lock:"),
    st.Page("pages/documentation.py", title="Documentation"),
])
nav.run()
