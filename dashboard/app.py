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

# Flat page list in pipeline order, no sidebar sections. Overview is the landing
# page (short, static, no compute) and Preprocessing is the one page that
# computes. Streams, Fusion and Explainability stay visible but locked (lock icon
# + a locked notice in the page body), so the shape of the system is legible
# without offering controls that do not work yet; each page says what unlocks it.
# Documentation is last: the long-form reference.
nav = st.navigation([
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/preprocess.py", title="Preprocessing"),
    st.Page("pages/streams.py", title="Streams", icon=":material/lock:"),
    st.Page("pages/fusion.py", title="Fusion", icon=":material/lock:"),
    st.Page("pages/explainability.py", title="Explainability", icon=":material/lock:"),
    st.Page("pages/documentation.py", title="Documentation"),
])
nav.run()
