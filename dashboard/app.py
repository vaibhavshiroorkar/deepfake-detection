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

nav = st.navigation({
    "Data Pre-processing": [
        st.Page("pages/preprocess.py", title="Preprocessing"),
    ],
    "Streams": [
        st.Page("pages/stream_visual.py", title="Visual"),
        st.Page("pages/stream_lipsync.py", title="LipSync"),
        st.Page("pages/stream_emotions.py", title="Emotions"),
    ],
    "Fusion": [
        st.Page("pages/fusion.py", title="Fusion"),
    ],
    "Explainability": [
        st.Page("pages/explainability.py", title="Explainability"),
    ],
})
nav.run()
