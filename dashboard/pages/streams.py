"""Single Streams page — Visual, Lip-Sync and Emotions as in-page tabs.

Replaces the old three-page "Streams" sidebar section: one nav entry, with the
three feature streams switched via tabs. Each tab's body lives in
dashboard/lib/stream_pages.py.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import stream_pages

st.title("Streams")
st.caption("Three feature streams feeding fusion — each produces a 256-d clip embedding, never a "
           "standalone score. Switch between them below.")

visual_tab, lipsync_tab, emotion_tab = st.tabs(["Visual", "Lip-Sync", "Emotions"])
with visual_tab:
    stream_pages.render_visual()
with lipsync_tab:
    stream_pages.render_lipsync()
with emotion_tab:
    stream_pages.render_emotions()
