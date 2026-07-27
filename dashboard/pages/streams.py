"""Streams: locked for now.

The three stream tabs (Visual, Lip-Sync, Emotions) and the configurable model
boxes behind them are built and still unit-tested. They are just not reachable
while the visual stream is re-validated after alignment changed the cached pixels.

To unlock: drop the icon from app.py's nav entry, and call
stream_pages.render_visual / render_lipsync / render_emotions from here in tabs.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import locked
from dashboard.lib.stream_spec import STREAMS

locked.render(st, STREAMS)
st.caption("The stream designs are documented in full on the Documentation page, under "
           "*Stream models*.")
