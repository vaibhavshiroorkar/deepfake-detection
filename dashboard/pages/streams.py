"""Streams: locked.

Dimmed and unclickable in the sidebar. The three stream tabs (Visual, Lip-Sync,
Emotions) and the configurable model boxes behind them are built and still
unit-tested in dashboard/lib/stream_pages.py; they are not reachable while the
visual stream is re-validated after alignment changed the cached pixels.

To unlock: drop STREAMS from locked.LOCKED, then replace the body below with
tabs calling stream_pages.render_visual / render_lipsync / render_emotions.
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
