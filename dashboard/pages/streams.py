"""Streams — locked for now.

The three stream tabs (Visual, Lip-Sync, Emotions) and the configurable model
boxes behind them are built and still unit-tested; they are just not reachable
while the visual stream is being re-validated after alignment changed the cached
pixels. Unlocking is a two-line change: drop the icon from app.py's nav entry
and call stream_pages.render_visual/render_lipsync/render_emotions from here
again, in the tabs they used to live in.

Locked the same way as Fusion and Explainability: visible in the sidebar so the
shape of the system stays obvious, with what lands here described rather than
stubbed as dead controls.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import STREAMS as S

st.title(f":material/lock: {S['title']}")
st.info(f"**Locked.** {S['status']}")
st.caption(S["note"])

st.subheader("What lands here")
st.markdown("\n".join(f"- **{name}** — {desc}" for name, desc in S["views"]))

st.caption("The stream designs are documented in full on the **Documentation** page → "
           "*Streams*.")
