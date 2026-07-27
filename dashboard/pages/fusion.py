"""Fusion: locked until Stage 6.

Visible in the sidebar so the shape of the system stays legible, but the fusion
MLP does not exist yet, so the controls that will live here are described rather
than stubbed. Design rationale is on the Documentation page.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import locked
from dashboard.lib.stream_spec import FUSION

locked.render(st, FUSION)
st.caption("The fusion design is documented in full on the Documentation page, under "
           "*Fusion & evaluation*.")
