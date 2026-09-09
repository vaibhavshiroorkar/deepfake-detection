"""Fusion: locked until Stage 6.

Dimmed and unclickable in the sidebar. The fusion MLP does not exist yet, so the
controls that will live here are described rather than stubbed. Design rationale
is on the Documentation page.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import locked
from deepfake_detection.dashboard.lib.stream_spec import FUSION
from deepfake_detection.dashboard.sections.fusion import render_fusion

locked.render(st, FUSION)

render_fusion(embedded=True)

st.caption("The fusion design is documented in full on the Documentation page, under "
           "*Fusion & evaluation*.")
