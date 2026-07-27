"""Fusion — locked until Stage 6.

Visible in the sidebar so the shape of the system is obvious, but nothing here
is interactive: the fusion MLP does not exist yet. The controls that will live
here are described, not stubbed, so there is no half-working UI to mistake for
the real thing. Design rationale lives on the Documentation page.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import FUSION as S

st.title(f":material/lock: {S['title']}")
st.info(f"**Locked.** {S['status']}")
st.caption(S["note"])

st.subheader("What lands here")
st.markdown("""
- **Stream selection** — which of the five streams are concatenated into the fusion input.
- **Fusion MLP** — hidden sizes, dropout, and the resulting input dimension (streams × 256).
- **Ablation view** — fused metrics across stream subsets, the Stage-7 result.
""")
st.caption("Read the fusion design on the **Documentation** page → *Fusion & evaluation*.")
