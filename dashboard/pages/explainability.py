"""Explainability — locked until Stage 10.

Visible in the sidebar, but there is nothing to explain until a trained model
exists, so the views are listed rather than stubbed as dead buttons.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import EXPLAINABILITY as S

st.title(f":material/lock: {S['title']}")
st.info(f"**Locked.** {S['status']}")
st.caption("Explainability needs a trained model to explain; these views populate after fusion "
           "is trained.")

st.subheader("What lands here")
st.markdown("\n".join(f"- **{name}** — {desc}" for name, desc in S["views"]))
