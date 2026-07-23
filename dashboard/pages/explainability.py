import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import EXPLAINABILITY as S

st.title(S["title"])
st.info(S["status"])

for view_name, desc in S["views"]:
    with st.container(border=True):
        st.markdown(f"### {view_name}")
        st.caption(desc)
        st.button("Open", key=f"xai_{view_name}_open", disabled=True,
                  help="Available after the streams and fusion are trained (Stage 10).")

st.divider()
st.caption("Read-only scaffold — explainability views populate once there is a trained "
           "model to explain (Stage 10). This page never trains (§7).")
