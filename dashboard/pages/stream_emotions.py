import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import EMOTION_STREAM as S

st.title(S["title"])
st.info(S["status"])
st.caption(S["note"])

for model_name, desc in S["models"]:
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            st.toggle("Enable", value=True, key=f"emotion_{model_name}_enabled")
            st.markdown(f"### {model_name}")
            st.caption(desc)
        with cfg:
            st.selectbox("Weights", ["(not downloaded)"], key=f"emotion_{model_name}_weights",
                         disabled=True)
            st.slider("Embedding dim", 128, 512, 256, step=64, key=f"emotion_{model_name}_dim",
                      disabled=True)
            st.button("Build & inspect", key=f"emotion_{model_name}_build", disabled=True,
                      help="Available once the emotion stream is built (Stage 5).")

st.divider()
st.caption("Read-only scaffold — the cross-attention model and its metrics appear here "
           "after Stage 5. This page never trains (§7).")
