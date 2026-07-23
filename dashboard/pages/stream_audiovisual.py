import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import AUDIOVISUAL_STREAM as S

st.title(S["title"])
st.info(S["status"])
st.subheader("Planned architecture")
for line in S["architecture"]:
    st.markdown(f"- {line}")
st.divider()
st.caption("W&B run metrics appear here after training. Read-only — this page never "
           "runs a model (PROJECT_OVERVIEW §7).")
