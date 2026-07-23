import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dashboard.lib.stream_spec import FUSION as S

st.title(S["title"])
st.info(S["status"])
st.caption(S["note"])

# Reflect the toggles set on the Visual streams page (best-effort — scaffold).
visual = st.session_state.get("stream_visual_enabled", {})
enabled = [name for name, on in visual.items() if on] or ["efficientnet", "xception"]

st.subheader("Streams to fuse")
all_streams = ["efficientnet", "xception", "dinov2", "lipsync", "emotion"]
built = {"efficientnet", "xception"}
for s in all_streams:
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        on = c1.toggle(s, value=(s in enabled and s in built), key=f"fuse_{s}",
                       disabled=s not in built)
        status = "built" if s in built else "not built yet"
        c2.caption(f"clip embedding (256-d) → concat → MLP → sigmoid  ·  {status}")

st.divider()
n = sum(1 for s in all_streams if st.session_state.get(f"fuse_{s}"))
st.metric("Fusion MLP input dim", f"{n} × 256 = {n * 256}")
st.caption("Read-only scaffold — the fusion MLP is built in Stage 6; the ablation over "
           "stream subsets is Stage 7. This page never trains (§7).")
