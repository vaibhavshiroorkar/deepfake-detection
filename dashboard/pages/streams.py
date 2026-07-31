"""Streams hub: quick control over all three streams, and the way into each one.

Configure here, run there. This page owns the settings that decide what each
stream *is* (which temporal model, how wide, frozen or not) and which streams
fusion will see. It deliberately runs nothing: a single number reported next to a
config box was the least informative thing the old page did, and the subpages
replace it with the whole forward pass.

The settings live in session_state via dashboard/lib/stream_pages.py, so a
subpage opens with whatever was set here.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import stream_pages
from dashboard.lib.stream_spec import STREAMS

st.title("Streams")
st.caption(STREAMS["note"])

stream_pages.render_inherited_clip(st)

st.header("Visual")
st.caption("Three artifact-focused backbones over the same face-crop sequence, never audio. "
           "All three are one config-driven module with a different backbone name.")

for key, (name, backbone) in stream_pages.VISUAL_MODELS.items():
    current = stream_pages.settings(key)
    with st.container(border=True):
        head, cfg = st.columns([1, 2])
        with head:
            current["enabled"] = st.toggle("Enable", value=bool(current["enabled"]),
                                           key=f"hub_{key}_enabled")
            st.markdown(f"### {name}")
            st.caption(f"`{backbone}`")
            st.caption("Included in fusion" if current["enabled"] else "Excluded from fusion")
        stream_pages.render_config_controls(cfg, key, ns="hub")
        # A button rather than a page_link: the subpage shows one backbone at a
        # time, so which one you clicked has to travel with you. st.switch_page
        # cannot run from a widget callback, hence the plain if.
        if st.button(f"Open {name} step by step", key=f"hub_{key}_open", width="stretch",
                     icon=":material/arrow_forward:"):
            st.session_state["visual_backbone"] = key
            st.switch_page("pages/stream_visual.py")

st.header("Cross-modal")
st.caption("Both compare two modalities by cross-attention rather than classifying either one.")
c1, c2 = st.columns(2)
for column, (key, (name, encoders, stage)) in zip((c1, c2), stream_pages.CROSS_MODAL.items()):
    with column, st.container(border=True):
        st.markdown(f"### {name}")
        st.caption(f"{encoders}  ·  Stage {stage}, not built")
        if st.button(f"Open {name}", key=f"hub_{key}_open", width="stretch",
                     icon=":material/arrow_forward:"):
            st.switch_page(f"pages/stream_{key}.py")

st.divider()
st.header("What fusion will see")
enabled = stream_pages.enabled_streams()
dims = [stream_pages.settings(key)["dim"] for key in enabled]
m1, m2 = st.columns(2)
m1.metric("Visual streams enabled", f"{len(enabled)} of {len(stream_pages.VISUAL_MODELS)}")
m2.metric("Concatenated width", sum(dims) if enabled else 0,
          help="The visual part of the fusion input: each enabled stream's embedding dim, "
               "concatenated. The two cross-modal streams add to this from Stage 4.")
if enabled:
    st.code(" + ".join(f"{stream_pages.VISUAL_MODELS[k][0]} [{d}]"
                       for k, d in zip(enabled, dims))
            + f"  ->  [{sum(dims)}]", language="text")
else:
    st.warning("Every visual stream is disabled, so fusion would have no visual evidence at all.")
st.caption("Which streams to keep is a Stage-7 ablation result, not a decision to make here. "
           "The toggles set what the Fusion page will read once it exists.")
