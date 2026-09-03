import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Evidence gate",
    "Fusion",
    "See how calibrated visual, audio, and sync evidence will enter late fusion.",
)
render_status(PageState.LOCKED)

st.subheader("Locked teaching path")
st.write(
    "Input: calibrated visual, audio, and synchronization logits with quality features."
)
st.write(
    "Processing: strict provenance checks, per-branch calibration, then logistic "
    "late fusion over complete evidence."
)
st.write("Output: a fused probability only when every required branch is available.")
st.error(
    "Fusion is locked. The current artifact is a software fixture, not a trained "
    "research fusion model."
)
st.write(
    "This page does not load the fixture or calculate a probability. Unlock it "
    "after genuine out-of-fold branch features support a trained fusion artifact."
)
