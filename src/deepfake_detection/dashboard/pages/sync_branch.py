import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Planned evidence",
    "Sync branch",
    "See how mouth-audio alignment will form a temporal evidence stream.",
)
render_status(PageState.PROTOTYPE)

st.subheader("Prototype teaching path")
st.write(
    "Input: mouth frames [B, T, C, H, W] and a time-aligned audio waveform [B, S]."
)
st.write(
    "Processing: ResNet-18 encodes mouth frames, Wav2Vec2 encodes audio, and "
    "temporal Transformers compare aligned tokens across eight offset classes."
)
st.write("Output: an offset-class vector and a raw synchronization anomaly logit.")
st.info("This is a prototype. Full training is incomplete.")
st.write(
    "No prototype checkpoint is loaded here and this page does not calculate a "
    "probability."
)
st.write(
    "Unlock condition: complete source-disjoint training and evaluate the "
    "alignment objective before using synchronization evidence."
)
