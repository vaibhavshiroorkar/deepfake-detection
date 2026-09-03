import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Planned evidence",
    "Audio branch",
    "See what remains before audio spoof evidence can support a research claim.",
)
render_status(PageState.PROTOTYPE)

st.subheader("Prototype teaching path")
st.write("Input: a four-second, 16 kHz waveform with shape [B, S].")
st.write(
    "Processing: Wav2Vec2 Base produces temporal tokens. A projection, learned "
    "attention pool, and linear head produce an audio-spoof logit."
)
st.write("Output: one uncalibrated audio logit and a clip embedding.")
st.info("This is a prototype. Full training is incomplete.")
st.write(
    "No prototype checkpoint is loaded here and this page does not calculate a "
    "probability."
)
st.write(
    "Unlock condition: train and evaluate the branch with valid-length masks, "
    "source-disjoint evidence, and the planned candidate comparison."
)
