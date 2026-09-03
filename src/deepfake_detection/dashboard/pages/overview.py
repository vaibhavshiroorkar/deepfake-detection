import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState

render_page_header(
    "Project map",
    "Overview",
    "Follow the detector from an uploaded clip to an evidence-limited prediction.",
)
render_status(PageState.READY)

st.subheader("The research question")
st.markdown(
    "How can a detector assess a talking-head clip while keeping development "
    "evidence separate from a generalization claim?"
)

st.subheader("Pipeline map")
for title, description in (
    (
        "1. Visual evidence",
        "Sample face frames and pass them to the visual development baseline.",
    ),
    (
        "2. Audio evidence",
        "Audio spoof evidence is a prototype and does not yet produce a result.",
    ),
    (
        "3. Sync evidence",
        "Mouth-audio timing is planned as a separate prototype evidence stream.",
    ),
    (
        "4. Late fusion",
        "Fusion will combine calibrated visual, audio, and sync evidence when "
        "all branches are evaluated.",
    ),
):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(description)

st.subheader("Dataset state")
for name, state in (
    (
        "FakeAVCeleb",
        "Current development data for the visual baseline.",
    ),
    (
        "Celeb-DF-v2",
        "Declared for later visual experiments. It does not support a current "
        "dashboard claim.",
    ),
    (
        "FaceForensics++",
        "Paused. Its incomplete download is not used for a current result.",
    ),
    (
        "MNW",
        "Evaluation-only. It cannot be used for training, validation, model "
        "selection, or threshold selection.",
    ),
):
    st.markdown(f"**{name}:** {state}")

st.subheader("Current research boundary")
st.markdown(
    "The visual development baseline is the only runnable detector here. It is "
    "a development result, not a generalization result. Audio, sync, and fusion "
    "remain unfinished research stages."
)
