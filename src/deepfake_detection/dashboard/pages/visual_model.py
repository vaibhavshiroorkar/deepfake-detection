import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState
from deepfake_detection.dashboard.state import (
    prediction_for_upload,
    prepared_for_upload,
    uploaded_clip,
)

MODEL_STAGES = (
    ("Face sequence", "[B, 16, 3, 224, 224]"),
    ("Frame encoder", "[B*16, 1280]"),
    ("Sequence restore", "[B, 16, 1280]"),
    ("GRU summary", "[B, 256]"),
    ("Visual logit", "[B]"),
)

_STAGE_NOTES = (
    "Sixteen normalized RGB face crops form one ordered clip input.",
    "EfficientNet-B0 encodes each face crop as 1,280 learned features.",
    "The frame features return to clip order before temporal modeling.",
    "A GRU condenses the ordered frame features into one 256-value summary.",
    "A linear head emits one logit. Sigmoid maps that logit to a probability.",
)


def _render_stage(index: int, name: str, shape: str, note: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {index}. {name}")
        st.code(shape, language=None)
        st.write(note)


render_page_header(
    "Stage 3",
    "3. Visual model",
    "Trace frame features through the frozen visual classifier.",
)
render_status(PageState.READY)

for stage_index, ((stage_name, stage_shape), stage_note) in enumerate(
    zip(MODEL_STAGES, _STAGE_NOTES, strict=True), start=1
):
    _render_stage(stage_index, stage_name, stage_shape, stage_note)

st.info(
    "These intermediate values are model computations, not an explanation of "
    "why a clip is authentic or manipulated."
)

clip = uploaded_clip(st.session_state)
prepared = prepared_for_upload(st.session_state, clip.sha256) if clip else None
prediction = prediction_for_upload(st.session_state, clip.sha256) if clip else None

if prepared is not None and prepared.visual_view is not None:
    st.subheader("Current input")
    st.markdown(f"**Prepared tensor shape:** `{prepared.visual_view.shape}`")

if prediction is None:
    st.caption(
        "Prediction runs the frozen classifier and returns its visual logit and "
        "sigmoid probability."
    )
else:
    st.subheader("Current classifier output")
    visual_logit = prediction.branch_logits.get("visual")
    if visual_logit is not None:
        st.markdown(f"**Visual logit:** `{visual_logit:+.3f}`")
    probability = prediction.probability
    if probability is not None:
        st.markdown(f"**Sigmoid probability:** `{probability:.1%}`")
