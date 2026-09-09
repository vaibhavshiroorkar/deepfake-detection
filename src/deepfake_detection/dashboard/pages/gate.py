"""Evidence gate: upload a clip, run the frozen visual baseline, read its limits.

This is the one page that produces a verdict, and the only page bound to the
frozen provenance in `configuration.py`. Everything else in the dashboard
teaches a stage; this page runs the shipped model over one clip and states what
the result is allowed to mean.

Four steps in one page rather than four pages. Each is an expander showing its
own waiting/ready/complete state, so the order is visible without navigating,
and derived state is keyed by the upload's sha256, so a new upload invalidates
preprocessing and the prediction rather than leaving a stale one on screen.
"""

import streamlit as st

from deepfake_detection.dashboard.components import render_step_status
from deepfake_detection.dashboard.sections.prediction import render_prediction
from deepfake_detection.dashboard.sections.preprocessing import render_preprocessing
from deepfake_detection.dashboard.sections.video_input import render_video_input
from deepfake_detection.dashboard.sections.visual_model import render_visual_model
from deepfake_detection.dashboard.state import (
    UploadedClip,
    prediction_for_upload,
    prepared_for_upload,
    uploaded_clip,
)
from deepfake_detection.dashboard.workflow import (
    StepState,
    WorkflowState,
    workflow_state,
)


def _current_workflow() -> tuple[UploadedClip | None, WorkflowState]:
    clip = uploaded_clip(st.session_state)
    prepared = prepared_for_upload(st.session_state, clip.sha256) if clip else None
    prediction = prediction_for_upload(st.session_state, clip.sha256) if clip else None
    return clip, workflow_state(
        has_upload=clip is not None,
        has_prepared=prepared is not None,
        has_prediction=prediction is not None,
    )


st.markdown(
    '<div class="thesis">Verdict follows <span>coverage.</span></div>',
    unsafe_allow_html=True,
)
st.write(
    "Inspect one talking-head video. Every result states which evidence was used "
    "and which research limits still apply."
)

clip, flow = _current_workflow()
with st.expander("1. Video input", expanded=clip is None):
    render_step_status(flow.video)
    render_video_input(embedded=True)

clip, flow = _current_workflow()
with st.expander(
    "2. Preprocessing",
    expanded=clip is not None and flow.preprocessing is not StepState.COMPLETE,
):
    render_step_status(flow.preprocessing)
    render_preprocessing(embedded=True)

clip, flow = _current_workflow()
with st.expander("3. Visual model"):
    render_step_status(flow.visual_model)
    render_visual_model(embedded=True)

clip, flow = _current_workflow()
with st.expander("4. Prediction", expanded=flow.prediction is StepState.READY):
    render_step_status(flow.prediction)
    render_prediction(embedded=True)

st.caption(
    "The teaching pages in the sidebar run a configurable stream over any clip. "
    "This page runs only the frozen baseline, which is the one result with "
    "provenance behind it."
)
