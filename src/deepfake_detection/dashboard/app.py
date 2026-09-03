from __future__ import annotations

from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.components import render_step_status
from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.pages.audio_branch import render_audio_branch
from deepfake_detection.dashboard.pages.documentation import render_documentation
from deepfake_detection.dashboard.pages.experiments import render_experiments
from deepfake_detection.dashboard.pages.fusion import render_fusion
from deepfake_detection.dashboard.pages.prediction import render_prediction
from deepfake_detection.dashboard.pages.preprocessing import render_preprocessing
from deepfake_detection.dashboard.pages.sync_branch import render_sync_branch
from deepfake_detection.dashboard.pages.video_input import render_video_input
from deepfake_detection.dashboard.pages.visual_model import render_visual_model
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

_THRESHOLD = 0.5
_DEVICE = "cuda"

st.set_page_config(
    page_title="Evidence Gate",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --paper: #F2F6F8;
        --ink: #14212B;
        --cobalt: #2457A6;
        --amber: #D98718;
        --evidence: #B73B45;
        --teal: #2C7A78;
        --line: #CBD7DE;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    h1, h2, h3 { font-family: Bahnschrift, "Arial Narrow", sans-serif; }
    p, label, button { font-family: Aptos, Calibri, sans-serif; }
    code, .score { font-family: "Cascadia Mono", Consolas, monospace; }
    .thesis {
        max-width: 840px;
        margin: 1.5rem 0 2.5rem;
        font: 600 clamp(2.5rem, 7vw, 5.8rem)/0.94 Bahnschrift, sans-serif;
        letter-spacing: -0.045em;
        color: var(--ink);
    }
    .thesis span { color: var(--cobalt); }
    .gate {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 2rem;
    }
    .channel {
        border: 1px solid var(--line);
        padding: 1rem;
        background: white;
    }
    .channel.available { border-top: 5px solid var(--teal); }
    .channel.missing { border-top: 5px solid var(--amber); }
    .channel strong {
        display: block;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .scope {
        display: inline-block;
        margin: 0 0 1rem;
        padding: .35rem .55rem;
        border: 1px solid var(--cobalt);
        color: var(--cobalt);
        font: 700 .78rem/1 "Cascadia Mono", Consolas, monospace;
        letter-spacing: .06em;
        text-transform: uppercase;
    }
    .limits {
        border: 1px solid var(--line);
        background: #E8EFF3;
        padding: 1rem 1.25rem;
        margin: 1rem 0 1.5rem;
    }
    .result {
        border-left: 8px solid var(--cobalt);
        background: white;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .result.fake { border-color: var(--evidence); }
    .result.indeterminate { border-color: var(--amber); }
    .result .score { font-size: 2rem; font-weight: 700; }
    .step-state {
        font: 700 .76rem/1 "Cascadia Mono", Consolas, monospace;
        letter-spacing: .06em;
        margin-bottom: .75rem;
        text-transform: uppercase;
    }
    .step-state.waiting { color: var(--amber); }
    .step-state.ready { color: var(--cobalt); }
    .step-state.complete { color: var(--teal); }
    .stButton button:focus-visible, input:focus-visible,
    [tabindex]:focus-visible {
        outline: 3px solid var(--cobalt) !important;
        outline-offset: 2px;
    }
    @media (max-width: 700px) {
        .gate { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

defaults = dashboard_defaults(root=Path.cwd())

st.markdown(
    '<div class="thesis">Verdict follows <span>coverage.</span></div>',
    unsafe_allow_html=True,
)
st.write(
    "Inspect one talking-head video. Every result states which evidence was used "
    "and which research limits still apply."
)

with st.sidebar:
    st.header("Frozen baseline")
    st.write("Visual-only EfficientNet-B0 plus GRU")
    st.caption(f"Checkpoint: {defaults.visual_checkpoint.name}")
    st.caption(f"Run: {defaults.run_id}")
    st.caption(f"Decision threshold: {_THRESHOLD:.2f}")
    st.caption(f"Compute device: {_DEVICE}")


def _current_workflow() -> tuple[UploadedClip | None, WorkflowState]:
    clip = uploaded_clip(st.session_state)
    prepared = prepared_for_upload(st.session_state, clip.sha256) if clip else None
    prediction = prediction_for_upload(st.session_state, clip.sha256) if clip else None
    return clip, workflow_state(
        has_upload=clip is not None,
        has_prepared=prepared is not None,
        has_prediction=prediction is not None,
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
with st.expander(
    "4. Prediction",
    expanded=flow.prediction is StepState.READY,
):
    render_step_status(flow.prediction)
    render_prediction(embedded=True)

with st.expander("Research and methodology"):
    experiments, audio, sync, fusion, documentation = st.tabs(
        (
            "Experiments",
            "Audio branch",
            "Sync branch",
            "Fusion",
            "Documentation",
        )
    )
    with experiments:
        render_experiments(embedded=True)
    with audio:
        render_audio_branch(embedded=True)
    with sync:
        render_sync_branch(embedded=True)
    with fusion:
        render_fusion(embedded=True)
    with documentation:
        render_documentation(embedded=True)
