from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.view_model import build_view_model
from deepfake_detection.inference.loading import InferenceConfig, load_prediction_engine

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
    .channel strong { display: block; text-transform: uppercase; letter-spacing: .08em; }
    .result {
        border-left: 8px solid var(--cobalt);
        background: white;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .result.fake { border-color: var(--evidence); }
    .result.indeterminate { border-color: var(--amber); }
    .result .score { font-size: 2rem; font-weight: 700; }
    .stButton button:focus-visible, input:focus-visible {
        outline: 3px solid var(--cobalt) !important;
        outline-offset: 2px;
    }
    @media (max-width: 700px) {
        .gate { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _engine(
    visual_checkpoint: str,
    audio_checkpoint: str,
    sync_checkpoint: str,
    fusion_model: str,
    code_version: str,
    threshold: float,
    device: str,
):
    return load_prediction_engine(
        InferenceConfig(
            visual_checkpoint=Path(visual_checkpoint),
            audio_checkpoint=Path(audio_checkpoint),
            sync_checkpoint=Path(sync_checkpoint),
            fusion_model=Path(fusion_model),
            code_version=code_version,
            threshold=threshold,
            device=device,
        )
    )


st.markdown(
    '<div class="thesis">Verdict follows <span>coverage.</span></div>',
    unsafe_allow_html=True,
)
st.write(
    "Inspect one talking-head video. The system issues a verdict only when visual, "
    "audio, and synchronization evidence are all available."
)

with st.sidebar:
    st.header("Model files")
    visual_checkpoint = st.text_input("Visual checkpoint")
    audio_checkpoint = st.text_input("Audio checkpoint")
    sync_checkpoint = st.text_input("Sync checkpoint")
    fusion_model = st.text_input("Fusion model")
    code_version = st.text_input("Preprocessing version", value="v1")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    device = st.selectbox("Compute device", ("cuda", "cpu"))

upload = st.file_uploader("Video file", type=("mp4", "mov", "mkv", "avi"))

if st.button("Analyze video", type="primary", disabled=upload is None):
    configured = {
        "Visual checkpoint": visual_checkpoint,
        "Audio checkpoint": audio_checkpoint,
        "Sync checkpoint": sync_checkpoint,
        "Fusion model": fusion_model,
    }
    missing = [name for name, value in configured.items() if not value]
    if missing:
        st.error(f"Set these model files first: {', '.join(missing)}")
    else:
        suffix = Path(upload.name).suffix or ".mp4"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temporary_path = Path(handle.name)
                handle.write(upload.getbuffer())
            with st.spinner("Checking evidence coverage and model scores"):
                result = _engine(
                    visual_checkpoint,
                    audio_checkpoint,
                    sync_checkpoint,
                    fusion_model,
                    code_version,
                    threshold,
                    device,
                ).predict(temporary_path)
            view = build_view_model(result)
            st.markdown(
                f'<div class="result {view.verdict}"><h2>{view.title}</h2>'
                f'<div class="score">{view.final_score}</div></div>',
                unsafe_allow_html=True,
            )
            gate = (
                '<div class="gate">'
                + "".join(
                    f'<div class="channel {status}"><strong>{name}</strong>{status}</div>'
                    for name, status in view.channels.items()
                )
                + "</div>"
            )
            st.markdown(gate, unsafe_allow_html=True)
            if view.blockers:
                st.subheader("Why no final verdict was issued")
                for blocker in view.blockers:
                    st.write(f"- {blocker.replace('_', ' ')}")
            st.subheader("Branch logits")
            st.json(view.branch_scores)
            st.caption(f"Preprocessing fingerprint: {view.preprocessing_fingerprint}")
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Analysis failed: {error}")
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
