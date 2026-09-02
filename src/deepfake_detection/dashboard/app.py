from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.view_model import build_view_model
from deepfake_detection.inference.loading import (
    VisualInferenceConfig,
    load_visual_prediction_engine,
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
    .channel strong { display: block; text-transform: uppercase; letter-spacing: .08em; }
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
def _visual_engine():
    return load_visual_prediction_engine(
        VisualInferenceConfig(
            visual_checkpoint=defaults.visual_checkpoint,
            code_version=defaults.code_version,
            expected_checkpoint_sha256=defaults.checkpoint_sha256,
            expected_run_id=defaults.run_id,
            expected_split_hash=defaults.split_hash,
            expected_git_commit=defaults.git_commit,
            expected_seed=defaults.seed,
            threshold=_THRESHOLD,
            device=_DEVICE,
        )
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

upload = st.file_uploader("Video file", type=("mp4", "mov", "mkv", "avi"))

if st.button("Analyze video", type="primary", disabled=upload is None):
    if not defaults.visual_checkpoint.is_file():
        st.error(f"Visual checkpoint is missing: {defaults.visual_checkpoint}")
    else:
        suffix = Path(upload.name).suffix or ".mp4"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temporary_path = Path(handle.name)
                handle.write(upload.getbuffer())
            with st.spinner("Checking evidence coverage and model scores"):
                result = _visual_engine().predict(temporary_path)
            view = build_view_model(result, threshold=_THRESHOLD)
            st.markdown(
                f'<div class="scope">{view.mode_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="result {view.verdict}"><h2>{view.title}</h2>'
                f'<div class="score">{view.final_score}</div>'
                f"<div>{view.threshold_label}</div></div>",
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
            if view.limitations:
                st.markdown(
                    '<div class="limits"><strong>Research limits</strong><br>'
                    + "<br>".join(view.limitations)
                    + "</div>",
                    unsafe_allow_html=True,
                )
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
