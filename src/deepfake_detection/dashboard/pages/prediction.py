from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.components import (
    render_page_header,
    render_status,
    require_upload,
)
from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.navigation import PageState
from deepfake_detection.dashboard.state import (
    clear_prediction_for_upload,
    prediction_for_upload,
    prepared_for_upload,
    store_prediction,
)
from deepfake_detection.dashboard.view_model import DashboardView, build_view_model
from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip


def _failure_guidance(error: OSError | RuntimeError | ValueError) -> str:
    detail = str(error).strip()
    if "cuda" in detail.lower():
        return (
            "CUDA is unavailable on this server. Run the dashboard on a "
            "CUDA-enabled host with a working PyTorch driver."
        )
    return f"Analysis failed: {detail or type(error).__name__}"


def _coverage_label(prepared: PreparedClip | None) -> str:
    if prepared is None:
        return "Not available. Run preprocessing to measure face coverage."
    return f"{prepared.quality.face_coverage:.1%}"


def _render_result(
    result: PredictionResult,
    prepared: PreparedClip | None,
) -> None:
    view: DashboardView = build_view_model(result)
    st.subheader(view.title)
    st.markdown(f"**Mode:** {view.mode_label}")
    st.markdown(f"**Visual classifier probability:** {view.final_score}")
    st.markdown(f"**Visual coverage:** {_coverage_label(prepared)}")
    st.markdown(
        f"**Visual logit:** {view.branch_scores.get('visual', 'Not available')}"
    )
    blockers = ", ".join(view.blockers) if view.blockers else "None"
    st.markdown(f"**Blockers:** {blockers}")
    st.markdown("**Limitations**")
    for limitation in view.limitations:
        st.markdown(f"- {limitation}")

    defaults = dashboard_defaults(root=Path.cwd())
    with st.expander("Technical details"):
        st.markdown(f"**Run ID:** `{defaults.run_id}`")
        st.markdown(f"**Checkpoint hash:** `{defaults.checkpoint_sha256}`")
        st.markdown(f"**Split hash:** `{defaults.split_hash}`")
        st.markdown(
            f"**Preprocessing fingerprint:** `{view.preprocessing_fingerprint}`"
        )


render_page_header(
    "Stage 4",
    "4. Prediction",
    "Run the frozen visual baseline and read its evidence limits.",
)
render_status(PageState.READY)

try:
    clip = require_upload()
except KeyError as error:
    if error.args != ("url_pathname",):
        raise
    clip = None

if clip is not None:
    prepared = prepared_for_upload(st.session_state, clip.sha256)
    result = prediction_for_upload(st.session_state, clip.sha256)
    st.markdown("**Fixed decision threshold: 0.50**")
    st.caption("Visual-only development baseline")
    if st.button(
        "Analyze video",
        key="analyze_video",
        type="primary",
        use_container_width=True,
    ):
        clear_prediction_for_upload(st.session_state, clip.sha256)
        result = None
        try:
            result = runtime.predict_upload(clip)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(_failure_guidance(error))
        else:
            store_prediction(st.session_state, clip.sha256, result)
    if result is not None:
        _render_result(result, prepared)
