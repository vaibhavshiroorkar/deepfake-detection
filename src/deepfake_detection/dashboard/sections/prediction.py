from html import escape
from pathlib import Path

import streamlit as st

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.components import (
    render_page_header,
    render_status,
    require_upload,
)
from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.state import (
    clear_prediction_for_upload,
    prediction_for_upload,
    prepared_for_upload,
    store_prediction,
)
from deepfake_detection.dashboard.status import PageState
from deepfake_detection.dashboard.view_model import DashboardView, build_view_model
from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip


def _failure_guidance(
    error: AssertionError | OSError | RuntimeError | ValueError,
) -> str:
    detail = str(error).strip()
    if "cuda" in detail.lower():
        # runtime.require_cuda already says which of the two states this is and
        # what to do about it, so pass its sentence through rather than
        # flattening it back to a generic one.
        if detail.startswith("CUDA is unavailable:"):
            return detail
        return (
            "CUDA is unavailable on this server. Run the dashboard on a "
            "CUDA-enabled host with a working PyTorch driver."
        )
    return f"Analysis failed: {detail or type(error).__name__}"


def _coverage_label(prepared: PreparedClip | None) -> str:
    if prepared is None:
        return "Not available. Run preprocessing to measure face coverage."
    return f"{prepared.quality.face_coverage:.1%}"


def _result_markup(view: DashboardView) -> tuple[str, ...]:
    scope = f'<div class="scope">{escape(view.mode_label)}</div>'
    result = (
        f'<div class="result {escape(view.verdict)}"><h2>{escape(view.title)}</h2>'
        f'<div class="score">{escape(view.final_score)}</div>'
        f"<div>{escape(view.threshold_label)}</div></div>"
    )
    gate = (
        '<div class="gate">'
        + "".join(
            f'<div class="channel {escape(status)}">'
            f"<strong>{escape(name)}</strong>{escape(status)}</div>"
            for name, status in view.channels.items()
        )
        + "</div>"
    )
    markup = [scope, result, gate]
    if view.limitations:
        markup.append(
            '<div class="limits"><strong>Research limits</strong><br>'
            + "<br>".join(escape(limitation) for limitation in view.limitations)
            + "</div>"
        )
    return tuple(markup)


def _render_result(
    result: PredictionResult,
    prepared: PreparedClip | None,
) -> None:
    view = build_view_model(result, threshold=0.5)
    for markup in _result_markup(view):
        st.markdown(markup, unsafe_allow_html=True)

    if view.blockers:
        st.subheader("Why no final verdict was issued")
        for blocker in view.blockers:
            st.write(f"- {blocker.replace('_', ' ')}")

    defaults = dashboard_defaults(root=Path.cwd())
    with st.expander("Technical details"):
        st.markdown(f"**Visual coverage:** {_coverage_label(prepared)}")
        st.markdown("**Branch logits**")
        st.json(view.branch_scores)
        st.markdown(f"**Run ID:** `{defaults.run_id}`")
        st.markdown(f"**Checkpoint hash:** `{defaults.checkpoint_sha256}`")
        st.markdown(f"**Split hash:** `{defaults.split_hash}`")
        st.markdown(
            f"**Preprocessing fingerprint:** `{view.preprocessing_fingerprint}`"
        )


def render_prediction(*, embedded: bool = False) -> None:
    if not embedded:
        render_page_header(
            "Stage 4",
            "4. Prediction",
            "Run the frozen visual baseline and read its evidence limits.",
        )
        render_status(PageState.READY)

    clip = require_upload()

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
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                st.error(_failure_guidance(error))
            else:
                store_prediction(st.session_state, clip.sha256, result)
        if result is not None:
            _render_result(result, prepared)


if __name__ == "__main__":
    render_prediction()
