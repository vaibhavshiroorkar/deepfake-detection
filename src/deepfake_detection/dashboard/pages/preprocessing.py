from __future__ import annotations

import streamlit as st

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.components import (
    render_page_header,
    render_status,
    require_upload,
)
from deepfake_detection.dashboard.navigation import PageState
from deepfake_detection.dashboard.state import (
    clear_prepared_for_upload,
    prepared_for_upload,
    store_prepared,
)
from deepfake_detection.views.contracts import PreparedClip

PREPROCESSING_STAGES = (
    "Media probe",
    "Timestamp sampling",
    "Face detection and tracking",
    "Face crop",
    "Resize and normalization",
    "Model tensor",
)

_STAGE_NOTES = (
    "Read duration, frame rate, and stream metadata before decoding.",
    "Choose sixteen timestamps spread across the clip content.",
    "Detect faces with MTCNN and join them with greedy IoU tracking.",
    "Expand the primary face box and crop one square region per timestamp.",
    "Resize each crop and apply the frozen ImageNet channel statistics.",
    "Stack sixteen normalized RGB crops as a model-ready NCHW tensor.",
)


def _failure_guidance(error: OSError | RuntimeError | ValueError) -> str:
    detail = str(error).strip()
    lowered = detail.lower()
    if "cuda" in lowered:
        return (
            "CUDA is unavailable on this server. Run the dashboard on a "
            "CUDA-enabled host with a working PyTorch driver."
        )
    if "preprocessing" in lowered and (
        "checkpoint" in lowered or "provenance" in lowered
    ):
        return (
            "Preprocessing provenance does not match the frozen checkpoint. "
            "Use the configured code version and preprocessing identity."
        )
    if "face" in lowered:
        return (
            "No stable face view could be prepared. Use a clip with one clear, "
            "well-lit face visible throughout."
        )
    if isinstance(error, OSError):
        return (
            "Video decoding failed. Check that the upload is readable and uses "
            "a codec supported by FFmpeg."
        )
    return f"Preprocessing failed: {detail or type(error).__name__}"


def _render_stages() -> None:
    for index, (name, note) in enumerate(
        zip(PREPROCESSING_STAGES, _STAGE_NOTES, strict=True), start=1
    ):
        with st.container(border=True):
            st.markdown(f"### {index}. {name}")
            st.write(note)


def _quality_blockers(prepared: PreparedClip) -> tuple[str, ...]:
    blockers: list[str] = []
    if not prepared.quality.stable_face_track:
        blockers.append("unstable face track")
    if prepared.quality.face_coverage < 0.80:
        blockers.append("low face coverage")
    return tuple(blockers)


def _render_output(prepared: PreparedClip) -> None:
    st.subheader("Prepared visual evidence")
    st.markdown(f"**Face coverage:** {prepared.quality.face_coverage:.1%}")
    track_status = "Stable" if prepared.quality.stable_face_track else "Unstable"
    st.markdown(f"**Face track:** {track_status}")
    st.markdown(f"**Preprocessing hash:** `{prepared.preprocessing_config_hash}`")
    if prepared.visual_view is None:
        blockers = _quality_blockers(prepared)
        blocker_text = ", ".join(blockers) if blockers else "missing face view"
        st.info(
            "No model tensor was produced. Quality blockers: "
            f"{blocker_text}. Use a clip with one clear face visible throughout."
        )
        return

    st.markdown(f"**Model tensor shape:** `{prepared.visual_view.shape}`")
    st.caption("Denormalized face crops shown in timestamp order")
    frames = runtime.display_face_frames(prepared.visual_view)
    for index, frame in enumerate(frames[:16], start=1):
        st.image(frame, caption=f"Face crop {index}")


render_page_header(
    "Stage 2",
    "2. Preprocessing",
    "See how one clip becomes a tracked and normalized face sequence.",
)
render_status(PageState.READY)
try:
    clip = require_upload()
except KeyError as error:
    if error.args != ("url_pathname",):
        raise
    clip = None

_render_stages()

if clip is not None:
    prepared = prepared_for_upload(st.session_state, clip.sha256)
    if st.button(
        "Run preprocessing",
        key="run_preprocessing",
        type="primary",
        use_container_width=True,
    ):
        clear_prepared_for_upload(st.session_state, clip.sha256)
        prepared = None
        try:
            prepared = runtime.prepare_uploaded_visual(clip)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(_failure_guidance(error))
        else:
            store_prepared(st.session_state, clip.sha256, prepared)
    if prepared is not None:
        _render_output(prepared)
