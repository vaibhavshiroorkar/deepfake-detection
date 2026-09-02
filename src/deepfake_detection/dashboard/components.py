from __future__ import annotations

import streamlit as st

from deepfake_detection.dashboard.navigation import PageSpec, PageState
from deepfake_detection.dashboard.state import UploadedClip, uploaded_clip


def render_page_header(step: str, title: str, summary: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.write(summary)


def require_upload() -> UploadedClip | None:
    clip = uploaded_clip(st.session_state)
    if clip is None:
        st.info("Start with 1. Video input, then return to this page.")
        st.page_link("pages/video_input.py", label="Go to Video input")
    return clip


def render_status(state: PageState) -> None:
    st.caption(f"Status: {state.value}")


def pipeline_stage_status(
    page: PageSpec,
    *,
    selected_slug: str,
    has_upload: bool,
    has_prepared: bool,
    has_prediction: bool,
) -> str:
    if page.slug == selected_slug:
        return "current"
    if page.state is not PageState.READY:
        return page.state.value
    if page.slug == "video-input" and has_upload:
        return "complete"
    if page.slug == "preprocessing" and has_prepared:
        return "complete"
    if page.slug in {"visual-model", "prediction"} and has_prediction:
        return "complete"
    return "ready"
