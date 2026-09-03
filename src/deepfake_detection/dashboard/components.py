from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import streamlit as st

from deepfake_detection.dashboard.status import PageState
from deepfake_detection.dashboard.workflow import StepState

if TYPE_CHECKING:
    from deepfake_detection.dashboard.state import UploadedClip


class PipelinePage(Protocol):
    slug: str
    state: PageState


def render_page_header(step: str, title: str, summary: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.write(summary)


def require_upload(*, show_page_link: bool = True) -> UploadedClip | None:
    from deepfake_detection.dashboard.state import uploaded_clip

    clip = uploaded_clip(st.session_state)
    if clip is None:
        st.info("Start with 1. Video input before using this section.")
        if show_page_link:
            st.page_link("pages/video_input.py", label="Go to Video input")
    return clip


def render_status(state: PageState) -> None:
    st.caption(f"Status: {state.value}")


def render_step_status(state: StepState) -> None:
    st.markdown(
        f'<div class="step-state {state.value}">{state.value}</div>',
        unsafe_allow_html=True,
    )


def pipeline_stage_status(
    page: PipelinePage,
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
