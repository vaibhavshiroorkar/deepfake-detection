from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from deepfake_detection.dashboard.status import PageState
from deepfake_detection.dashboard.workflow import StepState

if TYPE_CHECKING:
    from deepfake_detection.dashboard.state import UploadedClip
def render_page_header(step: str, title: str, summary: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.write(summary)


def require_upload() -> UploadedClip | None:
    from deepfake_detection.dashboard.state import uploaded_clip

    clip = uploaded_clip(st.session_state)
    if clip is None:
        st.info("Start with 1. Video input before using this section.")
    return clip


def render_status(state: PageState) -> None:
    st.markdown(f"Status: {state.value}")


def render_step_status(state: StepState) -> None:
    st.markdown(
        f'<div class="step-state {state.value}">{state.value}</div>',
        unsafe_allow_html=True,
    )
