from __future__ import annotations

import streamlit as st

from deepfake_detection.dashboard.navigation import PageState
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
