from __future__ import annotations

import streamlit as st

from deepfake_detection.dashboard.status import PageState


def render_page_header(step: str, title: str, summary: str) -> None:
    st.caption(step.upper())
    st.title(title)
    st.write(summary)


def render_status(state: PageState) -> None:
    st.markdown(f"Status: {state.value}")
