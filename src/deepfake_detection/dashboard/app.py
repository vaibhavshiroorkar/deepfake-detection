from __future__ import annotations

import streamlit as st

from deepfake_detection.dashboard.navigation import PAGES, PageState
from deepfake_detection.dashboard.state import (
    prediction_for_upload,
    prepared_for_upload,
    uploaded_clip,
)

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
        --quiet: #E8EFF3;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
    }
    .stMainBlockContainer {
        max-width: 840px;
        padding-top: 1.25rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        font-family: Bahnschrift, "Arial Narrow", sans-serif;
    }
    p, label, button, a {
        font-family: Aptos, Calibri, sans-serif;
    }
    code, pre, [data-testid="stMetricValue"] {
        font-family: "Cascadia Mono", Consolas, monospace;
    }
    .project-masthead {
        align-items: baseline;
        border-bottom: 1px solid var(--line);
        display: flex;
        gap: .75rem;
        margin-bottom: 1.25rem;
        padding-bottom: .6rem;
    }
    .project-masthead strong {
        color: var(--cobalt);
        font: 700 1rem/1 Bahnschrift, "Arial Narrow", sans-serif;
    }
    .project-masthead span {
        color: var(--ink);
        font: .9rem/1.2 Aptos, Calibri, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: var(--quiet);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border-left: 3px solid transparent;
        border-radius: 0;
        color: var(--ink);
        padding: .4rem .6rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
        background: #DDE7F3;
        border-left-color: var(--cobalt);
        color: var(--cobalt);
        font-weight: 700;
    }
    .pipeline-state {
        color: #51626D;
        font: .72rem/1.25 "Cascadia Mono", Consolas, monospace;
        margin: -.3rem 0 .35rem .75rem;
    }
    .pipeline-state.complete { color: var(--teal); }
    .pipeline-state.current { color: var(--cobalt); font-weight: 700; }
    .pipeline-state.prototype { color: var(--amber); }
    .pipeline-state.locked { color: var(--evidence); }
    a:focus-visible, button:focus-visible, input:focus-visible,
    [tabindex]:focus-visible {
        outline: 3px solid var(--cobalt) !important;
        outline-offset: 2px;
    }
    @media (max-width: 700px) {
        .stMainBlockContainer {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: .75rem;
        }
        .project-masthead {
            align-items: flex-start;
            flex-direction: column;
            gap: .2rem;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page_targets = {
    page.slug: st.Page(
        page.module,
        title=page.title,
        url_path=page.slug,
        default=index == 0,
    )
    for index, page in enumerate(PAGES)
}
selected_page = st.navigation(tuple(page_targets.values()), position="hidden")
selected_spec = next(page for page in PAGES if page.title == selected_page.title)


def _render_sidebar(selected_slug: str) -> None:
    clip = uploaded_clip(st.session_state)
    prepared = prepared_for_upload(st.session_state, clip.sha256) if clip else None
    prediction = prediction_for_upload(st.session_state, clip.sha256) if clip else None
    selected_title = next(page.title for page in PAGES if page.slug == selected_slug)

    with st.sidebar:
        st.header("Pipeline index")
        st.caption(f"Current: {selected_title}")
        for page in PAGES:
            st.page_link(
                page_targets[page.slug],
                label=page.navigation_label,
                use_container_width=True,
            )
            if page.slug == selected_slug:
                stage_state = "current"
            elif page.state is not PageState.READY:
                stage_state = page.state.value
            elif page.slug == "video-input" and clip is not None:
                stage_state = "complete"
            elif page.slug == "preprocessing" and prepared is not None:
                stage_state = "complete"
            elif page.slug in {"visual-model", "prediction"} and prediction is not None:
                stage_state = "complete"
            else:
                stage_state = "ready"
            st.markdown(
                f'<div class="pipeline-state {stage_state}">{stage_state}</div>',
                unsafe_allow_html=True,
            )


_render_sidebar(selected_spec.slug)

st.markdown(
    '<div class="project-masthead"><strong>Evidence Gate</strong>'
    "<span>Deepfake detector teaching dashboard</span></div>",
    unsafe_allow_html=True,
)
selected_page.run()
