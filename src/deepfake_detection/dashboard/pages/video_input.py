import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.navigation import PageState
from deepfake_detection.dashboard.state import store_upload

render_page_header(
    "Stage 1",
    "1. Video input",
    "Select the local talking-head clip used by every later pipeline stage.",
)
render_status(PageState.READY)

st.markdown(
    "Choose one local talking-head clip in MP4, MOV, MKV, or AVI format. The "
    "upload bytes stay in Streamlit session state. They are not written to a "
    "dataset or a training run."
)
upload = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "mkv", "avi"],
    accept_multiple_files=False,
)

if upload is not None:
    content = upload.getvalue()
    clip = store_upload(st.session_state, name=upload.name, content=content)
    st.video(content)
    st.markdown(f"**Filename:** `{clip.name}`")
    st.markdown(f"**Size:** {len(clip.content)} bytes")
    st.markdown(f"**SHA-256:** `{clip.sha256[:12]}`")
    st.page_link(
        "pages/preprocessing.py",
        label="Continue to Preprocessing",
        use_container_width=True,
    )
