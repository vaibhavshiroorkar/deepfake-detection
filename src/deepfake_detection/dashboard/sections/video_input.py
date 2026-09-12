import streamlit as st

from deepfake_detection.dashboard.components import render_page_header, render_status
from deepfake_detection.dashboard.lib import media_kind, selectors
from deepfake_detection.dashboard.state import clear_upload, store_upload, uploaded_clip
from deepfake_detection.dashboard.status import PageState

_UPLOADER_KEY = "dashboard.video_uploader"


def _remove_video() -> None:
    clear_upload(st.session_state)
    st.session_state.pop(_UPLOADER_KEY, None)


def _preview(st, clip, kind: str) -> None:
    """Show the upload as what it is. st.video on a JPEG renders a dead player.

    Wrapped because st.image decodes the bytes there and then, so a truncated or
    mislabelled picture raises inside the gate and takes the whole page down. The
    preview is a convenience; the file is still uploaded and still scoreable, and
    the decoder downstream is the one whose complaint is worth hearing.
    """
    try:
        if kind == media_kind.VIDEO:
            st.video(clip.content)
        elif kind == media_kind.IMAGE:
            st.image(clip.content)
        elif kind == media_kind.AUDIO:
            st.audio(clip.content)
    except Exception:
        st.caption("This file could not be previewed here. It was still uploaded.")


def render_video_input(*, embedded: bool = False) -> None:
    if not embedded:
        render_page_header(
            "Stage 1",
            "1. Video input",
            "Select the local talking-head clip used by every later pipeline stage.",
        )
        render_status(PageState.READY)

    st.markdown(
        "Choose one local file: a talking-head video, a face photograph, or a "
        "voice recording. The kind is detected from the file, and the panel below "
        "says which streams can read it. The upload bytes stay in Streamlit "
        "session state. They are not written to a dataset or a training run."
    )
    upload = st.file_uploader(
        "Choose a video, image or audio file",
        type=list(media_kind.UPLOAD_SUFFIXES),
        accept_multiple_files=False,
        key=_UPLOADER_KEY,
    )

    clip = uploaded_clip(st.session_state)
    if upload is not None:
        content = upload.getvalue()
        clip = store_upload(st.session_state, name=upload.name, content=content)

    if clip is not None:
        if upload is None:
            st.info(
                "The previously selected file is retained in this session. Remove it "
                "before selecting a different one or ending the session."
            )
        kind = media_kind.classify(clip.name)
        _preview(st, clip, kind)
        st.markdown(f"**Filename:** `{clip.name}`")
        st.markdown(f"**Size:** {len(clip.content)} bytes")
        st.markdown(f"**SHA-256:** `{clip.sha256[:12]}`")

        # Said here rather than after Analyze. Someone who uploads a photograph
        # should learn now that three of the four streams cannot read it,
        # instead of reading a verdict later and wondering what it was based on.
        selectors.render_routing(st, clip.name)

        st.button(
            "Remove file",
            key="remove_video",
            use_container_width=True,
            on_click=_remove_video,
        )


if __name__ == "__main__":
    render_video_input()
