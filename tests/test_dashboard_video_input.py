import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_video_input_stores_streamlit_upload_bytes_in_session_state() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    page.switch_page("pages/video_input.py").run()

    page.file_uploader[0].set_value(("clip.MP4", b"video bytes", "video/mp4")).run()

    assert not page.exception
    clip = page.session_state["dashboard.upload"]
    assert clip.name == "clip.MP4"
    assert clip.content == b"video bytes"
    assert clip.suffix == ".mp4"
    assert len(clip.sha256) == 64
    assert len(page.get("video")) == 1
    body = " ".join(item.value for item in page.markdown)
    assert "clip.MP4" in body
    assert "11 bytes" in body
    assert "96b050b919f3" in body
    assert [link.label for link in page.get("page_link")] == [
        "Continue to Preprocessing"
    ]


def test_video_input_explains_local_handling_and_shows_one_uploader() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/video_input.py"
    ).run()

    assert not page.exception
    assert len(page.file_uploader) == 1
    body = " ".join(item.value for item in page.markdown)
    assert "session state" in body.lower()
    assert "not written" in body.lower()
    assert "data" not in [button.label.lower() for button in page.button]


def test_video_input_remove_action_clears_the_uploader_and_derived_state() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    page.switch_page("pages/video_input.py").run()
    page.file_uploader[0].set_value(("sample.mp4", b"video", "video/mp4")).run()
    page.session_state["dashboard.prepared"] = object()
    page.session_state["dashboard.prediction"] = object()

    assert page.button(key="remove_video").label == "Remove video"

    page.button(key="remove_video").click().run()
    page.run()

    assert page.file_uploader[0].value is None
    assert "dashboard.upload" not in page.session_state.filtered_state
    assert "dashboard.prepared" not in page.session_state.filtered_state
    assert "dashboard.prediction" not in page.session_state.filtered_state
