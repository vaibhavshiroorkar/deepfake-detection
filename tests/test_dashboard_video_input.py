import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_video_input_stores_streamlit_upload_bytes_in_session_state() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py",
        default_timeout=30,
    ).run()

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
    assert not page.get("page_link")


def test_video_input_explains_local_handling_and_shows_one_uploader() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py"
    ).run()

    assert not page.exception
    assert len(page.file_uploader) == 1
    body = " ".join(item.value for item in page.markdown)
    assert "session state" in body.lower()
    assert "not written" in body.lower()
    assert "data" not in [button.label.lower() for button in page.button]


def test_video_input_remove_action_clears_the_uploader_and_derived_state() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py",
        default_timeout=30,
    ).run()
    page.file_uploader[0].set_value(("sample.mp4", b"video", "video/mp4")).run()
    page.session_state["dashboard.prepared"] = object()
    page.session_state["dashboard.prediction"] = object()

    # "file", not "video": the gate takes images and audio now too.
    assert page.button(key="remove_video").label == "Remove file"

    page.button(key="remove_video").click().run()
    page.run()

    assert page.file_uploader[0].value is None
    assert "dashboard.upload" not in page.session_state.filtered_state
    assert "dashboard.prepared" not in page.session_state.filtered_state
    assert "dashboard.prediction" not in page.session_state.filtered_state


def test_the_gate_accepts_an_image_and_says_what_will_run() -> None:
    """It took four video containers. A photograph was refused at the uploader,
    so the visual stream could never see one."""
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py",
        default_timeout=30,
    ).run()

    page.file_uploader[0].set_value(("face.jpg", b"image-bytes", "image/jpeg")).run()

    assert not page.exception
    captions = " ".join(item.value for item in page.caption).lower()
    assert "image" in captions
    assert "visual only" in captions


def test_the_gate_accepts_audio() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py",
        default_timeout=30,
    ).run()

    page.file_uploader[0].set_value(("voice.wav", b"audio-bytes", "audio/wav")).run()

    captions = " ".join(item.value for item in page.caption).lower()
    assert "audio only" in captions


def test_the_uploader_offers_every_kind_the_pipeline_reads() -> None:
    """The gate and the Streams picker must accept the same set, or one of them
    refuses a file the other can score."""
    from deepfake_detection.dashboard.lib import media_kind

    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/sections/video_input.py",
        default_timeout=30,
    ).run()

    # Streamlit dots the suffixes and adds its own aliases (tiff brings tif,
    # m4v brings mpeg4), so this is containment, not equality.
    offered = {item.lstrip(".") for item in page.file_uploader[0].proto.type}
    assert set(media_kind.UPLOAD_SUFFIXES) <= offered
    for kind in ("mp4", "jpg", "wav"):
        assert kind in offered
