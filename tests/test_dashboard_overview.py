import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_overview_names_the_current_system_and_limits() -> None:
    page = AppTest.from_file("src/deepfake_detection/dashboard/pages/overview.py").run()

    assert not page.exception
    body = " ".join(item.value for item in page.markdown)
    assert "FakeAVCeleb" in body
    assert "FaceForensics++" in body
    assert "MNW" in body
    assert "visual development baseline" in body.lower()
