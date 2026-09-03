import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard.state import UploadedClip
from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def _page_body(page: AppTest) -> str:
    elements = (*page.markdown, *page.info, *page.code)
    return " ".join(item.value for item in elements)


def test_visual_model_page_shows_the_frozen_shape_ladder() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    page.switch_page("pages/visual_model.py").run()

    body = _page_body(page)
    assert not page.exception
    for shape in (
        "[B, 16, 3, 224, 224]",
        "[B*16, 1280]",
        "[B, 16, 1280]",
        "[B, 256]",
        "[B]",
    ):
        assert shape in body


def test_visual_model_page_explains_outputs_without_claiming_explanations() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/visual_model.py"
    ).run()

    body = _page_body(page)
    assert "EfficientNet-B0" in body
    assert "GRU" in body
    assert "logit" in body.lower()
    assert "sigmoid" in body.lower()
    assert "not an explanation" in body.lower()


def test_visual_model_page_uses_the_shared_video_input_prerequisite() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/app.py", default_timeout=30
    ).run()
    page.switch_page("pages/visual_model.py").run()

    assert not page.exception
    assert any("Video input" in item.value for item in page.info)
    assert [link.label for link in page.get("page_link")] == ["Go to Video input"]


def test_visual_model_page_shows_real_stored_shapes_and_scores() -> None:
    clip_hash = "a" * 64
    prepared = PreparedClip(
        clip_id=clip_hash,
        visual_view=np.zeros((16, 3, 12, 12), dtype=np.float32),
        audio_view=None,
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(
            face_coverage=1.0,
            stable_face_track=True,
            audio_present=False,
            audio_clipped=False,
            av_duration_delta_sec=0.0,
        ),
        preprocessing_fingerprint="fixture",
    )
    result = PredictionResult(
        clip_id=clip_hash,
        verdict="fake",
        probability=0.777,
        branch_logits={"visual": 1.25},
        blockers=(),
        preprocessing_fingerprint="fixture",
    )
    page = AppTest.from_file("src/deepfake_detection/dashboard/pages/visual_model.py")
    page.session_state["dashboard.upload"] = UploadedClip(
        "sample.mp4", ".mp4", b"video", clip_hash
    )
    page.session_state["dashboard.prepared"] = (clip_hash, prepared)
    page.session_state["dashboard.prediction"] = (clip_hash, result)

    page.run()

    body = _page_body(page)
    assert not page.exception
    assert "(16, 3, 12, 12)" in body
    assert "+1.250" in body
    assert "77.7%" in body
