import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.state import UploadedClip
from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def _page_body(page: AppTest) -> str:
    elements = (
        *page.markdown,
        *page.info,
        *page.caption,
        *page.error,
        *page.subheader,
    )
    return " ".join(item.value for item in elements)


def _upload() -> UploadedClip:
    return UploadedClip("sample.mp4", ".mp4", b"video", "a" * 64)


def _prepared(clip_hash: str) -> PreparedClip:
    return PreparedClip(
        clip_id=clip_hash,
        visual_view=np.zeros((16, 3, 2, 2), dtype=np.float32),
        audio_view=None,
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(
            face_coverage=0.875,
            stable_face_track=True,
            audio_present=False,
            audio_clipped=False,
            av_duration_delta_sec=0.0,
        ),
        preprocessing_fingerprint="prepared-fingerprint",
    )


def _result(clip_hash: str) -> PredictionResult:
    return PredictionResult(
        clip_id=clip_hash,
        verdict="fake",
        probability=0.75,
        branch_logits={"visual": 1.099},
        blockers=("low face coverage",),
        preprocessing_fingerprint="prediction-fingerprint",
    )


def test_prediction_page_requires_an_upload_and_exposes_no_runtime_controls() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/prediction.py"
    ).run()

    body = _page_body(page)
    assert not page.exception
    assert "Video input" in body
    assert len(page.button) == 0
    assert not page.text_input
    assert not page.slider
    assert not page.selectbox


def test_prediction_page_shows_one_fixed_action_after_upload() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/prediction.py"
    )
    page.session_state["dashboard.upload"] = _upload()

    page.run()

    body = _page_body(page)
    assert not page.exception
    assert len(page.button) == 1
    assert page.button[0].label == "Analyze video"
    assert "Fixed decision threshold: 0.50" in body
    assert "Visual-only development baseline" in body
    assert not page.text_input
    assert not page.slider
    assert not page.selectbox


def test_prediction_page_persists_a_clicked_analysis_without_rerunning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _upload()
    result = _result(clip.sha256)
    calls: list[UploadedClip] = []
    monkeypatch.setattr(
        runtime,
        "predict_upload",
        lambda received: calls.append(received) or result,
    )
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/prediction.py"
    )
    page.session_state["dashboard.upload"] = clip
    page.session_state["dashboard.prepared"] = (clip.sha256, _prepared(clip.sha256))

    page.run()
    assert calls == []

    page.button(key="analyze_video").click().run()

    body = _page_body(page)
    assert not page.exception
    assert calls == [clip]
    assert page.session_state["dashboard.prediction"] == (clip.sha256, result)
    assert "Likely manipulated" in body
    assert "75.0%" in body
    assert "87.5%" in body
    assert "low face coverage" in body
    assert "cross-dataset generalization" in body
    assert "prediction-fingerprint" in body
    assert "4243b35e64c743b89cc33000cc9d3d3e" in body
    assert "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0" in body

    page.run()
    assert calls == [clip]


def test_prediction_page_reports_a_failed_analysis_without_storing_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(clip: UploadedClip) -> PredictionResult:
        raise RuntimeError("CUDA driver is unavailable")

    monkeypatch.setattr(runtime, "predict_upload", fail)
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/prediction.py"
    )
    page.session_state["dashboard.upload"] = _upload()

    page.run()
    page.button(key="analyze_video").click().run()

    assert not page.exception
    assert any("CUDA" in item.value for item in page.error)
    assert "dashboard.prediction" not in page.session_state.filtered_state
