import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.pages.preprocessing import PREPROCESSING_STAGES
from deepfake_detection.dashboard.state import UploadedClip
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def test_preprocessing_page_requires_video_input() -> None:
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/preprocessing.py"
    ).run()

    assert not page.exception
    assert any("Video input" in item.value for item in page.info)


def test_preprocessing_stage_names_match_the_real_pipeline() -> None:
    assert PREPROCESSING_STAGES == (
        "Media probe",
        "Timestamp sampling",
        "Face detection and tracking",
        "Face crop",
        "Resize and normalization",
        "Model tensor",
    )


def test_preprocessing_page_shows_the_prepared_visual_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedClip(
        clip_id="a" * 64,
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
        preprocessing_fingerprint="fixture",
        preprocessing_config_hash="fd372dbe" + "0" * 56,
    )
    monkeypatch.setattr(runtime, "prepare_uploaded_visual", lambda clip: prepared)
    page = AppTest.from_file("src/deepfake_detection/dashboard/pages/preprocessing.py")
    page.session_state["dashboard.upload"] = UploadedClip(
        "sample.mp4", ".mp4", b"video", "a" * 64
    )

    page.run()
    page.button(key="run_preprocessing").click().run()

    assert not page.exception
    assert page.session_state["dashboard.prepared"][1] == prepared
    body = " ".join(item.value for item in page.markdown)
    assert "(16, 3, 2, 2)" in body
    assert "87.5%" in body
    assert "Stable" in body
    assert "fd372dbe" in body
    assert len(page.image) == 16


def test_preprocessing_page_shows_quality_blockers_without_tensor_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedClip(
        clip_id="a" * 64,
        visual_view=None,
        audio_view=None,
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(
            face_coverage=0.25,
            stable_face_track=False,
            audio_present=False,
            audio_clipped=False,
            av_duration_delta_sec=0.0,
        ),
        preprocessing_fingerprint="fixture",
        preprocessing_config_hash="fd372dbe" + "0" * 56,
    )
    monkeypatch.setattr(runtime, "prepare_uploaded_visual", lambda clip: prepared)
    page = AppTest.from_file("src/deepfake_detection/dashboard/pages/preprocessing.py")
    page.session_state["dashboard.upload"] = UploadedClip(
        "sample.mp4", ".mp4", b"video", "a" * 64
    )

    page.run()
    page.button(key="run_preprocessing").click().run()

    assert not page.exception
    info = " ".join(item.value for item in page.info)
    assert "unstable face track" in info.lower()
    assert "low face coverage" in info.lower()
    assert "Model tensor shape" not in " ".join(item.value for item in page.markdown)
    assert not page.image


def test_preprocessing_runtime_failure_does_not_store_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(clip):
        raise RuntimeError("CUDA driver is unavailable")

    monkeypatch.setattr(runtime, "prepare_uploaded_visual", fail)
    page = AppTest.from_file("src/deepfake_detection/dashboard/pages/preprocessing.py")
    page.session_state["dashboard.upload"] = UploadedClip(
        "sample.mp4", ".mp4", b"video", "a" * 64
    )

    page.run()
    page.button(key="run_preprocessing").click().run()

    assert not page.exception
    assert any("CUDA" in item.value for item in page.error)
    assert "dashboard.prepared" not in page.session_state.filtered_state
