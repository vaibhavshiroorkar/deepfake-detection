from pathlib import Path

from deepfake_detection.dashboard.state import (
    prediction_for_upload,
    prepared_for_upload,
    store_prediction,
    store_prepared,
    store_upload,
    temporary_video,
    uploaded_clip,
)
from deepfake_detection.inference.predictor import PredictionResult
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def test_store_upload_hashes_bytes_and_normalizes_the_suffix() -> None:
    values: dict[str, object] = {}

    clip = store_upload(values, name="sample.MP4", content=b"video")

    assert clip.name == "sample.MP4"
    assert clip.suffix == ".mp4"
    assert len(clip.sha256) == 64
    assert uploaded_clip(values) == clip


def test_temporary_video_removes_the_file_after_use() -> None:
    clip = store_upload({}, name="sample.mp4", content=b"video")

    with temporary_video(clip) as path:
        assert path.read_bytes() == b"video"
        retained = Path(path)

    assert not retained.exists()


def test_a_new_upload_does_not_reuse_the_previous_prediction() -> None:
    values: dict[str, object] = {}
    first = store_upload(values, name="one.mp4", content=b"one")
    result = PredictionResult(
        clip_id="one",
        verdict="real",
        probability=0.1,
        branch_logits={"visual": -2.2},
        blockers=(),
        preprocessing_fingerprint="fixture",
    )
    store_prediction(values, first.sha256, result)

    second = store_upload(values, name="two.mp4", content=b"two")

    assert prediction_for_upload(values, second.sha256) is None


def test_prepared_state_is_available_only_for_its_upload() -> None:
    values: dict[str, object] = {}
    clip = store_upload(values, name="sample.mp4", content=b"video")
    prepared = PreparedClip(
        clip_id="sample",
        visual_view=None,
        audio_view=None,
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(
            face_coverage=0.0,
            stable_face_track=False,
            audio_present=False,
            audio_clipped=False,
            av_duration_delta_sec=0.0,
        ),
        preprocessing_fingerprint="fixture",
    )
    store_prepared(values, clip.sha256, prepared)

    assert prepared_for_upload(values, clip.sha256) == prepared
    assert prepared_for_upload(values, "different-upload") is None
