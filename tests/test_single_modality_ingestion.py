"""Preparing a still image and a sound file.

`prepare_visual` probes for a duration, samples sixteen timestamps and rejects
anything that does not decode to exactly that many frames, so a photograph fails
with "Video duration must be positive" before it reaches a model. These two
entry points are what let an image or an audio upload be scored at all.
"""

from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.preprocessor import MediaInfo, Preprocessor
from deepfake_detection.views.timeline import ViewConfig


class OneFaceDetector:
    """A face filling the middle of whatever frame it is given."""

    def detect(self, frame: np.ndarray):
        from deepfake_detection.views.tracking import Box
        from deepfake_detection.views.tracking import Detection as Real

        height, width = frame.shape[:2]
        box = Box(width * 0.25, height * 0.25, width * 0.75, height * 0.75)
        return (Real(box=box, confidence=0.99, landmarks=None),)


class StubDecoder:
    """Returns fixed media, so nothing here depends on ffmpeg or a real file."""

    def __init__(self, *, audio_present: bool = True, duration: float = 4.0) -> None:
        self._audio_present = audio_present
        self._duration = duration

    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(
            duration_sec=self._duration,
            video_fps=25.0,
            audio_duration_sec=self._duration if self._audio_present else 0.0,
            audio_present=self._audio_present,
        )

    def read_frames(self, path: Path, timestamps_sec):
        return tuple(
            np.full((240, 320, 3), 128, dtype=np.uint8) for _ in timestamps_sec
        )

    def read_image(self, path: Path):
        return (np.full((240, 320, 3), 128, dtype=np.uint8),)

    def read_audio(self, path, *, start_sec, duration_sec, sample_rate):
        return np.linspace(-0.5, 0.5, int(duration_sec * sample_rate), dtype=np.float32)


def record() -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": "upload",
            "dataset": "dashboard",
            "video_path": "upload.bin",
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": "upload",
        }
    )


def media(tmp_path: Path, name: str) -> Path:
    """A real file on disk: cache_fingerprint hashes the media bytes, so a
    prepared clip cannot be stamped from a path that does not exist."""
    path = tmp_path / name
    path.write_bytes(b"fixture-media-bytes")
    return path


def preprocessor(decoder: StubDecoder) -> Preprocessor:
    return Preprocessor(
        decoder=decoder,
        detector=OneFaceDetector(),
        config=ViewConfig(),
        code_version="test",
    )


def test_an_image_produces_one_visual_frame(tmp_path: Path) -> None:
    """One frame, not sixteen. Repeating it would claim the face does not
    change, which is a different statement from "this was one image"."""
    prepared = preprocessor(StubDecoder()).prepare_image(record(), media(tmp_path, "a.jpg"))

    assert prepared.visual_view is not None
    assert prepared.visual_view.shape[0] == 1
    assert prepared.visual_view.shape[1:] == (3, 224, 224)


def test_an_image_has_no_other_view(tmp_path: Path) -> None:
    """Three streams cannot read it, and the abstention machinery already knows
    what a None view means."""
    prepared = preprocessor(StubDecoder()).prepare_image(record(), media(tmp_path, "a.jpg"))

    assert prepared.audio_view is None
    assert prepared.sync_video_view is None
    assert prepared.sync_audio_view is None
    assert prepared.quality.audio_present is False


def test_audio_produces_the_window_the_branch_was_trained_on(tmp_path: Path) -> None:
    config = ViewConfig()
    prepared = preprocessor(StubDecoder()).prepare_audio(record(), media(tmp_path, "a.wav"))

    assert prepared.audio_view is not None
    assert prepared.audio_view.shape == (
        round(config.audio_seconds * config.sample_rate),
    )
    assert prepared.visual_view is None


def test_audio_is_normalised_like_the_video_path(tmp_path: Path) -> None:
    """Same _normalize_and_pad, so an uploaded wav reaches the branch as the
    tensor it was trained on rather than a differently scaled one."""
    prepared = preprocessor(StubDecoder()).prepare_audio(record(), media(tmp_path, "a.wav"))

    assert abs(float(prepared.audio_view.mean())) < 1e-5
    assert abs(float(prepared.audio_view.std()) - 1.0) < 1e-3


def test_a_file_with_no_audio_track_is_refused(tmp_path: Path) -> None:
    """Better than returning a window of silence, which would score."""
    engine = preprocessor(StubDecoder(audio_present=False))

    with pytest.raises(ValueError, match="no audio track"):
        engine.prepare_audio(record(), tmp_path / "silent.mp4")


def test_both_paths_stamp_the_same_provenance(tmp_path: Path) -> None:
    """A clip hashed differently from a video one would not match its cache
    entry."""
    engine = preprocessor(StubDecoder())
    image = engine.prepare_image(record(), media(tmp_path, "a.jpg"))
    audio = engine.prepare_audio(record(), media(tmp_path, "a.wav"))

    assert image.preprocessing_config_hash == audio.preprocessing_config_hash
    assert image.preprocessing_config_hash
