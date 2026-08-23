from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.preprocessor import MediaInfo, Preprocessor
from deepfake_detection.views.timeline import ViewConfig
from deepfake_detection.views.tracking import Box, Detection


class FixtureDecoder:
    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(
            duration_sec=5.0,
            video_fps=25.0,
            audio_duration_sec=5.0,
            audio_present=True,
        )

    def read_frames(
        self, path: Path, timestamps_sec: tuple[float, ...]
    ) -> tuple[np.ndarray, ...]:
        return tuple(
            np.full((16, 16, 3), fill_value=index, dtype=np.uint8)
            for index, _ in enumerate(timestamps_sec)
        )

    def read_audio(
        self,
        path: Path,
        *,
        start_sec: float,
        duration_sec: float,
        sample_rate: int,
    ) -> np.ndarray:
        return np.linspace(
            -0.5,
            0.5,
            round(duration_sec * sample_rate),
            dtype=np.float32,
        )


class FixtureDetector:
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return (Detection(Box(2, 2, 14, 14), 0.99),)


class NoFaceDetector:
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return ()


class ShortDecoder(FixtureDecoder):
    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(
            duration_sec=1.0,
            video_fps=25.0,
            audio_duration_sec=1.0,
            audio_present=True,
        )

    def read_frames(
        self, path: Path, timestamps_sec: tuple[float, ...]
    ) -> tuple[np.ndarray, ...]:
        assert all(0 <= timestamp < 1.0 for timestamp in timestamps_sec)
        return super().read_frames(path, timestamps_sec)


class RecordingDecoder(FixtureDecoder):
    def __init__(self) -> None:
        self.audio_requests: list[tuple[float, float]] = []

    def read_audio(
        self,
        path: Path,
        *,
        start_sec: float,
        duration_sec: float,
        sample_rate: int,
    ) -> np.ndarray:
        self.audio_requests.append((start_sec, duration_sec))
        return super().read_audio(
            path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        )


def fixture_record() -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": "clip-1",
            "dataset": "fixture",
            "video_path": "clip.mp4",
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": "id1",
            "leading_silence_sec": "0.5",
        }
    )


def test_preprocessor_builds_all_three_exact_views(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    preprocessor = Preprocessor(
        decoder=FixtureDecoder(),
        detector=FixtureDetector(),
        config=ViewConfig(),
        code_version="test",
    )

    prepared = preprocessor.prepare(fixture_record(), media)

    assert prepared.visual_view.shape == (16, 3, 224, 224)
    assert prepared.audio_view.shape == (64_000,)
    assert float(prepared.audio_view.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(prepared.audio_view.std()) == pytest.approx(1.0, abs=1e-6)
    assert prepared.sync_video_view.shape == (50, 3, 112, 112)
    assert prepared.sync_audio_view.shape == (32_000,)
    assert float(prepared.sync_audio_view.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(prepared.sync_audio_view.std()) == pytest.approx(1.0, abs=1e-6)
    assert prepared.sync_audio_context.shape == (42_240,)
    assert prepared.quality.full_fusion_blockers() == ()


def test_preprocessor_never_replaces_a_missing_face_with_the_full_frame(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    preprocessor = Preprocessor(
        decoder=FixtureDecoder(),
        detector=NoFaceDetector(),
        config=ViewConfig(),
        code_version="test",
    )

    prepared = preprocessor.prepare(fixture_record(), media)

    assert prepared.visual_view is None
    assert prepared.sync_video_view is None
    assert "unstable_face_track" in prepared.quality.full_fusion_blockers()


def test_short_clip_never_requests_frames_after_the_media_end(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    preprocessor = Preprocessor(
        decoder=ShortDecoder(),
        detector=FixtureDetector(),
        config=ViewConfig(),
        code_version="test",
    )

    prepared = preprocessor.prepare(fixture_record(), media)

    assert "insufficient_sync_duration" in prepared.quality.full_fusion_blockers()


def test_sync_offset_context_never_reintroduces_leading_silence(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    decoder = RecordingDecoder()
    preprocessor = Preprocessor(
        decoder=decoder,
        detector=FixtureDetector(),
        config=ViewConfig(),
        code_version="test",
    )

    preprocessor.prepare(fixture_record(), media)

    context_start, context_duration = decoder.audio_requests[-1]
    assert context_start == 0.5
    assert context_duration == 2.64


def test_leading_silence_ablation_keeps_the_original_timeline(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    decoder = RecordingDecoder()
    preprocessor = Preprocessor(
        decoder=decoder,
        detector=FixtureDetector(),
        config=ViewConfig(remove_leading_silence=False),
        code_version="test",
    )

    preprocessor.prepare(fixture_record(), media)

    context_start, _ = decoder.audio_requests[-1]
    assert context_start == 0.0
