from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views import cache as view_cache
from deepfake_detection.views.cache import preprocessing_config_hash
from deepfake_detection.views.preprocessor import MediaInfo, Preprocessor
from deepfake_detection.views.timeline import ViewConfig
from deepfake_detection.views.tracking import Box, Detection, Landmarks5, Point


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


class WideFixtureDecoder(FixtureDecoder):
    def read_frames(
        self, path: Path, timestamps_sec: tuple[float, ...]
    ) -> tuple[np.ndarray, ...]:
        frames = []
        for index, _ in enumerate(timestamps_sec):
            frame = np.zeros((64, 1200, 3), dtype=np.uint8)
            frame[0, 0, 0] = index
            frames.append(frame)
        return tuple(frames)


class FixtureDetector:
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return (Detection(Box(2, 2, 14, 14), 0.99),)


class AcceleratingFaceDetector:
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        index = int(frame[0, 0, 0])
        left = 0 if index == 0 else 10 + 20 * (index - 1)
        return (Detection(Box(left, 2, left + 20, 42), 0.99),)


class LandmarkFixtureDetector:
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return (
            Detection(
                Box(2, 2, 14, 14),
                0.99,
                Landmarks5(
                    eye_left=Point(5, 6),
                    eye_right=Point(11, 6),
                    nose=Point(8, 9),
                    mouth_left=Point(6, 12),
                    mouth_right=Point(10, 12),
                ),
            ),
        )


class GapLandmarkDetector(LandmarkFixtureDetector):
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        if int(frame[0, 0, 0]) == 5:
            return ()
        return super().detect(frame)


class BorderLandmarkDetector(LandmarkFixtureDetector):
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        detection = super().detect(frame)[0]
        return (
            Detection(
                detection.box,
                detection.confidence,
                Landmarks5(
                    eye_left=Point(5, 6),
                    eye_right=Point(frame.shape[1], 6),
                    nose=Point(8, 9),
                    mouth_left=Point(6, 12),
                    mouth_right=Point(10, 12),
                ),
            ),
        )


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


def test_view_config_rejects_unknown_mouth_crop_mode() -> None:
    with pytest.raises(ValueError, match="Mouth crop mode"):
        ViewConfig(mouth_crop_mode="full_frame")


def test_view_config_rejects_unknown_track_association_and_negative_gap() -> None:
    with pytest.raises(ValueError, match="Track association"):
        ViewConfig(track_association="centroid")
    with pytest.raises(ValueError, match="Track gap"):
        ViewConfig(track_max_gap=-1)


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


def test_default_box_crop_matches_explicit_box_mode(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    default = Preprocessor(
        decoder=FixtureDecoder(),
        detector=FixtureDetector(),
        config=ViewConfig(),
        code_version="test",
    ).prepare(fixture_record(), media)
    explicit = Preprocessor(
        decoder=FixtureDecoder(),
        detector=FixtureDetector(),
        config=ViewConfig(mouth_crop_mode="box"),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert np.array_equal(default.sync_video_view, explicit.sync_video_view)
    assert default.preprocessing_config_hash == explicit.preprocessing_config_hash


def test_landmark_crop_builds_sync_view_without_changing_visual_view(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    prepared = Preprocessor(
        decoder=FixtureDecoder(),
        detector=LandmarkFixtureDetector(),
        config=ViewConfig(mouth_crop_mode="landmark"),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert prepared.visual_view.shape == (16, 3, 224, 224)
    assert prepared.sync_video_view.shape == (50, 3, 112, 112)
    assert prepared.quality.landmark_coverage == 1.0
    assert prepared.quality.full_fusion_blockers() == ()


def test_landmark_crop_does_not_fall_back_when_landmarks_are_missing(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    prepared = Preprocessor(
        decoder=FixtureDecoder(),
        detector=FixtureDetector(),
        config=ViewConfig(mouth_crop_mode="landmark"),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert prepared.visual_view is not None
    assert prepared.sync_video_view is None
    assert prepared.quality.landmark_coverage == 0.0
    assert "missing_face_landmarks" in prepared.quality.full_fusion_blockers()


def test_landmark_crop_rejects_border_geometry_without_box_fallback(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    prepared = Preprocessor(
        decoder=FixtureDecoder(),
        detector=BorderLandmarkDetector(),
        config=ViewConfig(mouth_crop_mode="landmark"),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert prepared.sync_video_view is None
    assert prepared.quality.landmark_coverage == 0.0
    assert "missing_face_landmarks" in prepared.quality.full_fusion_blockers()


def test_landmark_crop_fills_a_track_gap_from_the_nearest_frame(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    prepared = Preprocessor(
        decoder=FixtureDecoder(),
        detector=GapLandmarkDetector(),
        config=ViewConfig(mouth_crop_mode="landmark"),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert prepared.sync_video_view.shape == (50, 3, 112, 112)
    assert prepared.quality.landmark_coverage == 1.0
    assert "missing_face_landmarks" not in prepared.quality.full_fusion_blockers()


def test_preprocessing_hash_covers_crop_mode_and_template_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    box_hash = preprocessing_config_hash(config=ViewConfig(), code_version="test")
    landmark_hash = preprocessing_config_hash(
        config=ViewConfig(mouth_crop_mode="landmark"),
        code_version="test",
    )
    monkeypatch.setattr(
        view_cache,
        "LANDMARK_TEMPLATE_REVISION",
        "five-point-lower-face-v2",
    )
    revised_hash = preprocessing_config_hash(config=ViewConfig(), code_version="test")

    assert box_hash != landmark_hash
    assert box_hash != revised_hash


def test_preprocessor_uses_motion_tracker_and_hashes_its_settings(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    greedy = Preprocessor(
        decoder=WideFixtureDecoder(),
        detector=AcceleratingFaceDetector(),
        config=ViewConfig(),
        code_version="test",
    ).prepare(fixture_record(), media)
    motion = Preprocessor(
        decoder=WideFixtureDecoder(),
        detector=AcceleratingFaceDetector(),
        config=ViewConfig(track_association="constant_velocity", track_max_gap=1),
        code_version="test",
    ).prepare(fixture_record(), media)

    assert greedy.visual_view is None
    assert greedy.sync_video_view is None
    assert motion.visual_view is not None
    assert motion.sync_video_view is not None
    assert greedy.preprocessing_config_hash != motion.preprocessing_config_hash


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
