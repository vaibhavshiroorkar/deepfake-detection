import numpy as np
import pytest

from deepfake_detection.preprocessing.ops import (
    audio,
    constants,
    detectors,
    extras_audio,
    extras_visual,
    faces,
)


class StubDetector:
    """A detector that reports one fixed face, or none."""

    def __init__(self, box=None, landmarks=None, probability=0.0) -> None:
        self.box = box
        self.landmarks = landmarks
        self.probability = probability

    def detect(self, frame_rgb):
        if self.box is None:
            return None, None, None
        return self.box, self.landmarks, self.probability


def make_frame(height=240, width=320) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


def make_landmarks() -> np.ndarray:
    return np.array(
        [[100, 90], [140, 90], [120, 110], [105, 130], [135, 130]], dtype=np.float32
    )


def test_detect_applies_the_confidence_threshold_not_the_detector() -> None:
    detector = StubDetector(np.array([80, 60, 160, 150.0]), make_landmarks(), 0.5)
    assert faces.detect(make_frame(), detector, conf_thresh=0.4)[0] is not None
    assert faces.detect(make_frame(), detector, conf_thresh=0.6) == (None, None, None)


def test_detect_crop_falls_back_to_a_plain_resize_without_a_face() -> None:
    face, mouth, detected = faces.detect_crop(make_frame(), StubDetector(), 0.9)
    assert detected is False
    assert face.shape == (constants.FRAME_SIZE, constants.FRAME_SIZE, 3)
    assert mouth.shape == (constants.MOUTH_SIZE, constants.MOUTH_SIZE, 3)


def test_detect_crop_returns_fixed_shapes_with_a_face() -> None:
    detector = StubDetector(np.array([80, 60, 160, 150.0]), make_landmarks(), 0.99)
    face, mouth, detected = faces.detect_crop(make_frame(), detector, 0.9)
    assert detected is True
    assert face.shape == (constants.FRAME_SIZE, constants.FRAME_SIZE, 3)
    assert mouth.shape == (constants.MOUTH_SIZE, constants.MOUTH_SIZE, 3)


def test_crop_and_resize_rejects_an_empty_region() -> None:
    frame = make_frame()
    assert faces.crop_and_resize(frame, np.array([10, 10, 10, 10.0]), margin=0.0) is None


def test_imagenet_normalize_centres_the_crop() -> None:
    normalized = faces.imagenet_normalize(np.full((8, 8, 3), 128, dtype=np.uint8))
    assert normalized.dtype == np.float32
    low, high = faces.normalized_range(normalized)
    assert low < high


def test_sample_timestamps_keeps_every_window_inside_the_clip() -> None:
    stamps = audio.sample_timestamps(4.0, 16, 0.35)
    assert len(stamps) == 16
    assert stamps[0] >= 0.35 / 2
    assert stamps[-1] <= 4.0 - 0.35 / 2


def test_sample_timestamps_starts_past_the_leading_silence() -> None:
    stamps = audio.sample_timestamps(4.0, 8, 0.35, start_offset=1.0)
    assert stamps[0] >= 1.0


def test_extract_windows_pads_a_short_tail() -> None:
    waveform = np.ones(1000, dtype=np.float32)
    windows = audio.extract_windows(waveform, 16000, [0.0, 0.05], 0.35)
    assert windows.shape == (2, int(0.35 * 16000))
    assert windows.dtype == np.float32


def test_extract_windows_handles_no_timestamps() -> None:
    assert audio.extract_windows(np.zeros(10, np.float32), 16000, [], 0.35).shape == (
        0,
        5600,
    )


def test_downmix_averages_channels() -> None:
    stereo = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    assert np.allclose(audio.downmix(stereo), [2.0, 2.0])


def test_resample_is_a_no_op_at_the_same_rate() -> None:
    waveform = np.ones(64, dtype=np.float32)
    assert audio.resample(waveform, 16000, 16000) is not None
    assert audio.resample(waveform, 16000, 16000).shape == (64,)


def test_leading_silence_is_measured_in_seconds() -> None:
    silence = np.zeros(8000, dtype=np.float32)
    tone = np.sin(np.linspace(0, 200, 8000)).astype(np.float32)
    measured = audio.leading_silence_sec(np.concatenate([silence, tone]), 16000)
    assert 0.2 < measured < 0.8


def test_visual_extras_preserve_shape_and_dtype() -> None:
    frame = make_frame(64, 64)
    for result in (
        extras_visual.sharpen(frame, 0.5),
        extras_visual.clahe(frame, 2.0),
        extras_visual.gaussian_blur(frame, 4),
        extras_visual.jpeg_recompress(frame, 40),
        extras_visual.downscale_upscale(frame, 0.25),
    ):
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


def test_audio_extras_preserve_length() -> None:
    waveform = np.sin(np.linspace(0, 400, 16000)).astype(np.float32)
    for result in (
        extras_audio.spectral_denoise(waveform, 16000, 1.0),
        extras_audio.rms_normalize(waveform, -20.0),
        extras_audio.bandpass(waveform, 16000, 80.0, 7000.0),
        extras_audio.add_noise(waveform, 20.0, np.random.default_rng(0)),
    ):
        assert result.shape == waveform.shape
        assert result.dtype == np.float32


def test_audio_extras_tolerate_an_empty_waveform() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert extras_audio.spectral_denoise(empty, 16000, 1.0).size == 0
    assert extras_audio.bandpass(empty, 16000, 80.0, 7000.0).size == 0
    assert extras_audio.mel_spectrogram(empty, 16000, 64, 160).shape == (64, 0)


def test_mel_spectrogram_has_one_row_per_band() -> None:
    waveform = np.sin(np.linspace(0, 400, 16000)).astype(np.float32)
    assert extras_audio.mel_spectrogram(waveform, 16000, 64, 160).shape[0] == 64


def test_build_rejects_an_unknown_detector() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        detectors.build("nosuchdetector")


def test_yunet_reports_where_its_weights_should_be() -> None:
    with pytest.raises(FileNotFoundError, match="fetch-yunet"):
        detectors.YuNetDetector(weights=detectors.YUNET_WEIGHTS.with_name("absent.onnx"))
