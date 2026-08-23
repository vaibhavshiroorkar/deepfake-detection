import numpy as np

from deepfake_detection.evaluation.corruptions import (
    add_audio_noise,
    compress_video_frames,
    prepend_silence,
)


def test_audio_noise_is_seeded_and_keeps_the_waveform_shape() -> None:
    waveform = np.ones(16_000, dtype=np.float32)

    first = add_audio_noise(waveform, snr_db=10.0, seed=17)
    second = add_audio_noise(waveform, snr_db=10.0, seed=17)

    assert first.shape == waveform.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, waveform)


def test_silence_shift_never_changes_the_model_input_length() -> None:
    waveform = np.arange(10, dtype=np.float32)

    shifted = prepend_silence(waveform, silence_samples=3)

    assert shifted.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_jpeg_compression_preserves_video_tensor_shape() -> None:
    frames = np.full((2, 16, 16, 3), 127, dtype=np.uint8)

    compressed = compress_video_frames(frames, jpeg_quality=30)

    assert compressed.shape == frames.shape
    assert compressed.dtype == np.uint8
