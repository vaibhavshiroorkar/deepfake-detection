"""MAIN audio ops: timestamps (with silence offset), decode helpers, windows."""
import numpy as np

from preprocessing.ops import audio as A


def _tone(sr=16000, secs=1.0, f=440.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_sample_timestamps_count_and_inset():
    ts = A.sample_timestamps(duration_sec=8.0, num_frames=16, window_sec=0.35)
    assert ts.shape == (16,)
    assert ts[0] >= 0.35 / 2 - 1e-9
    assert ts[-1] <= 8.0 - 0.35 / 2 + 1e-9
    assert np.all(np.diff(ts) > 0)


def test_sample_timestamps_offset_pushes_start_past_silence():
    ts = A.sample_timestamps(8.0, 16, 0.35, start_offset=2.0)
    assert ts[0] >= 2.0 + 0.35 / 2 - 1e-9            # first sample is past the silence
    assert np.all(np.diff(ts) > 0)


def test_sample_timestamps_degenerate_short_clip():
    ts = A.sample_timestamps(0.1, 4, 0.35)
    assert ts.shape == (4,) and np.all(ts >= 0)


def test_downmix_averages_channels():
    stereo = np.stack([np.ones(10, np.float32), np.full(10, 3.0, np.float32)])
    assert np.allclose(A.downmix(stereo), 2.0)


def test_resample_changes_length_proportionally():
    out = A.resample(_tone(16000, 1.0), 16000, 8000)
    assert abs(len(out) - 8000) <= 2


def test_leading_silence_detects_prefix():
    wav = np.concatenate([np.zeros(16000, np.float32), _tone(secs=0.5)])
    assert A.leading_silence_sec(wav, 16000, top_db=30.0) > 0.8


def test_trim_leading_silence_drops_prefix():
    wav = np.concatenate([np.zeros(16000, np.float32), _tone(secs=0.5)])
    trimmed, dropped = A.trim_leading_silence(wav, 16000, top_db=30.0)
    assert dropped > 0.8 and len(trimmed) < len(wav)


def test_extract_windows_shape():
    ts = np.linspace(0.2, 1.8, 16)
    out = A.extract_windows(_tone(secs=2.0), 16000, ts, 0.35)
    assert out.shape == (16, int(0.35 * 16000))
