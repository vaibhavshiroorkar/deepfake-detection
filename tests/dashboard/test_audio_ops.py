import numpy as np
from dashboard.lib import audio_ops as A


def _tone(sr=16000, secs=1.0, f=440.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_downmix_averages_channels():
    stereo = np.stack([np.ones(10, np.float32), np.full(10, 3.0, np.float32)])
    assert np.allclose(A.downmix(stereo), 2.0)


def test_resample_changes_length_proportionally():
    wav = _tone(sr=16000, secs=1.0)
    out = A.resample(wav, 16000, 8000)
    assert abs(len(out) - 8000) <= 2


def test_trim_silence_drops_leading_zeros():
    wav = np.concatenate([np.zeros(16000, np.float32), _tone(secs=0.5)])
    trimmed, dropped = A.trim_silence(wav, 16000, top_db=30.0)
    assert dropped > 0.8                      # ~1s of leading silence dropped
    assert len(trimmed) < len(wav)


def test_add_noise_hits_target_snr_within_tolerance():
    wav = _tone(secs=1.0)
    noisy = A.add_noise(wav, snr_db=10.0, rng=np.random.default_rng(0))
    noise = noisy - wav
    snr = 10 * np.log10(np.mean(wav ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10.0) < 1.5


def test_extract_windows_shape():
    wav = _tone(secs=2.0)
    ts = np.linspace(0.2, 1.8, 16)
    out = A.extract_windows(wav, 16000, ts, 0.35)
    assert out.shape == (16, int(0.35 * 16000))


def test_mel_spectrogram_rows_equal_n_mels():
    mel = A.mel_spectrogram(_tone(secs=1.0), 16000, n_mels=64, hop=256)
    assert mel.shape[0] == 64


def test_bandpass_runs_and_preserves_length():
    wav = _tone(secs=1.0)
    assert A.bandpass(wav, 16000, 300.0, 3000.0).shape == wav.shape
