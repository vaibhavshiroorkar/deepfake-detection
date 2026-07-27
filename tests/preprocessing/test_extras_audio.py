"""EXTRAS audio ops, noise, RMS, bandpass, spectral denoise, mel view."""
import numpy as np

from preprocessing.ops import extras_audio as AX


def _tone(sr=16000, secs=1.0, f=440.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_add_noise_hits_target_snr_within_tolerance():
    wav = _tone(secs=1.0)
    noisy = AX.add_noise(wav, snr_db=10.0, rng=np.random.default_rng(0))
    noise = noisy - wav
    snr = 10 * np.log10(np.mean(wav ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10.0) < 1.5


def test_rms_normalize_hits_target():
    wav = _tone(secs=1.0) * 0.01
    out = AX.rms_normalize(wav, target_db=-20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out ** 2)))
    assert abs(rms_db - (-20.0)) < 1.0


def test_bandpass_runs_and_preserves_length():
    wav = _tone(secs=1.0)
    assert AX.bandpass(wav, 16000, 300.0, 3000.0).shape == wav.shape


def test_spectral_denoise_preserves_length():
    wav = _tone(secs=1.0)
    assert AX.spectral_denoise(wav, 16000, 1.0).shape == wav.shape


def test_mel_spectrogram_rows_equal_n_mels():
    mel = AX.mel_spectrogram(_tone(secs=1.0), 16000, n_mels=64, hop=256)
    assert mel.shape[0] == 64
