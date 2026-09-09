"""Audio enhancement and degradation ops. Mono float32 [n] in and out.

Not part of the stored contract. These are robustness and augmentation probes
toggled independently in the dashboard, so the baseline is all of them off.
`mel_spectrogram` is a visualization view, not a pipeline output.
"""

from __future__ import annotations

import numpy as np


def spectral_denoise(waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Spectral gating: attenuate bins below strength * noise floor."""
    if waveform.size == 0:
        return waveform
    import librosa

    stft = librosa.stft(waveform)
    magnitude, phase = np.abs(stft), np.angle(stft)
    floor = np.median(magnitude, axis=1, keepdims=True) * strength
    magnitude = np.maximum(magnitude - floor, 0.0)
    gated = librosa.istft(magnitude * np.exp(1j * phase), length=len(waveform))
    return gated.astype(np.float32)


def rms_normalize(waveform: np.ndarray, target_db: float) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    rms = np.sqrt(np.mean(waveform**2)) + 1e-9
    target_rms = 10 ** (target_db / 20.0)
    return np.clip(waveform * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def bandpass(
    waveform: np.ndarray, sr: int, low_hz: float, high_hz: float
) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def add_noise(waveform: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    rng = rng or np.random.default_rng()
    signal_power = np.mean(waveform**2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape)
    return (waveform + noise.astype(np.float32)).astype(np.float32)


def mel_spectrogram(waveform: np.ndarray, sr: int, n_mels: int, hop: int) -> np.ndarray:
    if waveform.size == 0:
        return np.zeros((n_mels, 0), np.float32)
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, hop_length=hop
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
