"""EXTRAS — audio enhancement/degradation ops. Mono float32 [n] in/out unless noted.

Not part of the stored contract — robustness/augmentation probes toggled
independently in the dashboard (baseline = all off = the real pipeline).
mel_spectrogram is a visualization/feature view, not a pipeline output.
See docs/preprocessing.md.
"""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


def spectral_denoise(wav, sr: int, strength: float) -> np.ndarray:
    """Simple spectral gating: attenuate bins below strength*noise-floor."""
    if wav.size == 0:
        return wav
    stft = librosa.stft(wav)
    mag, phase = np.abs(stft), np.angle(stft)
    floor = np.median(mag, axis=1, keepdims=True) * strength
    mag = np.maximum(mag - floor, 0.0)
    out = librosa.istft(mag * np.exp(1j * phase), length=len(wav))
    return out.astype(np.float32)


def rms_normalize(wav, target_db: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-9
    target_rms = 10 ** (target_db / 20.0)
    return np.clip(wav * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def bandpass(wav, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    nyq = sr / 2.0
    low, high = max(low_hz / nyq, 1e-4), min(high_hz / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def add_noise(wav, snr_db: float, rng=None) -> np.ndarray:
    if wav.size == 0:
        return wav
    rng = rng or np.random.default_rng()
    sig_power = np.mean(wav ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def mel_spectrogram(wav, sr: int, n_mels: int, hop: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((n_mels, 0), np.float32)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, hop_length=hop)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
