"""Pure per-step audio preprocessing ops. Mono waveform float32 [n] in/out
unless noted. No Streamlit, no I/O."""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


def downmix(wav_2d) -> np.ndarray:
    arr = np.asarray(wav_2d, dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def resample(wav, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or wav.size == 0:
        return wav.astype(np.float32)
    return librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(wav, sr: int, top_db: float):
    if wav.size == 0:
        return wav, 0.0
    trimmed, index = librosa.effects.trim(wav, top_db=top_db)
    return trimmed, float(index[0] / sr)


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


def add_noise(wav, snr_db: float, rng=None) -> np.ndarray:
    if wav.size == 0:
        return wav
    rng = rng or np.random.default_rng()
    sig_power = np.mean(wav ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def extract_windows(wav, sr: int, timestamps, window_sec: float) -> np.ndarray:
    win = int(window_sec * sr)
    out = []
    for t in timestamps:
        center = int(t * sr)
        start = max(0, center - win // 2)
        end = start + win
        if end > len(wav):
            end = len(wav)
            start = max(0, end - win)
        w = wav[start:end]
        if len(w) < win:
            w = np.pad(w, (0, win - len(w)))
        out.append(w)
    return np.stack(out).astype(np.float32) if out else np.zeros((0, win), np.float32)


def mel_spectrogram(wav, sr: int, n_mels: int, hop: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((n_mels, 0), np.float32)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, hop_length=hop)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
