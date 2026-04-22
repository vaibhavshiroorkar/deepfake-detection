"""
Audio forensics. Heuristic signals for detecting synthesized speech.

Real voices carry a messy signature: pitch wanders, breath appears,
sensor noise fills the quiet parts, high frequencies roll off the way
a physical room and microphone demand. TTS and voice-clone output
often forgets one of these.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal


@dataclass
class Signal:
    name: str
    score: float
    detail: str


def _load(data: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if not sr or sr <= 0:
        raise ValueError("invalid sample rate")
    # Cap at 30 seconds for analysis
    max_samples = sr * 30
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio, sr


def _pitch_signal(audio: np.ndarray, sr: int) -> Signal:
    """Track fundamental frequency (F0) in voiced frames via
    autocorrelation and measure how much it varies. Human speech
    moves pitch constantly; TTS and some clone models are flatter."""
    frame = int(0.04 * sr)  # 40 ms
    hop = int(0.02 * sr)
    if len(audio) < frame * 2:
        return Signal("Pitch variability", 0.3, "Clip too short for a pitch reading.")

    min_p, max_p = int(sr / 400), int(sr / 70)  # 70–400 Hz voiced range
    pitches = []
    for i in range(0, len(audio) - frame, hop):
        seg = audio[i:i + frame]
        if np.abs(seg).mean() < 0.01:
            continue  # silence
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        if max_p >= len(ac):
            continue
        peak = np.argmax(ac[min_p:max_p]) + min_p
        if ac[peak] > 0.35:
            pitches.append(sr / peak)

    if len(pitches) < 8:
        return Signal(
            "Pitch variability",
            0.25,
            "Not enough voiced frames to measure pitch movement reliably.",
        )

    pitches = np.array(pitches)
    cv = float(pitches.std() / pitches.mean()) if pitches.mean() > 0 else 0.0
    # Natural speech cv ~0.15–0.30. TTS tends lower.
    if cv < 0.08:
        score = 0.85
    elif cv < 0.13:
        score = 0.55
    elif cv < 0.20:
        score = 0.25
    else:
        score = 0.1
    return Signal(
        "Pitch variability",
        score,
        f"Pitch coefficient of variation {cv:.3f} across {len(pitches)} voiced frames. "
        + (
            "Flatter than natural speech — consistent with TTS or voice cloning."
            if score > 0.5
            else "Pitch moves the way a person's does when they talk."
        ),
    )


def _silence_floor_signal(audio: np.ndarray, sr: int) -> Signal:
    """Look at the noise floor during quiet stretches. Real microphones
    pick up a continuous low-level hiss; TTS often delivers perfectly
    clean silence or a suspiciously even synthetic noise."""
    frame = int(0.05 * sr)
    rms = np.array([
        np.sqrt(np.mean(audio[i:i + frame] ** 2) + 1e-12)
        for i in range(0, len(audio) - frame, frame)
    ])
    if len(rms) < 6:
        return Signal("Silence floor", 0.3, "Clip too short to profile the silence.")
    quiet_frames = rms[rms < np.percentile(rms, 25)]
    if len(quiet_frames) < 3:
        return Signal("Silence floor", 0.2, "No clear quiet sections to analyze.")
    floor = float(quiet_frames.mean())
    floor_var = float(quiet_frames.std())
    # Perfect digital silence → floor near 0, variance near 0 → suspicious.
    score = 0.0
    if floor < 1e-4:
        score += 0.6
    elif floor < 5e-4:
        score += 0.3
    if floor_var < 1e-5:
        score += 0.35
    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Silence floor",
        score,
        f"Quiet-section RMS {floor:.5f}, variance {floor_var:.6f}. "
        + (
            "Silence is unnaturally clean — no microphone noise, no room tone."
            if score > 0.45
            else "The quiet parts carry natural room tone and microphone noise."
        ),
    )


def _spectral_signal(audio: np.ndarray, sr: int) -> Signal:
    """High-frequency roll-off and spectral flatness. A lot of neural
    vocoders smear or clip content above 4–6 kHz in characteristic ways."""
    # Welch PSD for stability
    nperseg = min(2048, len(audio))
    if nperseg < 256:
        return Signal("Spectral profile", 0.3, "Clip too short to profile the spectrum.")
    freqs, psd = sp_signal.welch(audio, fs=sr, nperseg=nperseg)
    psd = np.maximum(psd, 1e-12)

    total = psd.sum()
    if total <= 0:
        return Signal("Spectral profile", 0.3, "Silent clip.")
    # Ratio of energy above Nyquist/2 vs total — real speech has
    # meaningful energy up there; neural vocoders often cut it off.
    nyq = sr / 2
    hi_cut = freqs >= nyq * 0.55
    hi_energy_ratio = float(psd[hi_cut].sum() / total)
    # Spectral flatness (geometric mean / arithmetic mean)
    flatness = float(np.exp(np.mean(np.log(psd))) / psd.mean())

    score = 0.0
    if hi_energy_ratio < 0.003:
        score += 0.55
    elif hi_energy_ratio < 0.01:
        score += 0.25
    if flatness < 0.01:
        score += 0.3
    elif flatness > 0.25:
        score += 0.25
    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Spectral profile",
        score,
        f"High-band energy share {hi_energy_ratio*100:.2f}%, spectral flatness {flatness:.3f}. "
        + (
            "High frequencies roll off too sharply, or the spectrum is oddly flat — possible vocoder signature."
            if score > 0.45
            else "Frequency distribution looks like a real microphone capture."
        ),
    )


def _energy_rhythm_signal(audio: np.ndarray, sr: int) -> Signal:
    """Short-term energy skew. Natural speech has breathy attacks and
    trailing decays. Synthesis can flatten these transients."""
    frame = int(0.03 * sr)
    env = np.array([
        np.abs(audio[i:i + frame]).mean()
        for i in range(0, len(audio) - frame, frame)
    ])
    if len(env) < 20:
        return Signal("Energy rhythm", 0.25, "Clip too short for an envelope reading.")
    env = env / (env.max() + 1e-9)
    diffs = np.abs(np.diff(env))
    # Natural speech has bursty transitions; TTS smooths them.
    burstiness = float(diffs.std())
    score = float(np.clip(1.0 - burstiness * 4.5, 0.0, 1.0)) * 0.7
    return Signal(
        "Energy rhythm",
        score,
        f"Envelope burstiness {burstiness:.3f}. "
        + (
            "Volume envelope is unusually smooth — attacks and releases aren't landing the way a human speaker's would."
            if score > 0.5
            else "Speech envelope moves in natural bursts and decays."
        ),
    )


def _verdict(signals: list[Signal]) -> tuple[float, str]:
    weights = {
        "Pitch variability": 1.0,
        "Silence floor": 0.9,
        "Spectral profile": 1.0,
        "Energy rhythm": 0.7,
    }
    num = sum(s.score * weights.get(s.name, 0.5) for s in signals)
    denom = sum(weights.get(s.name, 0.5) for s in signals)
    score = num / denom if denom else 0.0
    peak = max((s.score for s in signals), default=0.0)
    score = 0.7 * score + 0.3 * peak
    if score < 0.3:
        label = "likely authentic"
    elif score < 0.55:
        label = "inconclusive"
    elif score < 0.75:
        label = "likely synthesized"
    else:
        label = "highly likely synthesized"
    return float(np.clip(score, 0.0, 1.0)), label


def analyze_audio(data: bytes, filename: str = "audio") -> dict[str, Any]:
    try:
        audio, sr = _load(data)
    except Exception as e:
        return {
            "kind": "audio",
            "filename": filename,
            "error": f"Could not decode audio: {e}. Try WAV, FLAC, OGG, or MP3.",
            "suspicion": 0.0,
            "verdict": "unreadable",
            "signals": [],
        }

    duration = len(audio) / sr if sr else 0
    if len(audio) == 0 or float(np.abs(audio).max()) < 1e-6:
        return {
            "kind": "audio",
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "error": "Audio is silent or empty.",
            "suspicion": 0.0,
            "verdict": "unreadable",
            "signals": [],
        }

    def _safe(fn, name: str) -> Signal:
        try:
            return fn(audio, sr)
        except Exception as exc:  # noqa: BLE001
            return Signal(name, 0.2, f"Could not compute this signal ({exc}).")

    signals = [
        _safe(_pitch_signal, "Pitch variability"),
        _safe(_silence_floor_signal, "Silence floor"),
        _safe(_spectral_signal, "Spectral profile"),
        _safe(_energy_rhythm_signal, "Energy rhythm"),
    ]
    score, label = _verdict(signals)

    return {
        "kind": "audio",
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "suspicion": score,
        "verdict": label,
        "confidence": round(abs(score - 0.5) * 2, 3),
        "signals": [
            {"name": s.name, "score": round(s.score, 3), "detail": s.detail}
            for s in signals
        ],
    }
