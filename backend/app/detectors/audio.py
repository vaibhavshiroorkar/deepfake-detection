"""
Audio forensics — Phase 2.

Primary signal: **Whisper classifier** — a trained MLP on top of frozen
Whisper-base encoder features.  The encoder was trained on ~680k hours of
real speech, making it a powerful feature extractor.  The classifier head
learns to distinguish real speech spectrograms from TTS/voice-clone outputs.

Fallback: if classifier head weights are not trained yet, the system uses
the original heuristic Whisper signal (adjacent-frame cosine similarity).

Supporting signals (kept from Phase 1):
  - Pitch variability (autocorrelation F0 tracking)
  - Silence floor (noise floor in quiet stretches)
  - Energy rhythm (envelope burstiness)
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


# ---------------------------------------------------------------------------
# Whisper-based signals (Phase 2 trained classifier + Phase 1 heuristic)
# ---------------------------------------------------------------------------

def _whisper_encoder_forward(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Run Whisper encoder and return hidden states (T, 512). None on failure."""
    try:
        import torch
        from ..models import get_device, get_whisper_encoder
    except Exception:
        return None

    if sr != 16000:
        new_len = int(round(len(audio) * 16000 / sr))
        if new_len < 16000:
            return None
        audio16 = sp_signal.resample(audio, new_len).astype(np.float32)
    else:
        audio16 = audio.astype(np.float32)

    try:
        encoder, processor = get_whisper_encoder()
        device = get_device()
        feats = processor(
            audio16, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            hidden = encoder(feats).last_hidden_state[0]  # [T, D]
        return hidden.detach().cpu().numpy()
    except Exception:
        return None


def _whisper_classifier_signal(hidden: np.ndarray) -> Signal:
    """Phase 2: trained classifier on Whisper encoder features."""
    try:
        from ..models import get_device, get_whisper_classifier
        from .whisper_classifier import predict_audio

        head = get_whisper_classifier()
        device = get_device()
        p_fake, p_real, stats = predict_audio(hidden, head, device)

        score = float(np.clip(p_fake, 0.0, 1.0))
        if score > 0.7:
            detail = (
                f"Whisper classifier: P(synthetic) = {p_fake:.3f}. "
                "Strong confidence in synthetic audio — spectral features diverge from real speech."
            )
        elif score > 0.5:
            detail = (
                f"Whisper classifier: P(synthetic) = {p_fake:.3f}. "
                "Classifier leans toward synthetic. Some spectral anomalies detected."
            )
        elif score > 0.35:
            detail = (
                f"Whisper classifier: P(synthetic) = {p_fake:.3f}. "
                "Classifier is uncertain. Features are borderline."
            )
        else:
            detail = (
                f"Whisper classifier: P(synthetic) = {p_fake:.3f}. "
                "Audio features look like real speech to the classifier."
            )
        return Signal("Whisper classifier", score, detail)
    except Exception as exc:
        return Signal(
            "Whisper classifier",
            0.0,
            f"Whisper classifier unavailable ({type(exc).__name__}); skipping.",
        )


def _whisper_heuristic_signal(hidden: np.ndarray) -> Signal:
    """Phase 1 heuristic: adjacent-frame cosine similarity + variance."""
    h = hidden
    if h.shape[0] < 4:
        return Signal("Whisper encoder", 0.2,
                      "Encoder produced too few frames to analyse.")

    feat_var = float(np.mean(np.var(h, axis=0)))
    feat_norm = float(np.mean(np.linalg.norm(h, axis=-1)))
    norms = np.linalg.norm(h, axis=-1, keepdims=True)
    nh = h / (norms + 1e-9)
    adj_sim = float(np.mean(np.sum(nh[:-1] * nh[1:], axis=-1)))

    score = 0.0
    if adj_sim > 0.985:
        score += 0.55
    elif adj_sim > 0.965:
        score += 0.30
    elif adj_sim > 0.940:
        score += 0.15
    if feat_var < 0.40:
        score += 0.30
    elif feat_var < 0.80:
        score += 0.15

    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Whisper encoder",
        score,
        f"Adjacent-frame cosine {adj_sim:.4f}, per-dim variance {feat_var:.3f}, "
        f"mean norm {feat_norm:.2f} across {h.shape[0]} encoder frames. "
        + (
            "Encoder representations are too uniform — characteristic of TTS or voice cloning."
            if score > 0.45
            else "Encoder representations vary like real speech."
        ),
    )


def _whisper_signal(audio: np.ndarray, sr: int) -> list[Signal]:
    """Returns 1–2 signals: trained classifier (if available) + heuristic fallback."""
    hidden = _whisper_encoder_forward(audio, sr)
    if hidden is None:
        return [Signal(
            "Whisper encoder", 0.0,
            "Whisper-base unavailable or clip too short; skipping.",
        )]

    signals: list[Signal] = []

    # Always try the trained classifier first
    cls_signal = _whisper_classifier_signal(hidden)
    if "unavailable" not in cls_signal.detail:
        signals.append(cls_signal)

    # Always include the heuristic as supporting evidence
    signals.append(_whisper_heuristic_signal(hidden))

    return signals


# ---------------------------------------------------------------------------
# Pretrained deepfake audio detector (wav2vec2 binary classifier)
# ---------------------------------------------------------------------------

def _deepfake_audio_signal(audio: np.ndarray, sr: int) -> Signal:
    """Standalone pretrained audio deepfake classifier.

    Works out of the box — no custom training required. Provides a second,
    model-level opinion on top of the Whisper-based path.
    """
    try:
        import torch
        from ..models import get_audio_deepfake_detector, get_device

        if sr != 16000:
            new_len = int(round(len(audio) * 16000 / sr))
            if new_len < 16000:
                return Signal(
                    "Audio deepfake detector",
                    0.0,
                    "Clip is too short to resample to 16 kHz; skipping.",
                )
            audio16 = sp_signal.resample(audio, new_len).astype(np.float32)
        else:
            audio16 = audio.astype(np.float32)

        model, processor, lmap = get_audio_deepfake_detector()
        device = get_device()
        inputs = processor(
            audio16, sampling_rate=16000, return_tensors="pt", padding=True,
        )
        input_values = inputs.get("input_values", inputs.get("input_features"))
        if input_values is None:
            return Signal(
                "Audio deepfake detector",
                0.0,
                "Detector processor returned unexpected output format.",
            )
        input_values = input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        p_fake = float(probs[lmap["fake_index"]])

        if p_fake > 0.7:
            detail_tail = "Strong signal that the audio is synthetic."
        elif p_fake > 0.55:
            detail_tail = "Leans synthetic."
        elif p_fake < 0.3:
            detail_tail = "Looks like real speech."
        else:
            detail_tail = "Classifier is uncertain."
        return Signal(
            "Audio deepfake detector",
            p_fake,
            f"{lmap['name']}: P(synthetic) = {p_fake:.3f}. {detail_tail}",
        )
    except Exception as exc:
        return Signal(
            "Audio deepfake detector",
            0.0,
            f"Pretrained audio detector unavailable ({type(exc).__name__}); skipping.",
        )


# ---------------------------------------------------------------------------
# Phase 1 heuristic signals (kept as supporting evidence)
# ---------------------------------------------------------------------------

def _pitch_signal(audio: np.ndarray, sr: int) -> Signal:
    """Track fundamental frequency (F0) in voiced frames via
    autocorrelation and measure how much it varies."""
    frame = int(0.04 * sr)  # 40 ms
    hop = int(0.02 * sr)
    if len(audio) < frame * 2:
        return Signal("Pitch variability", 0.3, "Clip too short for a pitch reading.")

    min_p, max_p = int(sr / 400), int(sr / 70)
    pitches = []
    for i in range(0, len(audio) - frame, hop):
        seg = audio[i:i + frame]
        if np.abs(seg).mean() < 0.01:
            continue
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
    """Look at the noise floor during quiet stretches."""
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


def _energy_rhythm_signal(audio: np.ndarray, sr: int) -> Signal:
    """Short-term energy skew."""
    frame = int(0.03 * sr)
    env = np.array([
        np.abs(audio[i:i + frame]).mean()
        for i in range(0, len(audio) - frame, frame)
    ])
    if len(env) < 20:
        return Signal("Energy rhythm", 0.25, "Clip too short for an envelope reading.")
    env = env / (env.max() + 1e-9)
    diffs = np.abs(np.diff(env))
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


# ---------------------------------------------------------------------------
# Verdict aggregation — Phase 2 weights
# ---------------------------------------------------------------------------

def _verdict(signals: list[Signal]) -> tuple[float, str]:
    weights = {
        "Whisper classifier":      2.5,   # Phase 2 primary — trained model
        "Audio deepfake detector": 2.2,   # Pretrained wav2vec2 binary classifier
        "Whisper encoder":         1.2,   # Phase 1 heuristic — supporting
        "Pitch variability":       1.0,
        "Silence floor":           0.9,
        "Energy rhythm":           0.7,
    }
    contributing = [s for s in signals if "unavailable" not in s.detail]
    num = sum(s.score * weights.get(s.name, 0.5) for s in contributing)
    denom = sum(weights.get(s.name, 0.5) for s in contributing)
    score = num / denom if denom else 0.0
    peak = max((s.score for s in contributing), default=0.0)
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

    # Whisper signals (trained classifier + heuristic) — computed together
    # since they share the encoder forward pass.
    whisper_signals = _whisper_signal(audio, sr)

    signals = [
        *whisper_signals,
        _safe(_deepfake_audio_signal, "Audio deepfake detector"),
        _safe(_pitch_signal, "Pitch variability"),
        _safe(_silence_floor_signal, "Silence floor"),
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
