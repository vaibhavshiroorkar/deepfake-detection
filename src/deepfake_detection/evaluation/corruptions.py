from __future__ import annotations

import cv2
import numpy as np


def add_audio_noise(
    waveform: np.ndarray,
    *,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(values))))
    if signal_rms == 0:
        return values.copy()
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(values.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    return values + noise * (target_noise_rms / noise_rms)


def prepend_silence(waveform: np.ndarray, *, silence_samples: int) -> np.ndarray:
    values = np.asarray(waveform)
    if silence_samples < 0:
        raise ValueError("Silence sample count cannot be negative")
    if silence_samples == 0:
        return values.copy()
    shifted = np.zeros_like(values)
    if silence_samples < len(values):
        shifted[silence_samples:] = values[:-silence_samples]
    return shifted


def compress_video_frames(
    frames: np.ndarray,
    *,
    jpeg_quality: int,
) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError(
            "Frames must have shape [time, height, width, 3] and uint8 data"
        )
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    decoded: list[np.ndarray] = []
    for frame in values:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality),
        )
        if not ok:
            raise ValueError("OpenCV could not encode a corruption frame")
        restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if restored is None:
            raise ValueError("OpenCV could not decode a corruption frame")
        decoded.append(restored)
    return np.stack(decoded)


def degrade_resolution(frames: np.ndarray, *, scale: float) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Frames must have shape [time, height, width, 3]")
    if not 0 < scale < 1:
        raise ValueError("Resolution scale must be in (0, 1)")
    height, width = values.shape[1:3]
    reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.stack(
        [
            cv2.resize(
                cv2.resize(frame, reduced_size, interpolation=cv2.INTER_AREA),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            for frame in values
        ]
    )
