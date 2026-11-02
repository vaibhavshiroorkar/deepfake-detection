"""Frozen AV-HuBERT features of 2 s windows, cached for the trained T-AV heads (I-8b, I-8c).

One window gives three passes of the same encoder, each [50, width] at 25 steps per second:

    fused   lips and audio together
    lips    lips only (audio stream zeroed, as in modality dropout)
    audio   audio only

stored as float16 in one .npz per window with the share of frames that had a face.
Decoding and MediaPipe run in worker processes (`prepare_window`); the encoder
runs in the caller's process.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from dfd.av.avhubert import AVHubert
from dfd.av.mouth import FPS, MouthCropper, crop_mouth, normalise_lips, read_window
from dfd.av.sync import model_inputs

WINDOW_S = 2.0
FRAMES = round(WINDOW_S * FPS)

_cropper: MouthCropper | None = None


def init_worker(landmarker: str | Path, eye_px: float) -> None:
    global _cropper
    _cropper = MouthCropper(landmarker, eye_px=eye_px)


def prepare_window(job: tuple[str, float]) -> tuple[float | str, np.ndarray | None, np.ndarray | None]:
    """(path, start) to (face share, uint8 crops [50, 96, 96], audio); a failure gives (message, None, None)."""
    path, start = job
    try:
        frames, audio = read_window(path, start, WINDOW_S)
        frames = frames[:FRAMES]
        poses = _cropper.poses(frames)
        found = np.array([p is not None for p in poses])
        if not found.any():
            return 0.0, None, audio
        have = np.flatnonzero(found)
        crops = np.stack([crop_mouth(f, poses[have[np.abs(have - i).argmin()]], _cropper.eye_px) for i, f in enumerate(frames)])
        return float(found.mean()), crops, audio
    except Exception as error:  # counted by the caller, not fatal
        return f"{type(error).__name__}: {str(error)[:120]}", None, None


@torch.no_grad()
def encode(model: AVHubert, crops: np.ndarray, audio: np.ndarray, device: str = "cuda") -> dict[str, np.ndarray]:
    video, sound = model_inputs(normalise_lips(crops), audio, device)
    return {
        "fused": model(video=video, audio=sound)[0].half().cpu().numpy(),
        "lips": model(video=video)[0].half().cpu().numpy(),
        "audio": model(audio=sound)[0].half().cpu().numpy(),
    }


def cache_path(root: Path, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode()).hexdigest()
    return root / digest[:2] / f"{digest}.npz"
