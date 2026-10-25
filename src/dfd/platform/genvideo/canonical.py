"""Canonical preprocessing protocol v2 (E-019; v1 is preserved in scripts/canonical/canonicalise.py): the same pixels path for training, evaluation and live inference.

Every clip is re-encoded to the first 2.0 s at 7 fps (14 frames), 224 x 224 centre crop, identical codec settings and colour tags, no audio,
no metadata, so container, duration, frame-rate and encoder cues cannot reach the detector (E-018 showed they separate DeepAction perfectly).
This does NOT remove pixel-level or temporal residues; E-019 audits those.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

PROTOCOL_ID = "canonical-v3"      # v2 = v1 plus setparams (v1 left colour tags copied from the source, a perfect shortcut); v3 = v2 plus a fixed GOP/B-frame pattern (v2 left content-dependent GOP statistics, E-019 v2 result)
N_FRAMES = 14
SIZE = 224
WINDOW_S = 2.0
FPS = 7
VF_V1 = f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase:flags=area,crop={SIZE}:{SIZE}"
VF = VF_V1 + ",setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"


def ffmpeg_args(src, dst) -> list:
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1", "-map_chapters", "-1", "-an", "-sn",
            "-t", f"{WINDOW_S:.1f}", "-vf", VF, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "14",
            "-bf", "2", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-x264-params", "no-info=1:scenecut=0:b-adapt=0", "-movflags", "+faststart", str(dst)]


def canonicalise(src, dst, timeout_s: float = 120.0) -> None:
    r = subprocess.run(ffmpeg_args(src, dst), capture_output=True, text=True, timeout=timeout_s, check=False)
    if r.returncode != 0:
        raise ValueError("canonical re-encode failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:200])


def read_frames(path) -> np.ndarray:
    """The 14 RGB frames (14, 224, 224, 3) uint8 of a canonical clip. Raises ValueError if fewer than 14 frames can be read."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < N_FRAMES:
        raise ValueError(f"canonical clip has {len(out)} frames, {N_FRAMES} needed")
    return np.stack(out[:N_FRAMES])
