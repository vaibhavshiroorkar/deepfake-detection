"""Spike: does inner-lip opening track speech loudness on neutral FakeAVCeleb test clips?

Opening = distance between inner-lip landmarks 13 and 14 over the eye distance, per 25 fps frame.
Loudness = RMS of the 640 audio samples (40 ms) under each frame, in dB.
For each clip: the Pearson correlation of the two traces over the 2 s window from 0 s, best lag in
-2..+2 frames. Real clips should correlate; cloned-voice clips (real lips, other words) less.
"""

import csv
import random
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from deepfake_detection.avsync.mouth import (
    DEFAULT_MODEL,
    FPS,
    LEFT_EYE,
    RIGHT_EYE,
    read_window,
)

N, SECONDS, HOP = 20, 2.0, 640


def opening(frames: np.ndarray) -> np.ndarray:
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    out = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            result = landmarker.detect_for_video(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
                ),
                int(i * 1000 / FPS),
            )
            if not result.face_landmarks:
                out.append(np.nan)
                continue
            p = np.array([(q.x, q.y) for q in result.face_landmarks[0]])
            eyes = np.linalg.norm(
                p[list(LEFT_EYE)].mean(0) - p[list(RIGHT_EYE)].mean(0)
            )
            out.append(np.linalg.norm(p[13] - p[14]) / eyes)
    return np.array(out)


def loudness(audio: np.ndarray, frames: int) -> np.ndarray:
    audio = np.pad(audio, (0, max(0, frames * HOP - len(audio))))[: frames * HOP]
    rms = np.sqrt((audio.reshape(frames, HOP) ** 2).mean(1) + 1e-10)
    return 20 * np.log10(rms)


def best_r(a: np.ndarray, b: np.ndarray) -> float:
    best = -1.0
    for lag in range(-2, 3):
        x, y = (a[lag:], b[: len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() > 20 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            best = max(best, float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return best


def main() -> None:
    test = {
        r["clip_id"]
        for r in csv.DictReader(open("runs/p001/split/test.csv", encoding="utf-8"))
    }
    rows = [
        r
        for r in csv.DictReader(
            open("data/manifests/fakeavceleb-neutral.csv", encoding="utf-8")
        )
        if r["clip_id"] in test
    ]
    rng = random.Random(17)  # noqa: S311 - a fixed clip sample, not a secret
    for kind in ("RealVideo-RealAudio", "RealVideo-FakeAudio"):
        group = [r for r in rows if r["manipulation_type"] == kind]
        rs = []
        for r in rng.sample(group, min(N, len(group))):
            frames, audio = read_window(Path("data") / r["video_path"], 0.0, SECONDS)
            rs.append(best_r(opening(frames), loudness(audio, len(frames))))
        rs = np.array(rs)
        print(
            f"{kind}: n={len(rs)} median r={np.median(rs):.2f} "
            f"[{np.percentile(rs, 25):.2f}, {np.percentile(rs, 75):.2f}] share r>0.3={np.mean(rs > 0.3):.2f}"
        )


if __name__ == "__main__":
    main()
