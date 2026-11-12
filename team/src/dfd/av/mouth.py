"""Audio-visual windows: 25 fps frames, 16 kHz audio and AV-HuBERT lip crops.

AV-HuBERT was pretrained on 96 x 96 grey mouth regions cut from faces aligned to
a mean face, then centre-cropped to 88. Its release aligns with dlib's 68
landmarks; here MediaPipe's Face Landmarker stands in. Each frame gets a
similarity transform that levels the eye line, scales the distance between the
eye centres to `eye_px` and puts the mouth centre in the middle of the crop.
Angle, scale and centre are averaged over `smooth` frames, as the release
smooths its landmarks, so the crop does not jitter from frame to frame.

`eye_px` is not AV-HuBERT's value (its mean face is not used here). It is a
parameter, chosen on development clips by `scripts/av/tune_mouth_crop.py`.

    read_window(path, start, seconds)   frames [T, H, W, 3] RGB uint8 at 25 fps, audio float32
    MouthCropper(model).lips(frames)    [T, 88, 88] float32 normalised, plus a found mask
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dfd.av.avhubert import LIP_MEAN, LIP_SIZE, LIP_STD, SAMPLE_RATE
from dfd.prep.video import probe

FPS = 25
CROP = 96
DEFAULT_EYE_PX = 64.0
DEFAULT_SMOOTH = 12
DEFAULT_MODEL = Path("data/models/mediapipe/face_landmarker.task")

# MediaPipe Face Mesh indices: the corners of each eye, and the mouth corners
# with the inner midpoints of the upper and lower lip.
RIGHT_EYE = (33, 133)
LEFT_EYE = (362, 263)
MOUTH = (61, 291, 13, 14)


def read_window(path: str | Path, start: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """`seconds` of video from `start`, resampled by ffmpeg to 25 fps and 16 kHz mono.

    Both streams are cut with the same input seek, so frame k and audio sample
    k * 640 refer to the same instant. A clip without audio gives an empty array.
    """
    info = probe(path)
    seek = ["-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path)]
    video = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", *seek, "-an", "-r", str(FPS),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if video.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {video.stderr.decode(errors='replace')[:300]}")
    frames = np.frombuffer(video.stdout, np.uint8).reshape(-1, info.height, info.width, 3)
    audio = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", *seek, "-vn", "-map", "0:a:0?", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if audio.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {audio.stderr.decode(errors='replace')[:300]}")
    return frames, np.frombuffer(audio.stdout, np.float32).copy()


@dataclass(frozen=True)
class Pose:
    """Where the mouth is in one frame: centre (x, y), eye-line angle in radians, eye distance."""

    cx: float
    cy: float
    angle: float
    eyes: float


def pose_from_landmarks(points: np.ndarray) -> Pose:
    """`points` [478, 2] in pixels."""
    right = points[list(RIGHT_EYE)].mean(0)
    left = points[list(LEFT_EYE)].mean(0)
    dx, dy = left - right
    centre = points[list(MOUTH)].mean(0)
    return Pose(float(centre[0]), float(centre[1]), float(np.arctan2(dy, dx)), float(np.hypot(dx, dy)))


def smooth_poses(poses: list[Pose | None], window: int = DEFAULT_SMOOTH) -> list[Pose | None]:
    """Centred moving average over the frames where a face was found.

    A frame without a face stays None; it is not filled from its neighbours,
    so the caller decides what a gap means.
    """
    found = [i for i, p in enumerate(poses) if p is not None]
    if not found:
        return list(poses)
    table = np.array([[poses[i].cx, poses[i].cy, poses[i].angle, poses[i].eyes] for i in found])
    out: list[Pose | None] = [None] * len(poses)
    half = window // 2
    for index in found:
        near = [r for r, i in enumerate(found) if abs(i - index) <= half]
        out[index] = Pose(*table[near].mean(0))
    return out


def crop_mouth(frame: np.ndarray, pose: Pose, eye_px: float = DEFAULT_EYE_PX) -> np.ndarray:
    """A [96, 96] grey uint8 mouth crop of an RGB frame."""
    scale = eye_px / max(pose.eyes, 1e-6)
    matrix = cv2.getRotationMatrix2D((pose.cx, pose.cy), np.degrees(pose.angle), scale)
    matrix[:, 2] += (CROP / 2 - pose.cx, CROP / 2 - pose.cy)
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.warpAffine(grey, matrix, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def normalise_lips(crops: np.ndarray) -> np.ndarray:
    """[T, 96, 96] uint8 to [T, 88, 88] float32 as AV-HuBERT's loader feeds the model."""
    offset = (CROP - LIP_SIZE) // 2
    centre = crops[:, offset : offset + LIP_SIZE, offset : offset + LIP_SIZE].astype(np.float32) / 255.0
    return (centre - LIP_MEAN) / LIP_STD


class MouthCropper:
    """MediaPipe Face Landmarker in video mode, one instance per clip at a time."""

    def __init__(self, model: str | Path = DEFAULT_MODEL, *, eye_px: float = DEFAULT_EYE_PX,
                 smooth: int = DEFAULT_SMOOTH) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        self._options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.eye_px = eye_px
        self.smooth = smooth

    def poses(self, frames: np.ndarray) -> list[Pose | None]:
        import mediapipe as mp

        out: list[Pose | None] = []
        # A fresh landmarker per clip: video mode needs rising timestamps.
        with self._vision.FaceLandmarker.create_from_options(self._options) as landmarker:
            for index, frame in enumerate(frames):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                result = landmarker.detect_for_video(image, int(index * 1000 / FPS))
                if not result.face_landmarks:
                    out.append(None)
                    continue
                height, width = frame.shape[:2]
                points = np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]])
                out.append(pose_from_landmarks(points))
        return smooth_poses(out, self.smooth)

    def lips(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalised lips [T, 88, 88] and a bool mask of frames with a face.

        Frames without a face get the crop of the nearest frame that has one,
        so the sequence keeps its timing; the mask says which ones those are.
        """
        poses = self.poses(frames)
        found = np.array([p is not None for p in poses])
        if not found.any():
            return np.zeros((len(frames), LIP_SIZE, LIP_SIZE), np.float32), found
        have = np.flatnonzero(found)
        crops = np.stack([
            crop_mouth(frame, poses[have[np.abs(have - i).argmin()]], self.eye_px)
            for i, frame in enumerate(frames)
        ])
        return normalise_lips(crops), found
