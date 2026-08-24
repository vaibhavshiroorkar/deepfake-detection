from __future__ import annotations

import cv2
import numpy as np

from .tracking import Landmarks5

LANDMARK_TEMPLATE_REVISION = "five-point-lower-face-v1"
NORMALIZED_FACE_TEMPLATE = (
    (0.315, 0.370),
    (0.685, 0.370),
    (0.500, 0.560),
    (0.365, 0.760),
    (0.635, 0.760),
)
CANONICAL_FACE_SIZE = (224, 224)
# Coordinates are left, top, right, and bottom in canonical pixels.
LOWER_FACE_REGION = (34, 107, 190, 219)


def _points(landmarks: Landmarks5) -> np.ndarray:
    return np.asarray(
        (
            (landmarks.eye_left.x, landmarks.eye_left.y),
            (landmarks.eye_right.x, landmarks.eye_right.y),
            (landmarks.nose.x, landmarks.nose.y),
            (landmarks.mouth_left.x, landmarks.mouth_left.y),
            (landmarks.mouth_right.x, landmarks.mouth_right.y),
        ),
        dtype=np.float64,
    )


def _target_points(output_size: tuple[int, int]) -> np.ndarray:
    height, width = output_size
    if height <= 0 or width <= 0:
        raise ValueError("Alignment output dimensions must be positive")
    scale = np.asarray((width - 1, height - 1), dtype=np.float64)
    return np.asarray(NORMALIZED_FACE_TEMPLATE, dtype=np.float64) * scale


def similarity_transform(
    source: Landmarks5,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Fit all five points to the template for a (height, width) output."""

    source_points = _points(source)
    if not np.isfinite(source_points).all():
        raise ValueError("Face landmarks must be finite")
    centered = source_points - source_points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-8) < 2:
        raise ValueError("Face landmark geometry is degenerate")

    target_points = _target_points(output_size)
    equations = np.empty((10, 4), dtype=np.float64)
    targets = np.empty(10, dtype=np.float64)
    for index, ((x, y), (target_x, target_y)) in enumerate(
        zip(source_points, target_points, strict=True)
    ):
        equations[2 * index] = (x, -y, 1.0, 0.0)
        equations[2 * index + 1] = (y, x, 0.0, 1.0)
        targets[2 * index] = target_x
        targets[2 * index + 1] = target_y
    parameters, _, rank, _ = np.linalg.lstsq(equations, targets, rcond=None)
    if rank != 4 or not np.isfinite(parameters).all():
        raise ValueError("Face landmark geometry is degenerate")
    scale_cosine, scale_sine, translation_x, translation_y = parameters
    if np.hypot(scale_cosine, scale_sine) <= 1e-8:
        raise ValueError("Face landmark geometry is degenerate")
    return np.asarray(
        (
            (scale_cosine, -scale_sine, translation_x),
            (scale_sine, scale_cosine, translation_y),
        ),
        dtype=np.float64,
    )


def aligned_lower_face(
    frame: np.ndarray,
    landmarks: Landmarks5,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Return a normalized CHW lower-face view from one BGR frame."""

    if height <= 0 or width <= 0:
        raise ValueError("Lower-face output dimensions must be positive")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Lower-face alignment expects a three-channel BGR image")
    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("Lower-face alignment expects a nonempty frame")
    points = _points(landmarks)
    if not np.isfinite(points).all():
        raise ValueError("Face landmarks must be finite")
    if not (
        np.all((0 <= points[:, 0]) & (points[:, 0] < frame_width))
        and np.all((0 <= points[:, 1]) & (points[:, 1] < frame_height))
    ):
        raise ValueError("Face landmarks must lie inside the frame")

    canonical_height, canonical_width = CANONICAL_FACE_SIZE
    matrix = similarity_transform(
        landmarks,
        output_size=CANONICAL_FACE_SIZE,
    )
    canonical = cv2.warpAffine(
        frame,
        matrix,
        (canonical_width, canonical_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    left, top, right, bottom = LOWER_FACE_REGION
    lower_face = canonical[top:bottom, left:right]
    resized = cv2.resize(lower_face, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (rgb - mean) / standard_deviation
    return normalized.transpose(2, 0, 1)
