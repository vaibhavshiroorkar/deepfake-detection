"""Visual preprocessing steps (RGB in, RGB out unless noted).

Each step is a standalone pure function, so pages and tests can call them in
isolation. `detect_crop` composes them into the single face plus mouth path.

Which detector runs is not decided here. These functions take one, and the
choice lives in `detectors`. No thresholding happens in the detectors: `detect`
is the one place `conf_thresh` is applied, so it means the same thing whichever
detector is loaded.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import FRAME_SIZE, IMAGENET_MEAN, IMAGENET_STD, MOUTH_SIZE


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def detect(
    frame_rgb: np.ndarray, detector: object, conf_thresh: float
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """Run a detector. Returns (box, landmarks5, prob) or (None, None, None).

    Only the single most-confident face is considered, and it must clear
    conf_thresh.
    """
    box, landmarks5, prob = detector.detect(frame_rgb)
    if box is None or prob < conf_thresh:
        return None, None, None
    return box, landmarks5, prob


def crop_and_resize(
    frame_rgb: np.ndarray,
    box: np.ndarray,
    size: int = FRAME_SIZE,
    margin: float = 0.2,
) -> np.ndarray | None:
    """Margin-padded bbox crop, resized to size x size RGB.

    Returns None if the crop is empty. `margin` pads the box and clamps to the
    frame, so unlike a warped canvas it can never introduce padding of its own.
    """
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pad_w, pad_h = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1m, y1m = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
    x2m, y2m = min(width, int(x2 + pad_w)), min(height, int(y2 + pad_h))
    crop = frame_rgb[y1m:y2m, x1m:x2m]
    if crop.size == 0:
        return None
    return _resize(crop, size)


def mouth_roi(
    frame_rgb: np.ndarray, landmarks5: np.ndarray, size: int = MOUTH_SIZE
) -> np.ndarray | None:
    """Square mouth crop centered on the two mouth-corner landmarks (points 3, 4).

    Feeds the lip-sync stream. Returns size x size RGB, or None if empty.
    """
    mouth_left, mouth_right = landmarks5[3], landmarks5[4]
    center_x = (mouth_left[0] + mouth_right[0]) / 2.0
    center_y = (mouth_left[1] + mouth_right[1]) / 2.0
    corner_dist = float(
        np.hypot(mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1])
    )
    half = max(corner_dist * 0.9, 20.0)
    height, width = frame_rgb.shape[:2]
    x1, x2 = int(max(0, center_x - half)), int(min(width, center_x + half))
    y1, y2 = int(max(0, center_y - half)), int(min(height, center_y + half))
    region = frame_rgb[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return _resize(region, size)


def detect_crop(
    frame_rgb: np.ndarray,
    detector: object,
    conf_thresh: float = 0.9,
    margin: float = 0.2,
    size: int = FRAME_SIZE,
    mouth_size: int = MOUTH_SIZE,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One detect call -> (face, mouth, detected).

    When no face clears conf_thresh, both outputs fall back to a plain resize of
    the frame and detected is False, so the caller always gets fixed-shape crops.
    Shapes must stay fixed for batching.
    """
    box, landmarks5, _ = detect(frame_rgb, detector, conf_thresh)
    if box is None:
        return _resize(frame_rgb, size), _resize(frame_rgb, mouth_size), False

    face = crop_and_resize(frame_rgb, box, size, margin)
    if face is None:
        face = _resize(frame_rgb, size)

    mouth = mouth_roi(frame_rgb, landmarks5, mouth_size)
    if mouth is None:
        mouth = _resize(face, mouth_size)
    return face, mouth, True


def imagenet_normalize(image_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 (H, W, 3), zero-centered with ImageNet mean and std."""
    arr = image_uint8.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def normalized_range(arr: np.ndarray) -> tuple[float, float]:
    return float(arr.min()), float(arr.max())
