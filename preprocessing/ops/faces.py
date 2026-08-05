"""MAIN visual preprocessing steps (RGB in, RGB out unless noted).

Each step is a standalone, pure function so pages/tests can call them in
isolation; `detect_crop` composes them into the single face+mouth path that both
extract_clip.py (the real pipeline) and the dashboard use.

Which detector runs is not decided here. These functions take one, and the
choice between MTCNN and YuNet lives in preprocessing/ops/detectors.py.

The face crop is a margin-padded bounding box. Five-point similarity alignment
used to sit here and was removed; it is parked in docs/ideas.md with the two
traps it carried, so re-adding it does not start from scratch.
"""
import cv2
import numpy as np

from preprocessing.ops.constants import (
    FRAME_SIZE, MOUTH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


def _resize(img, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def detect(frame_rgb, detector, conf_thresh: float):
    """Run a detector. Returns (box, landmarks5, prob) or (None, None, None).

    `detector` is anything from preprocessing/ops/detectors.py. Only the single
    most-confident face is considered, and it must clear conf_thresh. This is the
    one place the threshold is applied, so it means the same thing whichever
    detector is loaded.
    """
    box, landmarks5, prob = detector.detect(frame_rgb)
    if box is None or prob < conf_thresh:
        return None, None, None
    return box, landmarks5, prob


def crop_and_resize(frame_rgb, box, size: int = FRAME_SIZE, margin: float = 0.2):
    """Margin-padded bbox crop, resized to size×size RGB. None if the crop is empty.

    The only face-crop path. `margin` pads the box and clamps to the frame, so
    unlike a warped canvas it can never introduce padding of its own.
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1m, y1m = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    x2m, y2m = min(w, int(x2 + mw)), min(h, int(y2 + mh))
    crop = frame_rgb[y1m:y2m, x1m:x2m]
    if crop.size == 0:
        return None
    return _resize(crop, size)


def mouth_roi(frame_rgb, landmarks5, size: int = MOUTH_SIZE):
    """Square mouth crop centered on the two mouth-corner landmarks (points 3,4).

    The single mouth-crop implementation (feeds the lip-sync stream, Stage 4).
    Returns size×size RGB, or None if the region is empty.
    """
    ml, mr = landmarks5[3], landmarks5[4]
    cx, cy = (ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0
    corner_dist = float(np.hypot(mr[0] - ml[0], mr[1] - ml[1]))
    half = max(corner_dist * 0.9, 20.0)
    h, w = frame_rgb.shape[:2]
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    region = frame_rgb[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return _resize(region, size)


def detect_crop(frame_rgb, detector, conf_thresh: float = 0.9,
                margin: float = 0.2,
                size: int = FRAME_SIZE, mouth_size: int = MOUTH_SIZE):
    """One detect call → (face size×size, mouth mouth_size×mouth_size, detected).

    The single face+mouth code path. When no face clears conf_thresh, both
    outputs fall back to a plain resize of the frame and detected=False, so the
    caller always gets fixed-shape crops (shape must stay fixed for batching).
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


def imagenet_normalize(img_uint8) -> np.ndarray:
    """RGB uint8 → float32 (H,W,3), zero-centered with ImageNet mean/std."""
    arr = img_uint8.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def normalized_range(arr) -> tuple[float, float]:
    return float(arr.min()), float(arr.max())
