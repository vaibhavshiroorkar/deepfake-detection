"""MAIN visual preprocessing steps (RGB in, RGB out unless noted).

Each step is a standalone, pure function so pages/tests can call them in
isolation; `detect_align_crop` composes them into the single face+mouth path
that both extract_clip.py (the real pipeline) and the dashboard use.

SOTA note: `align_face` is the key upgrade over a plain detect+crop — it warps
the face onto a canonical 5-point template so the temporal model sees a
pose-normalized face (no in-plane rotation/scale jitter across frames). See
docs/preprocessing.md.
"""
import cv2
import numpy as np

from preprocessing.ops.constants import (
    ARCFACE_TEMPLATE_112, FRAME_SIZE, MOUTH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


def face_template(size: int = FRAME_SIZE, inset: float = 0.0) -> np.ndarray:
    """The 5-point destination template at `size`, optionally inset for context.

    `inset` shrinks the ArcFace template toward the center (0.2 => the face fills
    ~80% of the crop, leaving a border of hairline/jaw context). It defaults to
    0 -- the canonical ArcFace framing -- because insetting asks the source frame
    for context further out than the face, and a frame that cannot supply it just
    yields padding: on FakeAVCeleb's pre-cropped 224x224 frames, inset=0.2 leaves
    25% of the crop empty vs 14% at inset=0. Raise it only for datasets whose
    frames are whole scenes. NOT the same knob as crop_and_resize's bbox margin.
    """
    t = ARCFACE_TEMPLATE_112 * (size / 112.0)
    if inset:
        c = size / 2.0
        t = (t - c) * (1.0 - inset) + c
    return t.astype(np.float32)


def _resize(img, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def detect(frame_rgb, detector, conf_thresh: float):
    """MTCNN detect. Returns (box, landmarks5, prob) or (None, None, None).

    Only the single most-confident face is considered (the detector is built
    with keep_all=False); it must clear conf_thresh.
    """
    boxes, probs, points = detector.detect(frame_rgb, landmarks=True)
    if (boxes is None or probs is None or probs[0] is None
            or float(probs[0]) < conf_thresh):
        return None, None, None
    return boxes[0], points[0], float(probs[0])


def align_face(frame_rgb, landmarks5, size: int = FRAME_SIZE,
               inset: float = 0.0):
    """Similarity-align a face onto the 5-point template. Returns size×size RGB.

    Uses a partial-affine (rotation+scale+translation, no shear) estimate from
    the detected landmarks to the template, then warps. Returns None if the
    transform can't be estimated (caller falls back to a plain crop).

    Whatever the canvas samples from beyond the frame edge is padded with BLACK,
    never synthesized. Reflecting it (the previous behaviour) mirrored the face
    back into the crop as an upside-down second face — on a dataset of tight face
    crops, where the aligned canvas always overshoots, that fabricated facial
    structure in every frame. Padding is honest and constant; the model can learn
    "no data" from it, not a phantom face. Same choice as ArcFace's norm_crop.
    """
    src = np.asarray(landmarks5, dtype=np.float32)
    dst = face_template(size, inset)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame_rgb, M, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def crop_and_resize(frame_rgb, box, size: int = FRAME_SIZE, margin: float = 0.2):
    """Margin-padded bbox crop, resized to size×size RGB. None if the crop is empty.

    The no-landmark fallback / alignment-off path. (Formerly
    crop_faces.crop_and_resize_face, which returned BGR; this returns RGB so
    callers no longer reconvert.)
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


def detect_align_crop(frame_rgb, detector, conf_thresh: float = 0.9,
                      margin: float = 0.2, align: bool = True,
                      align_inset: float = 0.0,
                      size: int = FRAME_SIZE, mouth_size: int = MOUTH_SIZE):
    """One detect call → (face size×size, mouth mouth_size×mouth_size, detected).

    The single face+mouth code path. When no face clears conf_thresh, both
    outputs fall back to a plain resize of the frame and detected=False, so the
    caller always gets fixed-shape crops (shape must stay fixed for batching).
    When align=True the face is 5-point aligned; align=False uses the bbox crop.

    The two context knobs are deliberately separate: `margin` pads the bbox crop
    (and can clamp to the frame), `align_inset` insets the alignment template
    (and cannot — it pads instead). Feeding one value to both is what made
    aligned crops a quarter empty.
    """
    box, landmarks5, _ = detect(frame_rgb, detector, conf_thresh)
    if box is None:
        return _resize(frame_rgb, size), _resize(frame_rgb, mouth_size), False

    face = align_face(frame_rgb, landmarks5, size, align_inset) if align else None
    if face is None:
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
