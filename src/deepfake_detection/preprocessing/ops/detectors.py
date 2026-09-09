"""The face detectors, behind one small array interface.

Two are available and either can drive the whole pipeline:

  mtcnn  facenet-pytorch's MTCNN, the default.
  yunet  OpenCV's FaceDetectorYN, from the vendored ONNX weights.

Both answer the same call, `detect(frame_rgb) -> (box, landmarks5, prob)`, with
`(None, None, None)` for "no face here". `box` is x1, y1, x2, y2 in frame pixels
and `landmarks5` is a (5, 2) array.

No thresholding happens here. Both detectors report a confidence and
`faces.detect` is the single place that decides whether it is good enough, so one
`conf_thresh` means the same thing whichever detector is loaded.

Landmark order is left eye, right eye, nose, mouth-left, mouth-right, image-left
first. YuNet's own documentation calls its first point the right eye, but that is
the subject's right, which is the image-left point facenet-pytorch calls the left
eye. Same physical order, opposite naming convention.

These classes are adapters. The detection work itself lives in
`deepfake_detection.views.face_detector`, which is what training and evaluation
use, so the dashboard and the pipeline cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from deepfake_detection.views.face_detector import (
    MTCNNFaceDetector,
    YuNetFaceDetector,
)
from deepfake_detection.views.tracking import Detection

MTCNN_NAME = "mtcnn"
YUNET_NAME = "yunet"
DETECTOR_NAMES = (MTCNN_NAME, YUNET_NAME)
DEFAULT_DETECTOR = MTCNN_NAME

YUNET_WEIGHTS = Path("models/face_detection_yunet_2026may.onnx")

# Let the backends return nearly everything and gate it in faces.detect, the same
# way MTCNN's probability is gated. A backend threshold would otherwise be a
# second, invisible threshold that the dashboard slider could not reach.
_SCORE_FLOOR = 0.05


def _unpack(detections: tuple[Detection, ...]):
    """Most-confident detection -> (box, landmarks5, prob) arrays."""
    if not detections:
        return None, None, None
    best = detections[0]
    if best.landmarks is None:
        return None, None, None
    box = np.array(
        [best.box.left, best.box.top, best.box.right, best.box.bottom],
        dtype=np.float32,
    )
    marks = best.landmarks
    landmarks5 = np.array(
        [
            (marks.eye_left.x, marks.eye_left.y),
            (marks.eye_right.x, marks.eye_right.y),
            (marks.nose.x, marks.nose.y),
            (marks.mouth_left.x, marks.mouth_left.y),
            (marks.mouth_right.x, marks.mouth_right.y),
        ],
        dtype=np.float32,
    )
    return box, landmarks5, float(best.confidence)


class MTCNNDetector:
    """facenet-pytorch MTCNN, most-confident face only.

    `device` is "cpu" or "cuda". Pre-caching pins it to CPU so worker processes
    do not contend over the one GPU.
    """

    name = MTCNN_NAME

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._backend = MTCNNFaceDetector(confidence=_SCORE_FLOOR, device=device)

    def detect(self, frame_rgb: np.ndarray):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return _unpack(self._backend.detect(bgr))


class YuNetDetector:
    """OpenCV FaceDetectorYN, most-confident face only.

    CPU-only, so it ignores `device`. That is the point of it next to MTCNN: it
    costs nothing on the GPU and runs an order of magnitude faster per frame.
    """

    name = YUNET_NAME

    def __init__(
        self, weights: Path = YUNET_WEIGHTS, device: str | None = None
    ) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"YuNet weights not found at {weights}. "
                f"Fetch them with: uv run ddf detector fetch-yunet"
            )
        self.device = "cpu"
        self._backend = YuNetFaceDetector(model_path=weights, confidence=_SCORE_FLOOR)

    def detect(self, frame_rgb: np.ndarray):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return _unpack(self._backend.detect(bgr))


def build(name: str = DEFAULT_DETECTOR, device: str = "cpu"):
    """Construct a detector by name. Loading weights is slow: build once, reuse."""
    if name == MTCNN_NAME:
        return MTCNNDetector(device=device)
    if name == YUNET_NAME:
        return YuNetDetector(device=device)
    raise ValueError(
        f"Unknown detector {name!r}. Available: {', '.join(DETECTOR_NAMES)}"
    )
