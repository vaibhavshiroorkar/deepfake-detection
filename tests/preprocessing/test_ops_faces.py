"""MAIN visual ops: detection, 5-point alignment, crop, mouth ROI, normalize.

Alignment is tested without loading MTCNN — align/crop/mouth take explicit
landmarks, and detect_align_crop is driven by a duck-typed fake detector.
"""
import cv2
import numpy as np
import pytest

from preprocessing.ops import faces as F
from preprocessing.ops.constants import FRAME_SIZE, MOUTH_SIZE


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)


class FakeDetector:
    """Stands in for MTCNN: returns canned box/prob/landmarks from .detect()."""
    def __init__(self, box, points, prob):
        self.box, self.points, self.prob = box, points, prob

    def detect(self, frame_rgb, landmarks=False):
        boxes = np.array([self.box], dtype=np.float32)
        probs = np.array([self.prob], dtype=np.float32)
        if landmarks:
            return boxes, probs, np.array([self.points], dtype=np.float32)
        return boxes, probs


def test_face_template_shape_and_inset():
    t = F.face_template(FRAME_SIZE, margin=0.2)
    assert t.shape == (5, 2)
    assert t.min() > 0 and t.max() < FRAME_SIZE      # inset, inside the canvas


def test_align_identity_when_landmarks_equal_template(img):
    """src == dst template ⇒ near-identity warp ⇒ output ≈ input."""
    src = F.face_template(FRAME_SIZE, margin=0.2)
    base = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
    out = F.align_face(base, src, FRAME_SIZE, margin=0.2)
    assert out.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert np.mean(np.abs(out.astype(int) - base.astype(int))) < 2.0


def test_align_moves_landmarks_onto_template():
    """A pure rotation+scale of the template must align back to the template."""
    size, margin = FRAME_SIZE, 0.2
    template = F.face_template(size, margin)
    # Put the template near the center of a larger canvas, then rotate it.
    canvas = 400
    centered = template + (canvas - size) / 2.0
    R = cv2.getRotationMatrix2D((canvas / 2.0, canvas / 2.0), 25.0, 1.1)
    src = (R[:, :2] @ centered.T).T + R[:, 2]

    frame = np.zeros((canvas, canvas, 3), np.uint8)
    for (x, y) in src:                                # white marker at each src point
        cv2.rectangle(frame, (int(x) - 4, int(y) - 4), (int(x) + 4, int(y) + 4),
                      (255, 255, 255), -1)

    aligned = F.align_face(frame, src, size, margin)
    for (x, y) in template:                           # markers must land on template
        patch = aligned[int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5]
        assert patch.max() > 200


def test_crop_and_resize_shape_rgb(img):
    box = (50, 40, 200, 220)
    out = F.crop_and_resize(img, box, FRAME_SIZE, margin=0.2)
    assert out.shape == (FRAME_SIZE, FRAME_SIZE, 3) and out.dtype == np.uint8


def test_mouth_roi_is_96(img):
    landmarks = np.array([[100, 100], [160, 100], [130, 130],
                          [110, 165], [150, 165]], np.float32)
    out = F.mouth_roi(img, landmarks, MOUTH_SIZE)
    assert out.shape == (MOUTH_SIZE, MOUTH_SIZE, 3)


def test_detect_align_crop_detected(img):
    landmarks = np.array([[110, 110], [190, 110], [150, 150],
                          [120, 190], [180, 190]], np.float32)
    det = FakeDetector(box=(90, 90, 210, 215), points=landmarks, prob=0.99)
    face, mouth, detected = F.detect_align_crop(img, det, conf_thresh=0.9)
    assert detected is True
    assert face.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert mouth.shape == (MOUTH_SIZE, MOUTH_SIZE, 3)


def test_detect_align_crop_below_threshold_falls_back(img):
    det = FakeDetector(box=(90, 90, 210, 215),
                       points=np.zeros((5, 2), np.float32), prob=0.10)
    face, mouth, detected = F.detect_align_crop(img, det, conf_thresh=0.9)
    assert detected is False                          # plain resize fallback
    assert face.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert mouth.shape == (MOUTH_SIZE, MOUTH_SIZE, 3)


def test_imagenet_normalize_is_zero_centered():
    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, size=(FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    arr = F.imagenet_normalize(im)
    assert arr.dtype == np.float32
    lo, hi = F.normalized_range(arr)
    assert lo < 0 and hi > 0
