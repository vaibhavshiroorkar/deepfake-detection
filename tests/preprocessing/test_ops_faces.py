"""MAIN visual ops: detection, crop, mouth ROI, normalize.

No MTCNN is loaded: crop/mouth take explicit landmarks, and detect_crop is
driven by a duck-typed fake detector.

Five-point alignment used to be tested here. It was removed from the pipeline and
is parked in docs/ideas.md; the tests went with it, and the commit that deleted
them is the place to look if it comes back.
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


def test_crop_and_resize_shape_rgb(img):
    box = (50, 40, 200, 220)
    out = F.crop_and_resize(img, box, FRAME_SIZE, margin=0.2)
    assert out.shape == (FRAME_SIZE, FRAME_SIZE, 3) and out.dtype == np.uint8


def test_mouth_roi_is_96(img):
    landmarks = np.array([[100, 100], [160, 100], [130, 130],
                          [110, 165], [150, 165]], np.float32)
    out = F.mouth_roi(img, landmarks, MOUTH_SIZE)
    assert out.shape == (MOUTH_SIZE, MOUTH_SIZE, 3)


def test_detect_crop_detected(img):
    landmarks = np.array([[110, 110], [190, 110], [150, 150],
                          [120, 190], [180, 190]], np.float32)
    det = FakeDetector(box=(90, 90, 210, 215), points=landmarks, prob=0.99)
    face, mouth, detected = F.detect_crop(img, det, conf_thresh=0.9)
    assert detected is True
    assert face.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert mouth.shape == (MOUTH_SIZE, MOUTH_SIZE, 3)


def test_detect_crop_below_threshold_falls_back(img):
    det = FakeDetector(box=(90, 90, 210, 215),
                       points=np.zeros((5, 2), np.float32), prob=0.10)
    face, mouth, detected = F.detect_crop(img, det, conf_thresh=0.9)
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
