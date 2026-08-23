import numpy as np
import pytest

from deepfake_detection.views.face_detector import MTCNNFaceDetector


class FixtureMTCNN:
    def detect(self, image: np.ndarray):
        boxes = np.asarray([[1, 2, 10, 12], [20, 20, 30, 30]], dtype=np.float32)
        probabilities = np.asarray([0.95, 0.40], dtype=np.float32)
        return boxes, probabilities


def test_mtcnn_adapter_converts_bgr_and_filters_low_confidence_faces() -> None:
    detector = MTCNNFaceDetector(model=FixtureMTCNN(), confidence=0.80)
    blue_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255

    detections = detector.detect(blue_bgr)

    assert len(detections) == 1
    assert detections[0].box.left == 1.0
    assert detections[0].confidence == pytest.approx(0.95)
