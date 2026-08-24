from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from deepfake_detection.views.face_detector import MTCNNFaceDetector, YuNetFaceDetector
from deepfake_detection.views.tracking import Box, Detection, Landmarks5, Point


class FixtureMTCNN:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.landmarks_requested = False

    def detect(
        self, image: np.ndarray, *, landmarks: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.image = image.copy()
        self.landmarks_requested = landmarks
        boxes = np.asarray([[1, 2, 10, 12], [20, 20, 30, 30]], dtype=np.float32)
        probabilities = np.asarray([0.95, 0.40], dtype=np.float32)
        points = np.asarray(
            [
                [[8, 4], [3, 4], [5, 6], [8, 9], [2, 9]],
                [[28, 22], [23, 22], [25, 25], [28, 28], [22, 28]],
            ],
            dtype=np.float32,
        )
        return boxes, probabilities, points


class FixtureYuNet:
    def __init__(self, rows: np.ndarray | None) -> None:
        self.rows = rows
        self.input_sizes: list[tuple[int, int]] = []
        self.images: list[np.ndarray] = []

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        self.input_sizes.append(size)

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        self.images.append(image.copy())
        return 1, self.rows


def test_mtcnn_adapter_converts_bgr_filters_scores_and_canonicalizes_points() -> None:
    model = FixtureMTCNN()
    detector = MTCNNFaceDetector(model=model, confidence=0.80)
    blue_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255

    detections = detector.detect(blue_bgr)

    assert model.landmarks_requested is True
    assert model.image is not None
    assert model.image[0, 0].tolist() == [0, 0, 255]
    assert len(detections) == 1
    assert detections[0].box == Box(1.0, 2.0, 10.0, 12.0)
    assert detections[0].confidence == pytest.approx(0.95)
    assert detections[0].landmarks == Landmarks5(
        eye_left=Point(3.0, 4.0),
        eye_right=Point(8.0, 4.0),
        nose=Point(5.0, 6.0),
        mouth_left=Point(2.0, 9.0),
        mouth_right=Point(8.0, 9.0),
    )


@pytest.mark.parametrize(
    "result",
    [
        (None, None, None),
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
        ),
    ],
)
def test_mtcnn_adapter_accepts_empty_outputs(result: tuple[Any, Any, Any]) -> None:
    class EmptyMTCNN:
        def detect(self, image: np.ndarray, *, landmarks: bool) -> tuple[Any, Any, Any]:
            return result

    assert (
        MTCNNFaceDetector(model=EmptyMTCNN()).detect(
            np.zeros((8, 8, 3), dtype=np.uint8)
        )
        == ()
    )


@pytest.mark.parametrize(
    ("boxes", "probabilities", "points"),
    [
        (
            np.asarray([[1, 2, 3]], dtype=np.float32),
            np.asarray([0.9]),
            np.zeros((1, 5, 2)),
        ),
        (
            np.asarray([[1, 2, 3, 4]], dtype=np.float32),
            np.asarray([0.9, 0.8]),
            np.zeros((1, 5, 2)),
        ),
        (
            np.asarray([[1, 2, 3, 4]], dtype=np.float32),
            np.asarray([0.9]),
            np.zeros((1, 4, 2)),
        ),
    ],
)
def test_mtcnn_adapter_rejects_malformed_provider_arrays(
    boxes: np.ndarray, probabilities: np.ndarray, points: np.ndarray
) -> None:
    class MalformedMTCNN:
        def detect(
            self, image: np.ndarray, *, landmarks: bool
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return boxes, probabilities, points

    with pytest.raises(ValueError, match="MTCNN"):
        MTCNNFaceDetector(model=MalformedMTCNN()).detect(
            np.zeros((8, 8, 3), dtype=np.uint8)
        )


def test_yunet_adapter_parses_rows_filters_scores_and_uses_each_frame_size() -> None:
    rows = np.asarray(
        [
            [1, 2, 10, 20, 8, 4, 3, 4, 5, 6, 8, 9, 2, 9, 0.95],
            [20, 20, 5, 6, 24, 22, 21, 22, 23, 23, 24, 25, 21, 25, 0.40],
        ],
        dtype=np.float32,
    )
    model = FixtureYuNet(rows)
    detector = YuNetFaceDetector(model=model, confidence=0.80)
    blue_bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255

    detections = detector.detect(blue_bgr)
    detector.detect(np.zeros((12, 18, 3), dtype=np.uint8))

    assert model.input_sizes == [(32, 24), (18, 12)]
    assert model.images[0][0, 0].tolist() == [255, 0, 0]
    assert len(detections) == 1
    assert detections[0].box == Box(1.0, 2.0, 11.0, 22.0)
    assert detections[0].confidence == pytest.approx(0.95)
    assert detections[0].landmarks == Landmarks5(
        eye_left=Point(3.0, 4.0),
        eye_right=Point(8.0, 4.0),
        nose=Point(5.0, 6.0),
        mouth_left=Point(2.0, 9.0),
        mouth_right=Point(8.0, 9.0),
    )


@pytest.mark.parametrize(
    "rows",
    [None, np.empty((0, 15), dtype=np.float32)],
)
def test_yunet_adapter_accepts_empty_outputs(rows: np.ndarray | None) -> None:
    detector = YuNetFaceDetector(model=FixtureYuNet(rows))

    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == ()


@pytest.mark.parametrize(
    "rows",
    [
        np.empty((15,), dtype=np.float32),
        np.empty((1, 14), dtype=np.float32),
        np.empty((1, 16), dtype=np.float32),
    ],
)
def test_yunet_adapter_rejects_malformed_rows(rows: np.ndarray) -> None:
    detector = YuNetFaceDetector(model=FixtureYuNet(rows))

    with pytest.raises(ValueError, match="15 values"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_geometry_contract_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Box(value, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Point(0.0, value)
    with pytest.raises(ValueError, match="finite"):
        Detection(Box(0.0, 0.0, 1.0, 1.0), value)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_contract_rejects_scores_outside_probability_range(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Detection(Box(0.0, 0.0, 1.0, 1.0), confidence)


def test_landmark_contract_canonicalizes_provider_pair_order() -> None:
    landmarks = Landmarks5(
        eye_left=Point(9.0, 2.0),
        eye_right=Point(3.0, 1.0),
        nose=Point(6.0, 5.0),
        mouth_left=Point(8.0, 9.0),
        mouth_right=Point(4.0, 8.0),
    )

    assert landmarks.eye_left == Point(3.0, 1.0)
    assert landmarks.eye_right == Point(9.0, 2.0)
    assert landmarks.mouth_left == Point(4.0, 8.0)
    assert landmarks.mouth_right == Point(8.0, 9.0)
