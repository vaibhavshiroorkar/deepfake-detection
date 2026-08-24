from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .tracking import Box, Detection, Landmarks5, Point


def _validate_confidence(confidence: float) -> None:
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Detection confidence must be finite and in [0, 1]")


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Face detector expects a BGR image with three channels")


def _landmarks_from_points(points: np.ndarray) -> Landmarks5:
    return Landmarks5(
        eye_left=Point(float(points[0, 0]), float(points[0, 1])),
        eye_right=Point(float(points[1, 0]), float(points[1, 1])),
        nose=Point(float(points[2, 0]), float(points[2, 1])),
        mouth_left=Point(float(points[3, 0]), float(points[3, 1])),
        mouth_right=Point(float(points[4, 0]), float(points[4, 1])),
    )


def _sort_and_filter(
    detections: list[Detection], confidence: float
) -> tuple[Detection, ...]:
    accepted = [
        detection for detection in detections if detection.confidence >= confidence
    ]
    return tuple(
        sorted(accepted, key=lambda detection: detection.confidence, reverse=True)
    )


class MTCNNFaceDetector:
    def __init__(
        self,
        *,
        model: Any | None = None,
        confidence: float = 0.80,
        device: str = "cpu",
    ) -> None:
        _validate_confidence(confidence)
        if model is None:
            try:
                from facenet_pytorch import MTCNN
            except ImportError as error:
                raise RuntimeError(
                    "MTCNN requires the media dependency group"
                ) from error
            model = MTCNN(keep_all=True, device=device)
        self.model = model
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        _validate_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.detect(rgb, landmarks=True)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("MTCNN must return boxes, probabilities, and landmarks")
        boxes, probabilities, points = result
        if boxes is None and probabilities is None and points is None:
            return ()
        if boxes is None or probabilities is None or points is None:
            raise ValueError("MTCNN returned a partial detection result")

        box_rows = np.asarray(boxes)
        probability_rows = np.asarray(probabilities, dtype=object)
        point_rows = np.asarray(points)
        if box_rows.ndim != 2 or box_rows.shape[1] != 4:
            raise ValueError("MTCNN boxes must have shape (N, 4)")
        if probability_rows.ndim != 1:
            raise ValueError("MTCNN probabilities must have shape (N,)")
        if point_rows.ndim != 3 or point_rows.shape[1:] != (5, 2):
            raise ValueError("MTCNN landmarks must have shape (N, 5, 2)")
        if not (len(box_rows) == len(probability_rows) == len(point_rows)):
            raise ValueError("MTCNN detection arrays must have equal lengths")

        detections: list[Detection] = []
        for box, probability, landmark_points in zip(
            box_rows, probability_rows, point_rows, strict=True
        ):
            if probability is None:
                continue
            detections.append(
                Detection(
                    box=Box(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    confidence=float(probability),
                    landmarks=_landmarks_from_points(landmark_points),
                )
            )
        return _sort_and_filter(detections, self.confidence)


class YuNetFaceDetector:
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_path: Path | str | None = None,
        confidence: float = 0.80,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        _validate_confidence(confidence)
        if not 0 <= nms_threshold <= 1:
            raise ValueError("YuNet NMS threshold must be in [0, 1]")
        if top_k <= 0:
            raise ValueError("YuNet top_k must be positive")
        if model is None:
            if model_path is None:
                raise ValueError("YuNet requires a model path")
            resolved_path = Path(model_path).expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"YuNet model does not exist: {resolved_path}")
            model = cv2.FaceDetectorYN.create(
                str(resolved_path),
                "",
                (320, 320),
                confidence,
                nms_threshold,
                top_k,
            )
        self.model = model
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        _validate_frame(frame)
        height, width = frame.shape[:2]
        self.model.setInputSize((width, height))
        result = self.model.detect(frame)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("YuNet must return a status and detection rows")
        rows = result[1]
        if rows is None:
            return ()
        row_array = np.asarray(rows)
        if row_array.ndim != 2 or row_array.shape[1] != 15:
            raise ValueError("Each YuNet result row must contain 15 values")

        detections = []
        for row in row_array:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            landmark_points = np.asarray(row[4:14]).reshape(5, 2)
            detections.append(
                Detection(
                    box=Box(x, y, x + box_width, y + box_height),
                    confidence=float(row[14]),
                    landmarks=_landmarks_from_points(landmark_points),
                )
            )
        return _sort_and_filter(detections, self.confidence)
