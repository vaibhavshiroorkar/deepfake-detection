from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .tracking import Box, Detection


class MTCNNFaceDetector:
    def __init__(
        self,
        *,
        model: Any | None = None,
        confidence: float = 0.80,
        device: str = "cpu",
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be in [0, 1]")
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
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Face detector expects a BGR image with three channels")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probabilities = self.model.detect(rgb)
        if boxes is None or probabilities is None:
            return ()
        detections = [
            Detection(
                Box(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                float(probability),
            )
            for box, probability in zip(boxes, probabilities, strict=True)
            if probability is not None and float(probability) >= self.confidence
        ]
        return tuple(
            sorted(detections, key=lambda detection: detection.confidence, reverse=True)
        )
