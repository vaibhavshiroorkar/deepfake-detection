"""Face detection and square crop (data preparation only).

The detectors here are preprocessing tools, not the project's detection model
(D-023). Three CPU candidates that run through OpenCV only (no torch, no
onnxruntime) are provided so that they can be compared on the same frames:

    yunet      cv2.FaceDetectorYN with face_detection_yunet_2023mar.onnx (opencv_zoo)
    ultraface  Ultra-Light-Fast-Generic-Face-Detector-1MB, version-RFB-320.onnx, via cv2.dnn
    scrfd      InsightFace SCRFD 500M (det_500m.onnx from buffalo_sc), via cv2.dnn
               (InsightFace weights are for non-commercial research use)

Every detector returns all faces it finds as `Face(x, y, w, h, score)` in the
pixel coordinates of the frame it was given, sorted by decreasing area.
The crop rule (`crop_square`) is independent of the detector.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

DETECTORS = ("yunet", "ultraface", "scrfd")
DEFAULT_MODEL_DIR = Path("data/_prep_bench/models")
MODEL_FILES = {
    "yunet": "yunet_2023mar.onnx",
    "ultraface": "ultraface_rfb320.onnx",
    "scrfd": "det_500m.onnx",
}
DEFAULT_CROP_SIZE = 224
DEFAULT_ENLARGE = 1.3  # FF++ paper value (R-001); a parameter, not a decision
DEFAULT_JPEG_QUALITY = 95


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return self.w * self.h


def largest(faces: list[Face]) -> Face | None:
    """The face with the largest box area (ties: higher score)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.area, f.score))


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    if len(boxes_xyxy) == 0:
        return []
    wh = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                   boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], 1)
    keep = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), 0.0, thr)
    return [int(k) for k in np.array(keep).reshape(-1)]


def _fit(frame: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Downscale so that the long side is at most max_side. Returns (image, scale)."""
    if not max_side:
        return frame, 1.0
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return frame, 1.0
    return cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


class YuNet:
    name = "yunet"

    def __init__(self, model: str | Path, score_thr: float = 0.6, nms_thr: float = 0.3,
                 max_side: int | None = 640):
        self.max_side = max_side
        self._det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_thr, nms_thr, 5000)

    def detect(self, frame: np.ndarray) -> list[Face]:
        img, s = _fit(frame, self.max_side)
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, res = self._det.detect(img)
        if res is None:
            return []
        faces = [Face(float(r[0]) / s, float(r[1]) / s, float(r[2]) / s, float(r[3]) / s, float(r[14])) for r in res]
        return sorted(faces, key=lambda f: -f.area)


class UltraFace:
    name = "ultraface"
    W, H = 320, 240

    def __init__(self, model: str | Path, score_thr: float = 0.7, nms_thr: float = 0.3):
        self.net = cv2.dnn.readNet(str(model))
        self.score_thr, self.nms_thr = score_thr, nms_thr
        self.names = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 128, (self.W, self.H), (127, 127, 127), swapRB=True)
        self.net.setInput(blob)
        scores, boxes = self.net.forward(self.names)
        scores, boxes = scores[0, :, 1], boxes[0]  # boxes: normalised x1,y1,x2,y2
        m = scores > self.score_thr
        if not m.any():
            return []
        b = boxes[m] * np.array([w, h, w, h], np.float32)
        sc = scores[m]
        keep = _nms(b, sc, self.nms_thr)
        faces = [Face(float(b[k, 0]), float(b[k, 1]), float(b[k, 2] - b[k, 0]),
                      float(b[k, 3] - b[k, 1]), float(sc[k])) for k in keep]
        return sorted(faces, key=lambda f: -f.area)


class SCRFD:
    name = "scrfd"
    SIZE = 640
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, model: str | Path, score_thr: float = 0.5, nms_thr: float = 0.4):
        self.net = cv2.dnn.readNet(str(model))
        self.score_thr, self.nms_thr = score_thr, nms_thr
        self.names = self.net.getUnconnectedOutLayersNames()
        self._anchors: dict[int, np.ndarray] = {}

    def _centers(self, stride: int) -> np.ndarray:
        if stride not in self._anchors:
            n = self.SIZE // stride
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._anchors[stride] = np.repeat(c, self.NUM_ANCHORS, axis=0)
        return self._anchors[stride]

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        s = self.SIZE / max(h, w)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = np.zeros((self.SIZE, self.SIZE, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (self.SIZE, self.SIZE), (127.5,) * 3, swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.names)
        # outputs come as 3 score maps, 3 box maps, 3 landmark maps (shape tells which)
        by_stride: dict[int, dict[int, np.ndarray]] = {}
        for o in outs:
            for st in self.STRIDES:
                if o.shape[0] == (self.SIZE // st) ** 2 * self.NUM_ANCHORS:
                    by_stride.setdefault(st, {})[o.shape[1]] = o
        boxes, scores = [], []
        for st in self.STRIDES:
            sc = by_stride[st][1][:, 0]
            m = sc > self.score_thr
            if not m.any():
                continue
            d = by_stride[st][4][m] * st
            c = self._centers(st)[m]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
            scores.append(sc[m])
        if not boxes:
            return []
        b, sc = np.concatenate(boxes) / s, np.concatenate(scores)
        keep = _nms(b, sc, self.nms_thr)
        faces = [Face(float(b[k, 0]), float(b[k, 1]), float(b[k, 2] - b[k, 0]),
                      float(b[k, 3] - b[k, 1]), float(sc[k])) for k in keep]
        return sorted(faces, key=lambda f: -f.area)


def make_detector(name: str, model_dir: str | Path = DEFAULT_MODEL_DIR, **kw):
    """Build a detector by name. The ONNX file is looked up in model_dir."""
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}, choose from {DETECTORS}")
    path = Path(model_dir) / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"model file missing: {path} (see docs/_agents/S2C.md for the source)")
    return {"yunet": YuNet, "ultraface": UltraFace, "scrfd": SCRFD}[name](path, **kw)


def square_box(face: Face, enlarge: float) -> tuple[int, int, int]:
    """Square around the face centre with side = enlarge * max(w, h).

    Returns integer (x0, y0, side). The square is NOT clipped to the frame; the
    crop pads with black where it leaves the frame, so the face stays centred
    and the geometry does not depend on how close the face is to the border.
    """
    cx, cy = face.x + face.w / 2.0, face.y + face.h / 2.0
    side = max(2, int(round(enlarge * max(face.w, face.h))))
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def crop_square(frame: np.ndarray, x0: int, y0: int, side: int, out_size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Cut the square [x0, x0+side) x [y0, y0+side) (black outside the frame) and resize to out_size."""
    h, w = frame.shape[:2]
    patch = np.zeros((side, side, 3), frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_LINEAR
    return cv2.resize(patch, (out_size, out_size), interpolation=interp)


def crop_largest_face(frame: np.ndarray, faces: list[Face], enlarge: float = DEFAULT_ENLARGE,
                      out_size: int = DEFAULT_CROP_SIZE):
    """(crop or None, face or None, square (x0, y0, side) or None) for the largest face."""
    f = largest(faces)
    if f is None:
        return None, None, None
    x0, y0, side = square_box(f, enlarge)
    return crop_square(frame, x0, y0, side, out_size), f, (x0, y0, side)
