"""The face detectors, behind one interface.

Two are available and either can drive the whole pipeline:

  mtcnn  facenet-pytorch's MTCNN, the original and still the default.
  yunet  OpenCV's FaceDetectorYN, from vendored ONNX weights.

Both answer the same call, `detect(frame_rgb) -> (box, landmarks5, prob)`, with
`(None, None, None)` for "no face here". `box` is x1,y1,x2,y2 in frame pixels and
`landmarks5` is a (5,2) array. No thresholding happens in here: both detectors
report a confidence and `faces.detect` is the single place that decides whether
it is good enough, so one `conf_thresh` means the same thing whichever is loaded.

Landmark order is left eye, right eye, nose, mouth-left, mouth-right, IMAGE-left
first, and both detectors already agree on it. YuNet's own documentation calls
its first point the right eye, but that is the subject's right, which is the
image-left point that facenet-pytorch calls the left eye. Same physical order,
opposite naming convention. Checked against real frames rather than the docs, and
test_detectors.py pins it, because getting this wrong mirrors every face without
changing a single shape.
"""
from pathlib import Path

import numpy as np

MTCNN_NAME = "mtcnn"
YUNET_NAME = "yunet"
DETECTOR_NAMES = (MTCNN_NAME, YUNET_NAME)
DEFAULT_DETECTOR = MTCNN_NAME

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
YUNET_WEIGHTS = _REPO_ROOT / "checkpoints" / "yunet" / "face_detection_yunet_2023mar.onnx"

# Let YuNet return nearly everything and gate it in faces.detect, the same way
# MTCNN's probability is gated. YuNet's own score_threshold would otherwise be a
# second, invisible threshold that the dashboard slider could not reach.
_YUNET_SCORE_FLOOR = 0.05


class MTCNNDetector:
    """facenet-pytorch MTCNN, most-confident face only.

    `device` is "cpu" or "cuda"; pre-caching pins it to CPU so worker processes
    do not contend over the one GPU.
    """

    name = MTCNN_NAME

    def __init__(self, device: str = "cpu"):
        from facenet_pytorch import MTCNN

        self.device = device
        self._model = MTCNN(keep_all=False, device=device)

    def detect(self, frame_rgb):
        boxes, probs, points = self._model.detect(frame_rgb, landmarks=True)
        if boxes is None or probs is None or probs[0] is None:
            return None, None, None
        return (np.asarray(boxes[0], dtype=np.float32),
                np.asarray(points[0], dtype=np.float32),
                float(probs[0]))


class YuNetDetector:
    """OpenCV FaceDetectorYN, most-confident face only.

    CPU-only, so it ignores `device`. That is the point of it next to MTCNN: it
    costs nothing on the GPU and runs an order of magnitude faster per frame.

    The model is built once and re-pointed with setInputSize whenever the frame
    shape changes, because YuNet bakes the input size into its anchor grid and
    silently returns nonsense boxes if it is fed a different size than it was
    told about.
    """

    name = YUNET_NAME

    def __init__(self, weights: Path = YUNET_WEIGHTS, device: str | None = None):
        import cv2

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"YuNet weights not found at {weights}. They ship with the repo; "
                "if this is a partial checkout, fetch face_detection_yunet_2023mar.onnx "
                "from https://github.com/opencv/opencv_zoo (models/face_detection_yunet).")
        self.device = "cpu"
        self._input_size = (0, 0)
        self._model = cv2.FaceDetectorYN.create(
            str(weights), "", self._input_size, score_threshold=_YUNET_SCORE_FLOOR)

    def detect(self, frame_rgb):
        import cv2

        h, w = frame_rgb.shape[:2]
        if (w, h) != self._input_size:
            self._model.setInputSize((w, h))
            self._input_size = (w, h)

        # YuNet reads BGR. Rows are [x, y, w, h, 10 landmark coords, score].
        _, faces = self._model.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None, None, None

        row = faces[int(np.argmax(faces[:, -1]))]
        x, y, bw, bh = (float(v) for v in row[:4])
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        landmarks5 = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        return box, landmarks5, float(row[-1])


def build(name: str = DEFAULT_DETECTOR, device: str = "cpu"):
    """Construct a detector by name. Loading weights is slow: build once, reuse."""
    if name == MTCNN_NAME:
        return MTCNNDetector(device=device)
    if name == YUNET_NAME:
        return YuNetDetector(device=device)
    raise ValueError(f"Unknown detector {name!r}. Available: {', '.join(DETECTOR_NAMES)}")
