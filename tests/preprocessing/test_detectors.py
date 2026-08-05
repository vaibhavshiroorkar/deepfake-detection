"""The two face detectors and the contract they share.

The landmark-order test is the reason this file exists. MTCNN and YuNet describe
their five points with opposite naming conventions (YuNet's "right eye" is the
subject's right, which is the image-left point MTCNN calls the left eye), so the
documentation reads as though one of them needs reversing. Measured against real
frames, they already emit the same physical order. Reversing either would mirror
every face while changing no shape, which nothing downstream would catch, so the
ordering is pinned here rather than left to a comment.

Anything needing a real face skips when the dataset isn't extracted.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from preprocessing.ops import detectors as D

_DATA = Path(__file__).resolve().parents[2] / "data"


def _first_existing_clip():
    """Any real clip on disk, manifest first and a glob as the fallback.

    The fallback is not redundant. A manifest records video_path relative to a
    layout that may not be the one on this machine, and when it misses, a test
    that skips looks exactly like a test that passes. This one pins landmark
    ordering, so it has to run wherever clips exist at all.
    """
    for split in ("val", "train", "test"):
        csv = _DATA / f"{split}.csv"
        if not csv.exists():
            continue
        for _, row in pd.read_csv(csv).iterrows():
            vp = _DATA / row["video_path"]
            if vp.exists():
                return vp
    return next(iter(sorted(_DATA.rglob("*.mp4"))), None)


@pytest.fixture(scope="module")
def face_frames():
    """A few RGB frames from a real clip, or skip."""
    vp = _first_existing_clip()
    if vp is None:
        pytest.skip("No extracted dataset clip available.")
    cap = cv2.VideoCapture(str(vp))
    frames = []
    for ms in (400, 800, 1200):
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        pytest.skip(f"Could not decode frames from {vp}.")
    return frames


def test_build_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown detector"):
        D.build("retinaface")


def test_yunet_weights_ship_with_the_repo():
    assert D.YUNET_WEIGHTS.exists(), (
        f"{D.YUNET_WEIGHTS} is missing; YuNet cannot load without it.")


@pytest.mark.parametrize("name", D.DETECTOR_NAMES)
def test_detector_returns_the_shared_shape(name, face_frames):
    det = D.build(name, device="cpu")
    assert det.name == name
    box, landmarks5, prob = det.detect(face_frames[0])
    assert box is not None, f"{name} found no face in a frame that has one"
    assert box.shape == (4,)
    assert box[0] < box[2] and box[1] < box[3]        # x1y1x2y2, not xywh
    assert landmarks5.shape == (5, 2)
    assert 0.0 <= prob <= 1.0


def test_no_face_reads_as_no_detection():
    """Flat noise has no face. Both detectors must say so rather than guess."""
    rng = np.random.default_rng(0)
    blank = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    for name in D.DETECTOR_NAMES:
        box, landmarks5, prob = D.build(name, device="cpu").detect(blank)
        if box is None:
            assert landmarks5 is None and prob is None
        else:
            # A detector may return a low-confidence box; faces.detect gates it.
            assert prob < 0.9, f"{name} claims {prob:.2f} confidence on noise"


def test_both_detectors_agree_on_landmark_order(face_frames):
    """Point i means the same facial feature in both. See the module docstring."""
    mtcnn = D.build(D.MTCNN_NAME, device="cpu")
    yunet = D.build(D.YUNET_NAME, device="cpu")

    compared = 0
    for frame in face_frames:
        _, m_pts, m_prob = mtcnn.detect(frame)
        _, y_pts, y_prob = yunet.detect(frame)
        if m_pts is None or y_pts is None or m_prob < 0.9 or y_prob < 0.9:
            continue
        compared += 1

        same = float(np.linalg.norm(m_pts - y_pts, axis=1).sum())
        # Eyes swapped and mouth corners swapped: what the naming would imply.
        mirrored = y_pts[[1, 0, 2, 4, 3]]
        swapped = float(np.linalg.norm(m_pts - mirrored, axis=1).sum())
        assert same < swapped, (
            f"YuNet's landmark order no longer matches MTCNN's: as-is total "
            f"distance {same:.1f}, swapped {swapped:.1f}. If the weights changed, "
            f"reorder in YuNetDetector.detect rather than downstream.")

    if compared == 0:
        pytest.skip("No frame had a confident detection from both detectors.")
