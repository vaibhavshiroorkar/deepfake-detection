import numpy as np
import pytest

from dfd.prep import face as F


def test_square_box_geometry():
    f = F.Face(x=100, y=50, w=40, h=60, score=0.9)
    x0, y0, side = F.square_box(f, 1.3)
    assert side == round(1.3 * 60)           # enlarged longer side
    assert x0 + side / 2 == pytest.approx(120, abs=1)  # centred on the face
    assert y0 + side / 2 == pytest.approx(80, abs=1)
    assert F.square_box(f, 1.0)[2] == 60     # factor is a parameter


def test_crop_square_content_and_size():
    img = np.zeros((200, 300, 3), np.uint8)
    img[80:120, 140:180] = (0, 255, 0)       # a 40x40 "face"
    f = F.Face(140, 80, 40, 40, 1.0)
    crop, face, sq = F.crop_largest_face(img, [f], enlarge=1.3, out_size=130)
    assert crop.shape == (130, 130, 3)
    assert sq[2] == 52
    green = (crop[:, :, 1] == 255) & (crop[:, :, 0] == 0)
    ys, xs = np.nonzero(green)
    # the face covers 40/52 of the crop side, centred: about 100 of 130 pixels
    assert 98 <= xs.max() - xs.min() + 1 <= 102 and 98 <= ys.max() - ys.min() + 1 <= 102
    assert abs((xs.min() + xs.max()) / 2 - 64.5) <= 2 and abs((ys.min() + ys.max()) / 2 - 64.5) <= 2


def test_crop_pads_with_black_outside_frame():
    img = np.full((100, 100, 3), 200, np.uint8)
    crop = F.crop_square(img, -20, -20, 60, out_size=60)  # square sticks out at top left
    assert crop.shape == (60, 60, 3)
    assert (crop[:20, :] == 0).all() and (crop[:, :20] == 0).all()
    assert (crop[20:, 20:] == 200).all()
    far = F.crop_square(img, 500, 500, 30, out_size=16)   # entirely outside: all black, no crash
    assert far.shape == (16, 16, 3) and far.max() == 0


def test_largest_face_is_chosen_and_no_face_is_none():
    small, big = F.Face(0, 0, 10, 10, 0.99), F.Face(50, 50, 30, 30, 0.7)
    assert F.largest([small, big]) is big
    img = np.zeros((100, 100, 3), np.uint8)
    assert F.crop_largest_face(img, []) == (None, None, None)


def test_unknown_detector_and_missing_model(tmp_path):
    with pytest.raises(ValueError):
        F.make_detector("nope")
    with pytest.raises(FileNotFoundError):
        F.make_detector("yunet", tmp_path)
