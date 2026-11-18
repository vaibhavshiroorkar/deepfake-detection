import numpy as np

from deepfake_detection.avsync.mouth import locate_region


def test_region_is_centred_on_the_face_and_scaled():
    x0, y0, side = locate_region(np.array([900, 500, 100, 120]), 1920, 1080)
    assert side == 264  # 2.2 x the longer side
    assert (x0 + side / 2, y0 + side / 2) == (950, 560)  # centre of the box


def test_region_stays_inside_the_frame():
    x0, y0, side = locate_region(np.array([5, 1000, 100, 100]), 1920, 1080)
    assert x0 == 0 and y0 + side <= 1080


def test_region_never_exceeds_the_frame():
    x0, y0, side = locate_region(np.array([0, 0, 800, 800]), 1080, 1920)
    assert side == 1080 and x0 == 0
