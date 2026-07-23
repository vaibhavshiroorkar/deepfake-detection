import numpy as np
import pytest
from dashboard.lib import visual_ops as V


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.mark.parametrize("fn,kw", [
    (V.sharpen, {"amount": 1.0}),
    (V.denoise, {"strength": 5}),
    (V.clahe, {"clip_limit": 2.0}),
    (V.gaussian_blur, {"kernel": 5}),
    (V.jpeg_recompress, {"quality": 30}),
])
def test_op_preserves_shape_and_dtype(img, fn, kw):
    out = fn(img, **kw)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_downscale_upscale_returns_same_shape(img):
    out = V.downscale_upscale(img, factor=0.25)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_gaussian_blur_reduces_variance(img):
    assert V.gaussian_blur(img, kernel=9).var() < img.var()


def test_mouth_region_is_96(img):
    out = V.mouth_region(img, size=96)
    assert out.shape == (96, 96, 3)


def test_imagenet_normalize_range(img):
    arr = V.imagenet_normalize(img)
    assert arr.dtype == np.float32
    lo, hi = V.normalized_range(arr)
    assert lo < 0 and hi > 0            # zero-centered, not [0,1]
