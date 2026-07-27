"""EXTRAS visual ops, shape/dtype preservation and expected effects."""
import numpy as np
import pytest

from preprocessing.ops import extras_visual as VX


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.mark.parametrize("fn,kw", [
    (VX.sharpen, {"amount": 1.0}),
    (VX.denoise, {"strength": 5}),
    (VX.clahe, {"clip_limit": 2.0}),
    (VX.gaussian_blur, {"kernel": 5}),
    (VX.jpeg_recompress, {"quality": 30}),
])
def test_op_preserves_shape_and_dtype(img, fn, kw):
    out = fn(img, **kw)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_downscale_upscale_returns_same_shape(img):
    out = VX.downscale_upscale(img, factor=0.25)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_gaussian_blur_reduces_variance(img):
    assert VX.gaussian_blur(img, kernel=9).var() < img.var()
