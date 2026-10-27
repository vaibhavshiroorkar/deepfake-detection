import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("av")

from deepfake_detection.training.augment import (  # noqa: E402
    CODECS,
    VisualAugmentConfig,
    VisualAugmenter,
    codec_roundtrip,
    face_mask,
    self_blend,
    to_pixels,
    to_view,
)


def _clip(seed: int = 3) -> np.ndarray:
    """A smooth synthetic face-sized clip, so codecs have structure to keep."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    base = np.stack([x * 3, y * 3, (x + y) * 2], axis=-1).astype(np.float32)
    frames = [np.clip(base + rng.normal(0, 4, base.shape), 0, 255) for _ in range(4)]
    return to_view(np.stack(frames).astype(np.uint8))


def test_pixels_and_views_round_trip_exactly() -> None:
    view = _clip()
    assert np.allclose(to_view(to_pixels(view)), view, atol=1e-6)


@pytest.mark.parametrize("codec", CODECS)
def test_each_codec_keeps_shape_and_changes_pixels(codec: str) -> None:
    pixels = to_pixels(_clip())
    out = codec_roundtrip(pixels, codec, 0.0, np.random.default_rng(0))

    assert out.shape == pixels.shape and out.dtype == np.uint8
    assert not np.array_equal(out, pixels)
    assert np.abs(out.astype(int) - pixels).mean() < 40


def test_the_face_mask_is_soft_and_centred() -> None:
    mask = face_mask(64, 64, np.random.default_rng(1))
    assert mask.max() == pytest.approx(1.0)
    assert mask[32, 32] > 0.9 and mask[0, 0] < 0.05
    assert 0.05 < mask[32, 10] < 0.95 or 0.05 < mask[10, 32] < 0.95


def test_self_blending_alters_the_face_and_leaves_the_border() -> None:
    pixels = to_pixels(_clip())
    blended = self_blend(pixels, np.random.default_rng(2))
    change = np.abs(blended.astype(int) - pixels).mean(axis=(0, 3))

    assert change[32, 32] > change[1, 1]


def test_a_blended_genuine_clip_is_labelled_fake() -> None:
    augmenter = VisualAugmenter(
        VisualAugmentConfig(compression=0.0, self_blend=1.0), seed=17
    )
    view, label = augmenter(_clip(), 0.0)
    assert label == 1.0
    assert view.shape == _clip().shape and view.dtype == np.float32


def test_a_fake_clip_keeps_its_label() -> None:
    augmenter = VisualAugmenter(VisualAugmentConfig(self_blend=1.0), seed=17)
    _, label = augmenter(_clip(), 1.0)
    assert label == 1.0


def test_probabilities_are_validated() -> None:
    with pytest.raises(ValueError, match="compression must be a probability"):
        VisualAugmentConfig(compression=1.5)


def test_codec_swap_is_deterministic_and_keeps_labels() -> None:
    from deepfake_detection.training.augment import CodecSwap

    swap = CodecSwap()
    first, label = swap(_clip(), 0.0)
    second, _ = swap(_clip(), 0.0)
    assert label == 0.0
    assert np.array_equal(first, second)
    assert not np.allclose(first, _clip())
