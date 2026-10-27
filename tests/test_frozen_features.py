import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepfake_detection.inference.frozen_features import (  # noqa: E402
    _normalised,
    embed_windows,
    window_starts,
)


def test_each_encoder_applies_its_own_statistics() -> None:
    """PE expects 0.5 statistics; feeding it ImageNet ones was a silent bug."""
    encoder = _normalised(lambda frames: frames, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    pixels = torch.full((1, 3, 2, 2), 0.75)

    assert torch.allclose(encoder(pixels), torch.full((1, 3, 2, 2), 0.5))


def test_embedding_hands_the_encoder_unit_range_pixels() -> None:
    seen = []

    def encode(frames):
        seen.append(frames)
        return frames.mean(dim=(2, 3))

    encoder = _normalised(encode, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    windows = np.full((2, 3, 4, 4, 3), 255, dtype=np.uint8)

    vector = embed_windows(encoder, windows, device="cpu")

    assert float(seen[0].max()) == pytest.approx(1.0)
    assert vector.shape == (6,)
    assert np.allclose(vector[:3], 1.0) and np.allclose(vector[3:], 0.0)


def test_windows_cover_both_ends_of_the_clip() -> None:
    assert window_starts(100, 3, 32) == [0, 34, 68]
    assert window_starts(20, 3, 32) == [0]
