"""The pure pieces behind the stream pages: tensor building and the attention demo.

The train_command tests that used to live here are gone with the Train tab. The
dashboard does not train (PROJECT_OVERVIEW.md section 7) and no longer offers to
assemble the command for something else that does.
"""
import numpy as np
import pytest

from dashboard.lib import cross_modal
from dashboard.lib.inference import frames_to_tensor


def test_frames_to_tensor_shape_and_normalization():
    torch = __import__("torch")
    faces = [np.full((224, 224, 3), 128, np.uint8) for _ in range(16)]
    t = frames_to_tensor(faces)
    assert t.shape == (1, 16, 3, 224, 224)
    assert t.dtype == torch.float32
    # 128/255 normalized by ImageNet stats lands roughly in [-2, 2], not [0, 255]
    assert float(t.abs().max()) < 3.0


# --------------------------------------------------- cross-modal attention demo

def test_attention_rows_are_distributions():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    weights, out = cross_modal.scaled_dot_product_attention(q, k, k)
    assert weights.shape == (6, 6)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all()
    assert out.shape == (6, 8)


def test_attention_output_is_a_weighted_sum_of_values():
    """One query far closer to one key must return close to that key's value."""
    q = np.array([[10.0, 0.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[5.0, 5.0], [-5.0, -5.0]])
    weights, out = cross_modal.scaled_dot_product_attention(q, k, v)
    assert weights[0, 0] > 0.99
    assert out[0] == pytest.approx([5.0, 5.0], abs=0.1)


def test_attention_survives_large_scores():
    """The softmax subtracts the row max, so big dot products must not overflow."""
    q = np.full((2, 4), 500.0)
    weights, _ = cross_modal.scaled_dot_product_attention(q, q, q)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_diagonal_mass_is_one_for_perfectly_aligned_attention():
    assert cross_modal.diagonal_mass(np.eye(5)) == pytest.approx(1.0)


def test_diagonal_mass_is_small_when_attention_sits_off_the_diagonal():
    weights = np.eye(6)[::-1]        # anti-diagonal: every step attends to the wrong one
    assert cross_modal.diagonal_mass(weights) < 0.5


def test_random_projection_flattens_and_reshapes():
    frames = np.zeros((4, 96, 96, 3), dtype=np.float32)
    out = cross_modal.random_projection(frames, cross_modal.DEMO_DIM, seed=0)
    assert out.shape == (4, cross_modal.DEMO_DIM)


def test_random_projection_is_stable_for_a_fixed_seed():
    """The picture must not change between reruns of the same page."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(3, 20))
    first = cross_modal.random_projection(data, 8, seed=7)
    second = cross_modal.random_projection(data, 8, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, cross_modal.random_projection(data, 8, seed=8))
