"""The torch cross-attention must not drift from the documented numpy spec.

`dashboard/lib/cross_modal.py` is what the streams pages have shown and what the
handbook describes. If the trainable module quietly diverges from it, the page
stops describing the model and nobody finds out.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.dashboard.lib import cross_modal
from deepfake_detection.streams.cross_attention import (
    CrossModalAttention,
    diagonal_mass,
    scaled_dot_product_attention,
)


def test_attention_matches_the_numpy_specification() -> None:
    generator = np.random.default_rng(17)
    query = generator.normal(size=(7, 16)).astype(np.float32)
    key = generator.normal(size=(11, 16)).astype(np.float32)
    value = generator.normal(size=(11, 16)).astype(np.float32)

    expected_weights, expected_attended = cross_modal.scaled_dot_product_attention(
        query, key, value
    )
    result = scaled_dot_product_attention(
        torch.from_numpy(query).unsqueeze(0),
        torch.from_numpy(key).unsqueeze(0),
        torch.from_numpy(value).unsqueeze(0),
    )

    assert np.allclose(result.weights[0].numpy(), expected_weights, atol=1e-5)
    assert np.allclose(result.attended[0].numpy(), expected_attended, atol=1e-5)


def test_diagonal_mass_matches_the_numpy_specification() -> None:
    generator = np.random.default_rng(29)
    weights = generator.random((13, 13)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)

    expected = cross_modal.diagonal_mass(weights, band=1)
    result = diagonal_mass(torch.from_numpy(weights).unsqueeze(0), band=1)

    assert result.shape == (1,)
    assert float(result[0]) == pytest.approx(expected, abs=1e-5)


def test_diagonal_mass_separates_aligned_from_shuffled() -> None:
    """The metric has to discriminate before any model is trained, or it cannot
    be used to judge one afterwards."""
    size = 40
    aligned = torch.eye(size).unsqueeze(0)
    uniform = torch.full((1, size, size), 1.0 / size)

    assert float(diagonal_mass(aligned)) == pytest.approx(1.0, abs=1e-6)
    # A uniform map spreads mass everywhere, so only the ~3 in-band cells per row
    # count: roughly band-width over sequence length.
    assert float(diagonal_mass(uniform)) < 0.1


def test_diagonal_mass_handles_different_sampling_rates() -> None:
    """Video and audio arrive at different token rates, so a square map is the
    exception rather than the rule."""
    queries, keys = 20, 100
    weights = torch.zeros(1, queries, keys)
    for row in range(queries):
        weights[0, row, int(row * keys / queries)] = 1.0

    assert float(diagonal_mass(weights)) == pytest.approx(1.0, abs=1e-6)


def test_single_head_module_reduces_to_the_specification() -> None:
    """With identity projections one head must be the plain mechanism."""
    torch.manual_seed(17)
    module = CrossModalAttention(dim=8, num_heads=1)
    for layer in (
        module.query_projection,
        module.key_projection,
        module.value_projection,
        module.output_projection,
    ):
        torch.nn.init.eye_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    module.eval()

    query = torch.randn(1, 5, 8)
    key = torch.randn(1, 9, 8)
    value = torch.randn(1, 9, 8)

    with torch.no_grad():
        result = module(query=query, key=key, value=value)
    reference = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(result.weights, reference.weights, atol=1e-5)
    assert torch.allclose(result.attended, reference.attended, atol=1e-5)


def test_module_shapes_survive_unequal_sequence_lengths() -> None:
    module = CrossModalAttention(dim=32, num_heads=4)
    query = torch.randn(3, 50, 32)
    key = torch.randn(3, 99, 32)

    result = module(query=query, key=key, value=key)

    assert result.attended.shape == (3, 50, 32)
    # Averaged over heads, so one interpretable map per item.
    assert result.weights.shape == (3, 50, 99)
    assert torch.allclose(result.weights.sum(dim=-1), torch.ones(3, 50), atol=1e-5)


def test_attention_rejects_mismatched_widths() -> None:
    with pytest.raises(ValueError, match="share a feature width"):
        scaled_dot_product_attention(
            torch.randn(1, 4, 8), torch.randn(1, 4, 16), torch.randn(1, 4, 16)
        )


def test_attention_rejects_key_value_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share a time length"):
        scaled_dot_product_attention(
            torch.randn(1, 4, 8), torch.randn(1, 6, 8), torch.randn(1, 5, 8)
        )


def test_heads_must_divide_the_width() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        CrossModalAttention(dim=30, num_heads=4)
