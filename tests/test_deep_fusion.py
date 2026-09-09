"""Feature-level fusion over stream embeddings.

The behaviour worth pinning is stream dropout. The scalar late-fusion model put
a -3.019 coefficient on a branch scoring 0.4364 ROC-AUC, below chance, because
calibration could invert it into an in-domain gain. That gain did not survive a
change of corpus. A head that must work when a stream is absent cannot build its
decision on one stream that way.
"""

import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.fusion.deep import StreamFusion

DIMS = {"visual": 256, "lipsync": 256, "emotion": 128}


def embeddings(batch: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        name: torch.randn(batch, width, generator=generator)
        for name, width in DIMS.items()
    }


def test_forward_returns_one_logit_per_clip() -> None:
    model = StreamFusion(stream_dims=DIMS).eval()

    out = model(embeddings())

    assert out.logit.shape == (4,)
    assert out.presence.shape == (4, len(DIMS))


def test_streams_of_different_widths_are_projected_to_a_common_size() -> None:
    """Encoders emit different widths, so a stream must be swappable without
    rebuilding the head."""
    model = StreamFusion(stream_dims=DIMS, common_dim=64).eval()

    out = model(embeddings())

    assert out.logit.shape == (4,)
    for name, width in DIMS.items():
        assert model.projections[name].in_features == width
        assert model.projections[name].out_features == 64


def test_a_missing_stream_is_zeroed_and_flagged_not_dropped() -> None:
    """A clip one stream cannot read is still a clip. Removing it would shrink
    the denominator, which the abstention policy exists to prevent."""
    model = StreamFusion(stream_dims=DIMS).eval()
    values = embeddings()
    presence = {
        "visual": torch.ones(4),
        "lipsync": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "emotion": torch.ones(4),
    }

    out = model(values, presence)

    assert out.logit.shape == (4,)
    assert out.presence[:, model.stream_names.index("lipsync")].tolist() == [
        1.0,
        0.0,
        1.0,
        0.0,
    ]


def test_an_absent_stream_cannot_influence_the_result() -> None:
    """Whatever a stream emits must be ignored when it is marked absent, or the
    presence flag is decorative."""
    model = StreamFusion(stream_dims=DIMS).eval()
    values = embeddings()
    absent = {name: torch.ones(4) for name in DIMS}
    absent["lipsync"] = torch.zeros(4)

    with torch.no_grad():
        first = model(values, absent)
        noisy = dict(values)
        noisy["lipsync"] = torch.randn(4, DIMS["lipsync"]) * 100
        second = model(noisy, absent)

    assert torch.allclose(first.logit, second.logit, atol=1e-5)


def test_stream_dropout_is_active_only_in_training() -> None:
    model = StreamFusion(stream_dims=DIMS, stream_dropout=0.9)
    values = embeddings(batch=64)

    model.eval()
    with torch.no_grad():
        evaluated = model(values)
    assert evaluated.presence.min().item() == 1.0, "no stream may drop at eval"

    model.train()
    torch.manual_seed(17)
    trained = model(values)
    assert trained.presence.min().item() == 0.0, "streams must drop while training"


def test_stream_dropout_can_be_disabled() -> None:
    model = StreamFusion(stream_dims=DIMS, stream_dropout=0.0)
    model.train()

    out = model(embeddings(batch=32))

    assert out.presence.min().item() == 1.0


def test_gradients_reach_every_stream() -> None:
    """A stream the head cannot learn from is dead weight, and the failure is
    silent."""
    model = StreamFusion(stream_dims=DIMS, stream_dropout=0.0)
    model.train()
    out = model(embeddings())
    out.logit.sum().backward()

    for name in DIMS:
        weight = model.projections[name].weight
        assert weight.grad is not None
        assert weight.grad.abs().sum() > 0, f"{name} received no gradient"


def test_missing_embedding_is_an_error_not_a_silent_zero() -> None:
    model = StreamFusion(stream_dims=DIMS).eval()
    values = embeddings()
    del values["emotion"]

    with pytest.raises(ValueError, match="Missing stream embeddings: emotion"):
        model(values)


def test_rejects_an_empty_stream_set() -> None:
    with pytest.raises(ValueError, match="At least one stream"):
        StreamFusion(stream_dims={})


def test_rejects_an_impossible_dropout_rate() -> None:
    with pytest.raises(ValueError, match="Stream dropout"):
        StreamFusion(stream_dims=DIMS, stream_dropout=1.0)
