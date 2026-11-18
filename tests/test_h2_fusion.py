import numpy as np
import pytest

from deepfake_detection.fusion.h2 import (
    SeedPair,
    cross_fit_scores,
    fit_fusion,
    holm,
    logit,
    mean_seed_difference,
)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    groups = np.array([f"id{i // 4}" for i in range(n)])
    visual = labels * 0.8 + rng.normal(0, 1, n)
    audio = labels * 1.5 + rng.normal(0, 1, n)
    sync = rng.normal(0, 1, n)
    return labels, groups, np.column_stack([visual, audio, sync])


def test_logit_clips_extremes():
    values = logit([0.0, 0.5, 1.0])
    assert np.isfinite(values).all()
    assert values[1] == pytest.approx(0.0)


def test_fusion_uses_the_informative_branch():
    labels, _, x = _data()
    fusion = fit_fusion(x, labels)
    weights = fusion.model.coef_[0]
    assert weights[1] > weights[0] > abs(weights[2])


def test_cross_fit_never_scores_an_identity_it_trained_on(monkeypatch):
    labels, groups, x = _data()
    seen = []
    import deepfake_detection.fusion.h2 as h2

    real_fit = h2.fit_fusion

    def spy(logits, fold_labels):
        seen.append(len(fold_labels))
        return real_fit(logits, fold_labels)

    monkeypatch.setattr(h2, "fit_fusion", spy)
    scores = cross_fit_scores(x, labels, groups)
    assert len(seen) == 5 and all(n < len(labels) for n in seen)
    assert scores.shape == (len(labels),)


def test_fusion_beats_visual_when_audio_carries_signal():
    labels, groups, x = _data()
    fused = fit_fusion(x, labels).score(x)
    pair = SeedPair(labels, groups, fused, x[:, 0])
    result = mean_seed_difference([pair, pair, pair], samples=200)
    assert result["estimate"] > 0.05
    assert result["lower"] > 0
    assert result["p_one_sided"] < 0.01
    assert result["identities"] == 100


def test_seeds_must_cover_the_same_clips():
    labels, groups, x = _data()
    a = SeedPair(labels, groups, x[:, 1], x[:, 0])
    b = SeedPair(labels[::-1].copy(), groups, x[:, 1], x[:, 0])
    with pytest.raises(ValueError):
        mean_seed_difference([a, b], samples=10)


def test_holm_step_down():
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {
        "a": True,
        "b": False,
        "c": False,
    }
    assert holm({"a": 0.2, "b": 0.001, "c": 0.5}) == {"a": False, "b": True, "c": False}
