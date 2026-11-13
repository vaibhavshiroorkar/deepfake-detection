import math

import numpy as np
import pytest

from deepfake_detection.evaluation.conformal import (
    conformal_threshold,
    minimum_calibration_size,
    neyman_pearson_threshold,
    operating_point,
)


def test_minimum_size_matches_the_known_59() -> None:
    # (1 - 0.05)^59 < 0.05 <= (1 - 0.05)^58
    assert minimum_calibration_size(0.05, 0.05) == 59


def test_too_few_real_clips_flag_nothing() -> None:
    assert conformal_threshold([0.1] * 10, alpha=0.05) == math.inf
    assert neyman_pearson_threshold([0.1] * 58, alpha=0.05, delta=0.05) == math.inf
    assert neyman_pearson_threshold(np.linspace(0, 1, 59), alpha=0.05, delta=0.05) == 1.0


def test_conformal_threshold_holds_fpr_on_average() -> None:
    generator = np.random.default_rng(0)
    rates = []
    for _ in range(2000):
        calibration = generator.normal(size=100)
        threshold = conformal_threshold(calibration, alpha=0.1)
        rates.append((generator.normal(size=2000) > threshold).mean())
    assert np.mean(rates) <= 0.1 + 0.005


def test_umbrella_threshold_holds_fpr_with_high_probability() -> None:
    generator = np.random.default_rng(1)
    violations = 0
    trials = 1000
    for _ in range(trials):
        threshold = neyman_pearson_threshold(generator.normal(size=200), alpha=0.05, delta=0.1)
        violations += (generator.normal(size=20000) > threshold).mean() > 0.05
    assert violations / trials <= 0.1 + 0.03


def test_a_shifted_target_breaks_a_source_threshold_but_not_a_target_one() -> None:
    generator = np.random.default_rng(2)
    source_real = generator.normal(0, 1, 500)
    target_real = generator.normal(1.5, 1, 500)  # a codec change moves every real score up
    target_fake = generator.normal(4.0, 1, 500)
    labels = [0] * 300 + [1] * 500
    scores = np.concatenate([target_real[200:], target_fake])
    source = operating_point(conformal_threshold(source_real, 0.05), labels, scores)
    target = operating_point(conformal_threshold(target_real[:200], 0.05), labels, scores)
    assert source.false_positive_rate > 0.3
    assert target.false_positive_rate < 0.1
    assert target.true_positive_rate > 0.7  # about 0.80 expected at 2.5 SD separation


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        conformal_threshold([], 0.05)
    with pytest.raises(ValueError):
        conformal_threshold([0.1, float("nan")], 0.05)
    with pytest.raises(ValueError):
        neyman_pearson_threshold([0.1], 0.05, 1.5)
