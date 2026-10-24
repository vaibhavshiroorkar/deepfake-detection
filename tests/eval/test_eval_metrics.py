"""A1: metrics against scikit-learn and against hand-computed values."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss, roc_auc_score

from dfd.eval import metrics as M


def random_set(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 400))
    y = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(int)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    s = np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1) * rng.uniform(0, 1), 0.3), 0, 1)
    if seed % 2:
        s = np.round(s, 2)  # many ties
    return y, s


@pytest.mark.parametrize("seed", range(24))
def test_a1_against_sklearn(seed):
    y, s = random_set(seed)
    assert abs(M.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-9
    clipped = np.clip(s, M.CLIP_EPS, 1 - M.CLIP_EPS)
    assert abs(M.log_loss(y, s) - sk_log_loss(y, clipped, labels=[0, 1])) < 1e-9
    assert abs(M.brier(y, s) - brier_score_loss(y, s)) < 1e-9


# hand example: 7 samples, threshold 0.5
Y = np.array([1, 1, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1])


def test_hand_balanced_accuracy_precision_recall():
    # predicted fake: 0.9, 0.8, 0.6 -> TP=2, FP=1, TN=3, FN=1
    assert M.balanced_accuracy(Y, S, 0.5) == pytest.approx((2 / 3 + 3 / 4) / 2, abs=1e-15)
    p, r = M.precision_recall(Y, S, 0.5)
    assert p == pytest.approx(2 / 3, abs=1e-15)
    assert r == pytest.approx(2 / 3, abs=1e-15)
    assert M.accuracy_at(Y, S, 0.5) == pytest.approx(5 / 7, abs=1e-15)


def test_hand_eer():
    # (FPR, FNR) by falling threshold: (0,1) (0,2/3) (0,1/3) (1/4,1/3) (1/4,0) ...
    # smallest gap is at t=0.6: EER = (1/4 + 1/3) / 2
    assert M.eer(Y, S) == pytest.approx(7 / 24, abs=1e-15)
    assert M.eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert M.eer([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all tied


def test_hand_auc_with_ties():
    # pairs: (0.8 vs 0.2) win, (0.8 vs 0.8) tie, (0.4 vs 0.2) win, (0.4 vs 0.8) loss
    assert M.roc_auc([1, 1, 0, 0], [0.8, 0.4, 0.2, 0.8]) == pytest.approx((1 + 0.5 + 1 + 0) / 4)


P = np.array([0.1, 0.3, 0.6, 0.8, 0.95])
YP = np.array([0, 1, 1, 1, 0])


def test_hand_ece_top_label_and_class1():
    # confidences .9 .7 .6 .8 .95; correct 1 0 1 1 0; 5 bins of width .2
    # bin (.4,.6]: {.6} gap .4 weight 1/5; (.6,.8]: {.7,.8} acc .5 conf .75 weight 2/5;
    # (.8,1]: {.9,.95} acc .5 conf .925 weight 2/5
    assert M.ece(YP, P, n_bins=5) == pytest.approx(0.2 * 0.4 + 0.4 * 0.25 + 0.4 * 0.425, abs=1e-12)
    # class 1: singleton bins, mean |y - p|
    assert M.ece_class1(YP, P, n_bins=5) == pytest.approx((0.1 + 0.7 + 0.4 + 0.2 + 0.95) / 5, abs=1e-12)
    assert M.brier(YP, P) == pytest.approx(1.6025 / 5, abs=1e-12)


def test_ece_perfectly_calibrated_bin_is_zero():
    # 4 samples at p=0.75, three fake: acc = conf = 0.75
    assert M.ece([1, 1, 1, 0], [0.75] * 4) == pytest.approx(0.0, abs=1e-15)
    assert M.ece_class1([1, 1, 1, 0], [0.75] * 4) == pytest.approx(0.0, abs=1e-15)


def test_log_loss_clipping():
    assert M.log_loss([1], [0.0]) == pytest.approx(-np.log(1e-7), abs=1e-6)
    assert M.log_loss([0], [1.0]) == pytest.approx(-np.log(1e-7), abs=1e-6)


def test_weights_equal_repetition():
    y, s = random_set(3)
    w = np.random.default_rng(0).integers(0, 4, y.size)
    idx = np.repeat(np.arange(y.size), w)
    for f in (M.roc_auc, M.eer, M.brier, M.log_loss, M.ece, M.ece_class1):
        assert f(y, s, weights=w) == pytest.approx(f(y[idx], s[idx]), abs=1e-12), f.__name__
    assert M.balanced_accuracy(y, s, 0.45, weights=w) == pytest.approx(M.balanced_accuracy(y[idx], s[idx], 0.45))


def test_evaluators_match_public_functions():
    y, s = random_set(5)
    w = np.random.default_rng(1).integers(0, 3, y.size).astype(float)
    pairs = {"auc": M.roc_auc, "eer": M.eer, "brier": M.brier, "log_loss": M.log_loss,
             "ece": M.ece, "ece_class1": M.ece_class1}
    for name, f in pairs.items():
        assert M.make_evaluator(name, y, s)(w) == pytest.approx(f(y, s, weights=w), abs=1e-12)
    assert M.make_evaluator("balanced_accuracy", y, s, threshold=0.5)(w) == pytest.approx(
        M.balanced_accuracy(y, s, 0.5, weights=w))


def test_one_class_errors():
    for f in (M.roc_auc, M.eer):
        with pytest.raises(M.OneClassError):
            f([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(M.OneClassError):
        M.balanced_accuracy([0, 0], [0.1, 0.2], 0.5)
    with pytest.raises(M.OneClassError):
        M.precision_recall([0, 0], [0.1, 0.2], 0.5)
    # zero weight on one class is the same thing
    with pytest.raises(M.OneClassError):
        M.roc_auc([0, 1], [0.1, 0.9], weights=[1, 0])


def test_input_errors():
    with pytest.raises(ValueError):
        M.roc_auc([0, 2], [0.1, 0.9])
    with pytest.raises(ValueError):
        M.brier([0, 1], [0.1, 1.2])
    with pytest.raises(ValueError):
        M.roc_auc([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError):
        M.roc_auc([0, 1], [0.1])
