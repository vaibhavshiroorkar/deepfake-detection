"""Threshold rule (PROTOCOL 5) and temperature scaling (A2)."""

import numpy as np
import pytest

from dfd.eval import metrics as M
from dfd.eval.calibration import apply_temperature, fit_temperature, logit, nll, sigmoid
from dfd.eval.threshold import choose_threshold_max_balanced_accuracy as choose


def test_threshold_hand_examples():
    assert choose([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).threshold == pytest.approx(0.5)
    # BA .75 at midpoints .2 and .75; .75 is closer to 0.5
    c = choose([0, 1, 0, 1], [0.1, 0.3, 0.6, 0.9])
    assert c.threshold == pytest.approx(0.75)
    assert c.balanced_accuracy == pytest.approx(0.75)
    assert c.n_candidates == 3


def test_threshold_is_deterministic():
    y = [0, 1, 0, 1, 0, 1]
    s = [0.0, 0.3, 0.5, 0.5, 0.5, 0.7]
    assert choose(y, s) == choose(y, s)


@pytest.mark.parametrize("seed", range(10))
def test_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 60
    y = (rng.random(n) < 0.4).astype(int)
    y[:2] = [0, 1]
    s = np.round(rng.random(n) * (1 + 0.5 * y), 1 if seed % 2 else 3)
    u = np.unique(s)
    mids = (u[:-1] + u[1:]) / 2
    ba = np.array([M.balanced_accuracy(y, s, t) for t in mids])
    best = np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))
    want = mids[best][np.argmin(np.abs(mids[best] - 0.5))]
    got = choose(y, s)
    assert got.threshold == pytest.approx(want, abs=1e-12)
    assert got.balanced_accuracy == pytest.approx(ba.max(), abs=1e-12)


def test_threshold_errors():
    with pytest.raises(M.OneClassError):
        choose([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        choose([0, 1], [0.5, 0.5])


def test_logit_clipping_and_sigmoid():
    assert logit(0.0) == pytest.approx(np.log(1e-7 / (1 - 1e-7)))
    assert logit(1.0) == pytest.approx(-logit(0.0))
    assert logit(0.5) == 0.0
    z = np.array([-800.0, -3.0, 0.0, 3.0, 800.0])
    assert np.all(np.isfinite(sigmoid(z)))
    assert sigmoid(0.0) == 0.5


@pytest.mark.parametrize("t_true", [0.4, 1.0, 1.7, 3.0])
def test_a2_temperature_recovery(t_true):
    # logits z; the true probability of fake is sigmoid(z / t_true)
    rng = np.random.default_rng(int(t_true * 100))
    n = 200_000
    z = rng.normal(0, 3.0 * t_true, n)
    y = (rng.random(n) < sigmoid(z / t_true)).astype(int)
    fit = fit_temperature(z, y)
    assert not fit.at_bound
    assert abs(fit.temperature / t_true - 1) < 0.02
    assert fit.nll <= fit.nll_at_one + 1e-12


def test_temperature_matches_scipy_minimiser():
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(7)
    z = rng.normal(0, 4, 5000)
    y = (rng.random(5000) < sigmoid(z / 2.5)).astype(int)
    ref = minimize_scalar(lambda t: nll(z, y, t), bounds=(0.05, 50), method="bounded",
                          options={"xatol": 1e-10})
    assert fit_temperature(z, y).temperature == pytest.approx(ref.x, rel=1e-5)


def test_temperature_bound_and_ranking():
    z = np.array([-5.0, -4.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1])  # separable: T runs to the lower bound
    assert fit_temperature(z, y).at_bound
    p = apply_temperature(np.array([-2.0, 0.5, 3.0]), 2.0)
    assert np.all(np.diff(p) > 0)
    with pytest.raises(M.OneClassError):
        fit_temperature(z, np.ones(4, dtype=int))
    with pytest.raises(ValueError):
        apply_temperature(z, 0.0)
