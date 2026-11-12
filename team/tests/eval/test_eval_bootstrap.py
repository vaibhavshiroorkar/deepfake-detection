"""Cluster bootstrap: A3 (coverage), A4 (paired identical runs), rules and speed."""

import math
import time

import numpy as np
import pytest

from dfd.eval import metrics as M
from dfd.eval.bootstrap import (BASE_SEED, ClusterResampler, bootstrap_many, cluster_bootstrap,
                                paired_bootstrap)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def simulate(rng, n_groups=80, per_group=6, d=1.0, sd_group=0.7):
    """Half fake, half real in each group; a group effect shifts all its scores."""
    g = np.repeat(np.arange(n_groups), per_group)
    y = np.tile([1, 1, 1, 0, 0, 0], n_groups)
    s = d * y + rng.normal(0, sd_group, n_groups)[g] + rng.normal(0, 1, g.size)
    return y, s, g


def test_a3_coverage_of_true_auc():
    # true AUC of a random (fake, real) pair: different groups with prob 1 - 1/G
    G, d, sg = 80, 1.0, 0.7
    truth = (1 - 1 / G) * phi(d / math.sqrt(2 * (sg**2 + 1))) + (1 / G) * phi(d / math.sqrt(2))
    rng = np.random.default_rng(20261024)
    hits = 0
    for _ in range(200):
        y, s, g = simulate(rng, G, d=d, sd_group=sg)
        r = cluster_bootstrap(y, s, g, "auc", n_boot=2000, seed=int(rng.integers(1 << 31)))
        hits += r.contains(truth)
    print("coverage", hits / 200)
    assert hits >= 180


def test_a4_paired_identical_runs_give_exact_zero():
    rng = np.random.default_rng(1)
    y, s, g = simulate(rng)
    for scores in (s, np.stack([s, s + 0.1 * rng.normal(size=s.size)])):
        diff, a, b = paired_bootstrap(y, scores, scores.copy(), g, "auc", n_boot=300)
        assert diff.estimate == 0.0
        assert diff.low == 0.0 and diff.high == 0.0
        assert np.all(diff.replicates == 0.0)
        assert a.low == b.low and a.high == b.high


def test_paired_interval_is_tighter_than_unpaired_for_correlated_runs():
    rng = np.random.default_rng(2)
    y, s, g = simulate(rng, n_groups=120)
    s2 = s + rng.normal(0, 0.3, s.size)  # a noisy copy: strongly correlated runs
    diff, a, b = paired_bootstrap(y, s, s2, g, "auc", n_boot=500)
    assert diff.se < np.hypot(a.se, b.se)
    assert diff.estimate == pytest.approx(M.roc_auc(y, s) - M.roc_auc(y, s2), abs=1e-15)


def test_reproducible_and_seed_changes_result():
    rng = np.random.default_rng(3)
    y, s, g = simulate(rng)
    r1 = cluster_bootstrap(y, s, g, "auc", n_boot=200)
    r2 = cluster_bootstrap(y, s, g, "auc", n_boot=200)
    r3 = cluster_bootstrap(y, s, g, "auc", n_boot=200, seed=BASE_SEED + 1)
    assert np.array_equal(r1.replicates, r2.replicates)
    assert not np.array_equal(r1.replicates, r3.replicates)
    assert r1.level == 0.95


def test_resample_matches_explicit_expansion():
    rng = np.random.default_rng(4)
    y, s, g = simulate(rng, n_groups=30)
    rs = ClusterResampler(y, g, seed=11)
    w = rs.draw()
    idx = np.repeat(np.arange(y.size), w.astype(int))
    assert M.roc_auc(y, s, weights=w) == pytest.approx(M.roc_auc(y[idx], s[idx]), abs=1e-12)
    # groups are drawn whole: each group's samples share one weight
    for gi in np.unique(g):
        assert len(set(w[g == gi])) == 1
    assert w.sum() == y.size  # 30 groups of equal size


def test_one_class_resamples_are_redrawn():
    # 6 groups; only group 0 holds fakes, so many draws miss it
    g = np.repeat(np.arange(6), 4)
    y = np.zeros(24, dtype=int)
    y[:4] = 1
    rs = ClusterResampler(y, g, seed=5)
    for w in rs.weights(300):
        assert (w * y).sum() > 0 and (w * (1 - y)).sum() > 0
    assert rs.n_redrawn > 0
    with pytest.raises(ValueError):
        ClusterResampler(np.zeros(4, dtype=int), np.arange(4))


def test_multi_seed_rule():
    rng = np.random.default_rng(6)
    y, s0, g = simulate(rng, n_groups=40)
    stack = np.stack([s0, s0 + rng.normal(0, 0.5, s0.size), s0 + rng.normal(0, 0.5, s0.size)])
    r = cluster_bootstrap(y, stack, g, "auc", n_boot=150)
    per_seed = [M.roc_auc(y, row) for row in stack]
    assert r.estimate == pytest.approx(np.mean(per_seed), abs=1e-14)
    assert r.seed_sd == pytest.approx(np.std(per_seed, ddof=1), abs=1e-14)
    assert r.seed_estimates == pytest.approx(per_seed)
    # replicate b is the mean over seeds of the metric on resample b
    rs = ClusterResampler(y, g, seed=BASE_SEED)
    manual = [np.mean([M.roc_auc(y, row, weights=w) for row in stack]) for w in rs.weights(150)]
    assert np.allclose(r.replicates, manual, atol=1e-14)
    lo, hi = np.quantile(manual, [0.025, 0.975])
    assert (r.low, r.high) == pytest.approx((lo, hi), abs=1e-14)


def test_many_metrics_share_resamples_and_nan_dropped():
    rng = np.random.default_rng(8)
    y, s, g = simulate(rng, n_groups=40)
    p = 1 / (1 + np.exp(-s))
    ev = {"auc": [M.make_evaluator("auc", y, p)], "brier": [M.make_evaluator("brier", y, p)],
          "prec": [M.make_evaluator("precision", y, p, threshold=2.0)]}  # nothing predicted fake
    res, rs = bootstrap_many(y, g, ev, n_boot=100)
    assert res["prec"].n_nan == 100
    assert math.isnan(res["prec"].low)
    assert res["brier"].low < res["brier"].estimate < res["brier"].high


def test_speed_50000_samples_5000_groups():
    rng = np.random.default_rng(9)
    n, G = 50_000, 5_000
    g = rng.integers(0, G, n)
    y = (rng.random(n) < 0.5).astype(int)
    s = np.clip(rng.normal(0.5 + 0.1 * y, 0.3), 0, 1)
    t = time.perf_counter()
    r = cluster_bootstrap(y, s, g, "auc", n_boot=2000)
    dt = time.perf_counter() - t
    print(f"bootstrap 2000 x 50000 samples x 5000 groups: {dt:.2f} s")
    assert dt < 30
    assert r.low < r.estimate < r.high
