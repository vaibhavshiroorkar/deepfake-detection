"""Cluster (group) percentile bootstrap (PROTOCOL.md section 6, MATH_NOTES.md section 8).

Resampling unit: the group (`group_id`). One resample draws G groups with
replacement from the G groups and keeps every sample of a drawn group. A group
drawn k times gives each of its samples weight k, so a resample is a weight
vector over samples and metrics are evaluated with weights (no copying of
samples, no Python loop over samples). Group structure is precomputed.

Rules
  * B = 2000 resamples, 95 percent percentile interval (quantiles alpha/2 and
    1 - alpha/2 of the replicates, numpy 'linear' interpolation), base seed
    20261024. The draw of resample b depends only on the seed, the groups and
    the labels, not on the metric or the scores, so two runs on the same
    samples see identical resamples (this is what makes the comparison paired).
  * A resample without both classes is redrawn (D-024 item c). The number of
    redraws is reported. If the data itself has one class only, redrawing is
    off (require_both_classes=False) and only class-free metrics make sense.
  * Paired difference: replicate-wise metric(A) - metric(B) on the same groups.
  * Several seeds (section 6): for each resample the metric is computed for each
    seed's scores and averaged over seeds; the interval is that of this
    seed-averaged value. `seed_sd` is the standard deviation (ddof=1) over seeds
    of the point estimate. Scores are given as shape (n_seeds, n).
  * Replicates that are nan (for example precision when nothing is predicted
    fake) are dropped; their number is reported in `n_nan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .metrics import _labels, make_evaluator

BASE_SEED = 20261024
N_BOOT = 2000
LEVEL = 0.95
MAX_REDRAWS = 10000


@dataclass
class BootstrapResult:
    estimate: float  # point estimate on the full data (mean over seeds)
    low: float
    high: float
    se: float  # standard deviation of the replicates
    n_boot: int
    level: float
    n_nan: int = 0
    seed_sd: float | None = None  # sd over seeds of the point estimate, None for one seed
    seed_estimates: list = field(default_factory=list)
    replicates: np.ndarray | None = field(default=None, repr=False)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "se": self.se,
            "seed_sd": self.seed_sd,
            "n_nan": self.n_nan,
        }


class ClusterResampler:
    """Draws group resamples as per-sample weight vectors."""

    def __init__(self, y, groups, seed: int = BASE_SEED, require_both_classes: bool = True):
        y = _labels(y)
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("groups and labels must have the same length")
        _, self.gidx = np.unique(groups, return_inverse=True)
        self.n_groups = int(self.gidx.max()) + 1
        self.n1 = np.bincount(self.gidx, weights=y, minlength=self.n_groups)
        self.n0 = np.bincount(self.gidx, minlength=self.n_groups) - self.n1
        self.require_both = require_both_classes
        if require_both_classes and (self.n1.sum() == 0 or self.n0.sum() == 0):
            raise ValueError("data has one class only, cannot redraw to get both classes")
        self.rng = np.random.default_rng(seed)
        self.n_redrawn = 0

    def draw(self) -> np.ndarray:
        g = self.n_groups
        for _ in range(MAX_REDRAWS):
            counts = np.bincount(self.rng.integers(0, g, size=g), minlength=g)
            if not self.require_both or (counts @ self.n1 > 0 and counts @ self.n0 > 0):
                return counts[self.gidx].astype(float)
            self.n_redrawn += 1
        raise RuntimeError("could not draw a resample with both classes")

    def weights(self, n_boot: int):
        for _ in range(n_boot):
            yield self.draw()


def summarise(estimate, reps, level=LEVEL, seed_sd=None, seed_estimates=None) -> BootstrapResult:
    reps = np.asarray(reps, dtype=float)
    ok = reps[~np.isnan(reps)]
    n_nan = int(reps.size - ok.size)
    if ok.size == 0:
        low = high = se = float("nan")
    else:
        a = (1.0 - level) / 2.0
        low, high = (float(v) for v in np.quantile(ok, [a, 1.0 - a]))
        se = float(ok.std(ddof=1)) if ok.size > 1 else float("nan")
    return BootstrapResult(
        estimate=float(estimate), low=low, high=high, se=se, n_boot=int(reps.size),
        level=level, n_nan=n_nan, seed_sd=seed_sd,
        seed_estimates=list(seed_estimates or []), replicates=reps,
    )


def bootstrap_many(
    y,
    groups,
    evaluators: Mapping[str, Sequence[Callable[[np.ndarray], float]]],
    *,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    seed: int = BASE_SEED,
    require_both_classes: bool = True,
):
    """Bootstrap several named metrics on the same resamples.

    `evaluators[name]` is a list with one weight-evaluator per training seed
    (see metrics.make_evaluator). Returns (dict name -> BootstrapResult, resampler).
    """
    rs = ClusterResampler(y, groups, seed, require_both_classes)
    ones = np.ones(rs.gidx.size)
    names = list(evaluators)
    point = {k: np.array([f(ones) for f in evaluators[k]]) for k in names}
    reps = {k: np.empty((n_boot, len(evaluators[k]))) for k in names}
    for b, w in enumerate(rs.weights(n_boot)):
        for k in names:
            for j, f in enumerate(evaluators[k]):
                reps[k][b, j] = f(w)
    out = {}
    for k in names:
        pe = point[k]
        sd = float(pe.std(ddof=1)) if pe.size > 1 else None
        out[k] = summarise(pe.mean(), reps[k].mean(axis=1), level, sd, pe.tolist())
    return out, rs


def _per_seed(y, scores, metric, kw):
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError("scores must have shape (n,) or (n_seeds, n)")
    if callable(metric):
        return [lambda w, row=row: metric(y, row, weights=w) for row in s]
    return [make_evaluator(metric, y, row, **kw) for row in s]


def cluster_bootstrap(y, scores, groups, metric="auc", *, n_boot=N_BOOT, level=LEVEL,
                      seed=BASE_SEED, **kw) -> BootstrapResult:
    """Group bootstrap of one metric. `scores` is (n,) or (n_seeds, n).

    `metric` is a name known to metrics.make_evaluator (extra keyword arguments
    such as threshold= go to it) or a callable f(y, s, weights=None).
    """
    ev = {"m": _per_seed(y, scores, metric, kw)}
    res, _ = bootstrap_many(y, groups, ev, n_boot=n_boot, level=level, seed=seed)
    return res["m"]


def paired_bootstrap(y, scores_a, scores_b, groups, metric="auc", *, n_boot=N_BOOT,
                     level=LEVEL, seed=BASE_SEED, **kw):
    """Paired comparison on the same resampled groups.

    Returns (diff, a, b): `diff` is the BootstrapResult of metric(A) - metric(B)
    (replicate-wise difference of the seed-averaged values), a and b are the
    results of the two candidates. `diff.seed_sd` is None.
    """
    return paired_from_evaluators(
        y, groups, _per_seed(y, scores_a, metric, kw), _per_seed(y, scores_b, metric, kw),
        n_boot=n_boot, level=level, seed=seed)


def paired_from_evaluators(y, groups, evals_a, evals_b, *, n_boot=N_BOOT, level=LEVEL,
                           seed=BASE_SEED, require_both_classes=True):
    """Paired bootstrap from lists of weight-evaluators (one per seed on each side).

    Returns (diff, a, b) as in paired_bootstrap. Both sides see the same resamples.
    """
    res, _ = bootstrap_many(y, groups, {"a": evals_a, "b": evals_b}, n_boot=n_boot, level=level,
                            seed=seed, require_both_classes=require_both_classes)
    a, b = res["a"], res["b"]
    diff = summarise(a.estimate - b.estimate, a.replicates - b.replicates, level)
    return diff, a, b
