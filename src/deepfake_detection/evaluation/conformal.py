"""False-alarm control from real clips alone.

A detector's AUC can hold on a new corpus while its threshold stops meaning
anything: the same score that flagged 5 percent of real clips at home flags 40
percent after a codec change. A platform cannot label fakes in its own traffic,
but it has endless real clips. These functions set the threshold from real
target-domain scores only.

Two guarantees, from the order statistics of n real scores (higher = more fake):

- `conformal_threshold`: on average over calibration draws, a new real clip is
  flagged with probability at most alpha (split conformal; Vovk et al. 2005,
  Bates et al. 2023).
- `neyman_pearson_threshold`: with probability at least 1 - delta over the
  calibration draw, the false-positive rate is at most alpha (the umbrella
  algorithm of Tong, Feng and Li, Science Advances 2018). It needs more real
  clips; `minimum_calibration_size` says how many.

Both assume the calibration clips and the clips being screened come from the
same distribution. That is the assumption the experiments test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


def _scores(values: Sequence[float]) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=float))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Calibration scores must be a non-empty set of finite values")
    return array


def conformal_threshold(real_scores: Sequence[float], alpha: float) -> float:
    """Flag a clip when its score is strictly above the returned value."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = _scores(real_scores)
    rank = math.ceil((scores.size + 1) * (1 - alpha))
    if rank > scores.size:
        # Too few real clips for this alpha: nothing can be flagged safely.
        return math.inf
    return float(scores[rank - 1])


def neyman_pearson_threshold(real_scores: Sequence[float], alpha: float, delta: float) -> float:
    """Smallest order statistic whose FPR exceeds alpha with probability <= delta."""
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must be in (0, 1)")
    scores = _scores(real_scores)
    n = scores.size
    for rank in range(1, n + 1):
        # P(FPR > alpha) when thresholding at the rank-th smallest score.
        violation = binom.sf(rank - 1, n, 1 - alpha)
        if violation <= delta:
            return float(scores[rank - 1])
    return math.inf


def minimum_calibration_size(alpha: float, delta: float) -> int:
    """Fewest real clips for which the umbrella threshold exists at all."""
    return math.ceil(math.log(delta) / math.log(1 - alpha))


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    false_positive_rate: float
    true_positive_rate: float
    real: int
    fake: int


def operating_point(
    threshold: float, labels: Sequence[int], scores: Sequence[float]
) -> OperatingPoint:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    flagged = scores_array > threshold
    real, fake = labels_array == 0, labels_array == 1
    return OperatingPoint(
        threshold=float(threshold),
        false_positive_rate=float(flagged[real].mean()) if real.any() else math.nan,
        true_positive_rate=float(flagged[fake].mean()) if fake.any() else math.nan,
        real=int(real.sum()),
        fake=int(fake.sum()),
    )
