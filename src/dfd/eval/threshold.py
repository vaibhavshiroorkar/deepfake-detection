"""Decision threshold chosen on validation data (PROTOCOL.md section 5).

Rule: candidates are the midpoints between adjacent distinct validation scores.
Predict fake when score >= t. Pick the candidate with the largest balanced
accuracy BA(t) = (TPR + TNR) / 2. Among equal maxima take the one closest to
0.5; if two are equally close, the lower one. Deterministic.

Equality of maxima is decided in exact integer arithmetic:
BA = (TP * n0 + TN * n1) / (2 * n0 * n1), so the integer TP * n0 + TN * n1 is
compared, not a float.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import OneClassError, _floats, _labels


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    balanced_accuracy: float  # on the data the threshold was chosen on
    n_candidates: int


def choose_threshold_max_balanced_accuracy(y, score) -> ThresholdChoice:
    y = _labels(y)
    s = _floats(score, "scores", y.size)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        raise OneClassError("threshold selection needs both classes in the validation data")
    u, inv = np.unique(s, return_inverse=True)
    if u.size < 2:
        raise ValueError("validation scores are constant, no midpoint threshold exists")
    c1 = np.bincount(inv, weights=y, minlength=u.size).astype(np.int64)
    c0 = np.bincount(inv, minlength=u.size).astype(np.int64) - c1
    # threshold between u[k] and u[k+1]: fake predicted for blocks k+1 and up
    tp = n1 - np.cumsum(c1)[:-1]
    tn = np.cumsum(c0)[:-1]
    num = tp * n0 + tn * n1
    lo, hi = u[:-1], u[1:]
    mid = lo + (hi - lo) / 2.0
    mid = np.where((mid <= lo) | (mid >= hi), hi, mid)  # adjacent floats: split exactly
    best = np.flatnonzero(num == num.max())
    pick = best[np.argmin(np.abs(mid[best] - 0.5))]  # first minimum is the lower threshold
    return ThresholdChoice(
        threshold=float(mid[pick]),
        balanced_accuracy=float(num[pick] / (2.0 * n0 * n1)),
        n_candidates=int(mid.size),
    )
