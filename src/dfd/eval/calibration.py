"""Temperature scaling (PROTOCOL.md section 5, MATH_NOTES.md section 7).

  p = sigmoid(z / T),  T > 0
  NLL(T) = sum_i w_i [ln(1 + e^{z_i/T}) - y_i z_i / T]

NLL is convex in a = 1/T (logistic regression with one feature and no
intercept), so the minimiser is the root of the monotone derivative
  g(a) = sum_i w_i (sigmoid(a z_i) - y_i) z_i.
fit_temperature does a bounded bisection for that root on a in [1/t_max, 1/t_min],
geometrically (on ln a). No scipy. If the root lies outside the bounds the bound
is returned and `at_bound` is True (for example perfectly separated logits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import CLIP_EPS, OneClassError, _floats, _labels, _weights


def logit(p, eps=CLIP_EPS):
    """ln(p / (1 - p)) with p clipped to [eps, 1 - eps]."""
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def apply_temperature(z, temperature):
    """sigmoid(z / T). Does not change the ranking, hence not the AUC or EER."""
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    return sigmoid(np.asarray(z, dtype=float) / temperature)


def nll(z, y, temperature, weights=None):
    """Mean negative log likelihood of sigmoid(z / T)."""
    y = _labels(y)
    z = _floats(z, "logits", y.size)
    w = _weights(weights, y.size)
    u = z / temperature
    return float(np.dot(w, np.logaddexp(0.0, u) - y * u) / w.sum())


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll: float  # mean NLL at the fitted T
    nll_at_one: float  # mean NLL at T = 1, for reference
    at_bound: bool
    n: int


def fit_temperature(z, y, weights=None, t_min=1e-2, t_max=1e2, tol=1e-12, max_iter=200):
    y = _labels(y)
    z = _floats(z, "logits", y.size)
    w = _weights(weights, y.size)
    if not (w[y == 1].sum() > 0 and w[y == 0].sum() > 0):
        raise OneClassError("temperature fitting needs both classes")
    if not 0 < t_min < t_max:
        raise ValueError("need 0 < t_min < t_max")

    def grad(a):
        return float(np.dot(w * z, apply_temperature(z, 1.0 / a) - y))

    lo, hi = 1.0 / t_max, 1.0 / t_min
    if grad(lo) >= 0:
        a, at_bound = lo, True
    elif grad(hi) <= 0:
        a, at_bound = hi, True
    else:
        at_bound = False
        for _ in range(max_iter):
            mid = np.sqrt(lo * hi)
            if grad(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1.0 < tol:
                break
        a = np.sqrt(lo * hi)
    t = float(1.0 / a)
    return TemperatureFit(t, nll(z, y, t, w), nll(z, y, 1.0, w), at_bound, int(y.size))
