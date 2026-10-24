"""Metrics of the Stage 2 protocol (docs/PROTOCOL.md section 4, docs/MATH_NOTES.md).

numpy only. Labels are 0 (real) or 1 (fake, the positive class). Scores are
probabilities of fake unless stated. Every public function takes optional
non-negative `weights`; an integer weight w_i is the same as repeating sample i
w_i times. The bootstrap uses this: a resample of groups is a weight vector.

Formulas
  AUC   = sum_{i fake, j real} [1(s_i > s_j) + 0.5 * 1(s_i = s_j)] / (n1 * n0)
  EER   = (FPR(t) + FNR(t)) / 2 at the candidate t that minimises |FPR - FNR|
          (t runs over the distinct scores, predict fake when s >= t, and over
          t = +inf; ties in the gap go to the first t in order of rising FPR)
  BA    = (TPR + TNR) / 2 at a given threshold, predict fake when s >= t
  log loss = -mean[y ln p + (1 - y) ln(1 - p)], p clipped to [1e-7, 1 - 1e-7]
  Brier = mean (p - y)^2
  ECE   = sum_m |B_m|/n * |acc(B_m) - conf(B_m)| with M equal-width bins
          (m-1)/M < c <= m/M. Top-label: c = max(p, 1-p), acc = 1(yhat = y),
          yhat = 1(p >= 0.5). Class-1: c = p, acc = y.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

CLIP_EPS = 1e-7
ECE_BINS = 15


class OneClassError(ValueError):
    """Raised when a metric needs both classes and the input has only one."""


# ---------------------------------------------------------------- input checks

def _vec(x, name):
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    return a


def _labels(y):
    a = _vec(y, "labels")
    if a.size == 0:
        raise ValueError("labels is empty")
    if a.dtype == bool:
        return a.astype(np.int64)
    af = a.astype(float)
    if not np.all((af == 0) | (af == 1)):
        raise ValueError("labels must be 0 or 1")
    return af.astype(np.int64)


def _floats(x, name, n):
    a = _vec(x, name).astype(float)
    if a.size != n:
        raise ValueError(f"{name} has length {a.size}, labels have {n}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _probs(p, n):
    a = _floats(p, "probabilities", n)
    if a.min() < 0.0 or a.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    return a


def _weights(w, n):
    if w is None:
        return np.ones(n)
    a = _floats(w, "weights", n)
    if a.min() < 0.0:
        raise ValueError("weights must be non-negative")
    return a


def _prepare(y, s, weights, prob=False):
    y = _labels(y)
    s = _probs(s, y.size) if prob else _floats(s, "scores", y.size)
    return y, s, _weights(weights, y.size)


# ----------------------------------------------------------------- AUC and EER

def _block_weights(inv, n_blocks, y, w):
    """Positive and negative weight in each distinct-score block (ascending)."""
    wy = w * y
    wp = np.bincount(inv, weights=wy, minlength=n_blocks)
    wn = np.bincount(inv, weights=w - wy, minlength=n_blocks)
    return wp, wn


def _auc_from_blocks(wp, wn):
    tp, tn = wp.sum(), wn.sum()
    if tp <= 0 or tn <= 0:
        raise OneClassError("AUC needs at least one fake and one real sample")
    below = np.cumsum(wn) - wn  # negative weight strictly below each block
    return float(np.dot(wp, below + 0.5 * wn) / (tp * tn))


def _eer_from_blocks(wp, wn):
    tp, tn = wp.sum(), wn.sum()
    if tp <= 0 or tn <= 0:
        raise OneClassError("EER needs at least one fake and one real sample")
    # thresholds in descending order: +inf first, then each distinct score
    cum_n = np.concatenate(([0.0], np.cumsum(wn[::-1])))
    cum_p = np.concatenate(([0.0], np.cumsum(wp[::-1])))
    fpr = cum_n / tn
    fnr = (tp - cum_p) / tp
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5 * (fpr[i] + fnr[i]))


def roc_auc(y, score, weights=None) -> float:
    """Area under the ROC curve, Mann-Whitney form, ties count one half."""
    y, s, w = _prepare(y, score, weights)
    _, inv = np.unique(s, return_inverse=True)
    wp, wn = _block_weights(inv, inv.max() + 1, y, w)
    return _auc_from_blocks(wp, wn)


def eer(y, score, weights=None) -> float:
    """Equal error rate by the nearest-crossing rule of MATH_NOTES section 3."""
    y, s, w = _prepare(y, score, weights)
    _, inv = np.unique(s, return_inverse=True)
    wp, wn = _block_weights(inv, inv.max() + 1, y, w)
    return _eer_from_blocks(wp, wn)


# ------------------------------------------------------- threshold-based rates

def _confusion(y, s, w, threshold):
    pred = s >= threshold
    pos = y == 1
    tp = float(w[pos & pred].sum())
    fn = float(w[pos & ~pred].sum())
    fp = float(w[~pos & pred].sum())
    tn = float(w[~pos & ~pred].sum())
    return tp, fp, tn, fn


def _ba(tp, fp, tn, fn):
    if tp + fn <= 0 or tn + fp <= 0:
        raise OneClassError("balanced accuracy needs both classes")
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def balanced_accuracy(y, score, threshold, weights=None) -> float:
    """(TPR + TNR) / 2, predicting fake when score >= threshold."""
    y, s, w = _prepare(y, score, weights)
    return _ba(*_confusion(y, s, w, threshold))


def precision_recall(y, score, threshold, weights=None):
    """(precision, recall) = (TP / (TP + FP), TP / (TP + FN)).

    Precision is nan when nothing is predicted fake. Recall needs a fake sample.
    """
    y, s, w = _prepare(y, score, weights)
    tp, fp, tn, fn = _confusion(y, s, w, threshold)
    if tp + fn <= 0:
        raise OneClassError("recall needs at least one fake sample")
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    return prec, tp / (tp + fn)


def accuracy_at(y, p, threshold=0.5, weights=None) -> float:
    """Plain accuracy at a fixed threshold (reference only, default 0.5)."""
    y, p, w = _prepare(y, p, weights, prob=True)
    ok = ((p >= threshold).astype(np.int64) == y).astype(float)
    return float(np.dot(w, ok) / w.sum())


# --------------------------------------------------------- probability metrics

def _log_loss_vec(y, p, eps):
    q = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(q) + (1 - y) * np.log1p(-q))


def log_loss(y, p, weights=None, eps=CLIP_EPS) -> float:
    """Mean negative log likelihood, p clipped to [eps, 1 - eps]."""
    y, p, w = _prepare(y, p, weights, prob=True)
    return float(np.dot(w, _log_loss_vec(y, p, eps)) / w.sum())


def brier(y, p, weights=None) -> float:
    """Mean squared error of the probability of fake."""
    y, p, w = _prepare(y, p, weights, prob=True)
    return float(np.dot(w, (p - y) ** 2) / w.sum())


def _ece_parts(y, p, n_bins, top_label):
    """Per-sample bin index and (accuracy - confidence) term."""
    if top_label:
        conf = np.maximum(p, 1.0 - p)
        acc = ((p >= 0.5).astype(np.int64) == y).astype(float)
    else:
        conf, acc = p, y.astype(float)
    # bin m is ((m-1)/M, m/M]; rounding guards float noise such as 0.6 * 15
    idx = np.clip(np.ceil(np.round(conf * n_bins, 9)).astype(np.int64) - 1, 0, n_bins - 1)
    return idx, acc - conf


def _ece(idx, gap, w, n_bins):
    # |B_m|/n * |acc - conf| = |sum_{i in B_m} (acc_i - conf_i)| / n
    per_bin = np.bincount(idx, weights=w * gap, minlength=n_bins)
    return float(np.abs(per_bin).sum() / w.sum())


def ece(y, p, n_bins=ECE_BINS, weights=None) -> float:
    """Top-label ECE (Guo et al.), equal-width bins. The protocol version."""
    y, p, w = _prepare(y, p, weights, prob=True)
    idx, gap = _ece_parts(y, p, n_bins, True)
    return _ece(idx, gap, w, n_bins)


def ece_class1(y, p, n_bins=ECE_BINS, weights=None) -> float:
    """ECE of the probability of the fake class (not the protocol version)."""
    y, p, w = _prepare(y, p, weights, prob=True)
    idx, gap = _ece_parts(y, p, n_bins, False)
    return _ece(idx, gap, w, n_bins)


# ---------------------------------------------------------- weight evaluators

def make_evaluator(name: str, y, s, **kw) -> Callable[[np.ndarray], float]:
    """Return f(weights) -> metric for fixed labels and scores.

    Inputs are validated once and everything that does not depend on the
    weights is precomputed, so f is O(n) numpy with no sorting. The bootstrap
    calls f once per resample. Names: auc, eer, balanced_accuracy (threshold=),
    precision (threshold=), recall (threshold=), accuracy (threshold=0.5),
    log_loss, brier, ece, ece_class1 (n_bins=).
    """
    y = _labels(y)
    n = y.size
    if name in ("auc", "eer"):
        s = _floats(s, "scores", n)
        _, inv = np.unique(s, return_inverse=True)
        nb = int(inv.max()) + 1
        core = _auc_from_blocks if name == "auc" else _eer_from_blocks
        return lambda w: core(*_block_weights(inv, nb, y, w))
    if name in ("balanced_accuracy", "precision", "recall"):
        s = _floats(s, "scores", n)
        pred = s >= kw["threshold"]
        pos = y == 1
        masks = [(pos & pred), (~pos & pred), (~pos & ~pred), (pos & ~pred)]
        m = [x.astype(float) for x in masks]

        def conf(w):
            return tuple(float(np.dot(w, x)) for x in m)

        if name == "balanced_accuracy":
            return lambda w: _ba(*conf(w))

        def pr(w):
            tp, fp, _, fn = conf(w)
            if name == "recall":
                if tp + fn <= 0:
                    raise OneClassError("recall needs at least one fake sample")
                return tp / (tp + fn)
            return tp / (tp + fp) if tp + fp > 0 else float("nan")

        return pr
    # remaining metrics are weighted means of a fixed per-sample vector
    p = _probs(s, n)
    if name == "accuracy":
        v = ((p >= kw.get("threshold", 0.5)).astype(np.int64) == y).astype(float)
    elif name == "log_loss":
        v = _log_loss_vec(y, p, kw.get("eps", CLIP_EPS))
    elif name == "brier":
        v = (p - y) ** 2
    elif name in ("ece", "ece_class1"):
        nbins = kw.get("n_bins", ECE_BINS)
        idx, gap = _ece_parts(y, p, nbins, name == "ece")
        return lambda w: _ece(idx, gap, w, nbins)
    else:
        raise ValueError(f"unknown metric {name!r}")
    return lambda w: float(np.dot(w, v) / w.sum())
