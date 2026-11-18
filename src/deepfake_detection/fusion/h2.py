"""H2 of P-001: late fusion of the visual, audio and sync paths against visual-only.

Amendment A8 fixes every choice here before any branch model existed:
logistic regression on standardized branch logits, fit on FakeAVCeleb
validation; its validation scores come from 5-fold cross-fitting grouped by
identity; the model applied elsewhere is refit on all of validation. The test
statistic is the mean over seeds of AUC(fusion) - AUC(visual), with one
identity bootstrap shared by all seeds, and Holm across the H2 targets.

The label is the clip label (fake if either stream is fake). The visual path
was trained on the video label, so on clips with only fake audio it is
expected to miss; that is part of what fusion is supposed to add.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BRANCHES = ("visual", "audio", "sync")
EPS = 1e-6


def logit(probability: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class Fusion:
    """A fitted fusion: standardization of each branch logit, then logistic regression."""

    mean: np.ndarray
    scale: np.ndarray
    model: LogisticRegression

    def score(self, logits: np.ndarray) -> np.ndarray:
        return self.model.predict_proba((logits - self.mean) / self.scale)[:, 1]


def fit_fusion(logits: np.ndarray, labels: Sequence[int]) -> Fusion:
    mean = logits.mean(axis=0)
    scale = logits.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(
        C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000
    )
    model.fit((logits - mean) / scale, np.asarray(labels, dtype=int))
    return Fusion(mean, scale, model)


def cross_fit_scores(
    logits: np.ndarray, labels: Sequence[int], groups: Sequence[str], folds: int = 5
) -> np.ndarray:
    """Out-of-fold fusion scores: each clip is scored by a model that never saw its identity."""
    labels = np.asarray(labels, dtype=int)
    scores = np.empty(len(labels))
    for train, held in GroupKFold(n_splits=folds).split(logits, labels, groups):
        scores[held] = fit_fusion(logits[train], labels[train]).score(logits[held])
    return scores


@dataclass(frozen=True)
class SeedPair:
    """One seed's fusion and visual-only scores on the same clips, with labels and identities."""

    labels: np.ndarray
    groups: np.ndarray
    fusion: np.ndarray
    visual: np.ndarray


def _difference(pair: SeedPair, index: np.ndarray) -> float:
    labels = pair.labels[index]
    if labels.min() == labels.max():
        raise ValueError("one class")
    return float(
        roc_auc_score(labels, pair.fusion[index])
        - roc_auc_score(labels, pair.visual[index])
    )


def mean_seed_difference(
    pairs: Sequence[SeedPair],
    *,
    samples: int = 1000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict:
    """Mean over seeds of AUC(fusion) - AUC(visual), with an identity bootstrap shared by every seed.

    All seeds must score the same clips in the same order, so one resample of
    identities picks the same clips for each seed.
    """
    groups = pairs[0].groups
    for pair in pairs[1:]:
        if not np.array_equal(pair.groups, groups) or not np.array_equal(
            pair.labels, pairs[0].labels
        ):
            raise ValueError("Seeds must score the same clips in the same order")
    identities = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in identities}
    everything = np.arange(len(groups))
    per_seed = [_difference(p, everything) for p in pairs]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = np.concatenate(
            [rows_of[g] for g in rng.choice(identities, size=len(identities))]
        )
        try:
            values.append(float(np.mean([_difference(p, index) for p in pairs])))
        except ValueError:  # a draw with one class: skipped and counted out
            continue
    values = np.asarray(values)
    tail = (1 - confidence) / 2
    return {
        "estimate": float(np.mean(per_seed)),
        "per_seed": per_seed,
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1 - tail)),
        "p_one_sided": float(np.mean(values <= 0)),
        "successful_samples": int(values.size),
        "clips": int(len(groups)),
        "identities": int(identities.size),
    }


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise level alpha."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    rejected: dict[str, bool] = {name: False for name in p_values}
    for rank, (name, p) in enumerate(ordered):
        if p > alpha / (len(ordered) - rank):
            break
        rejected[name] = True
    return rejected
