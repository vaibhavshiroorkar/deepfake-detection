from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    fpr: float
    fnr: float
    eer: float
    fpr_at_95_tpr: float
    brier: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    label: int
    probability: float | None
    source_identity: str
    method: str
    race: str = "unknown"
    gender: str = "unknown"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    total: int
    scored: int
    abstained: int
    coverage: float
    metrics: BinaryMetrics | None


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    balanced_accuracy: float


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> float:
    if bins <= 0:
        raise ValueError("Calibration bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probabilities, edges[1:-1])
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def binary_metrics(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    calibration_bins: int = 10,
) -> BinaryMetrics:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Labels and probabilities must have equal nonzero length")
    if set(labels) != {0, 1}:
        raise ValueError("Binary metrics require both classes")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    y_pred = (y_prob >= threshold).astype(np.int64)

    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (false_negative + true_positive)

    curve_fpr, curve_tpr, _ = roc_curve(y_true, y_prob)
    curve_fnr = 1 - curve_tpr
    eer_index = int(np.argmin(np.abs(curve_fpr - curve_fnr)))
    eer = float((curve_fpr[eer_index] + curve_fnr[eer_index]) / 2)
    eligible = np.flatnonzero(curve_tpr >= 0.95)
    fpr_at_95_tpr = float(curve_fpr[eligible].min()) if eligible.size else 1.0

    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        fpr=fpr,
        fnr=fnr,
        eer=eer,
        fpr_at_95_tpr=fpr_at_95_tpr,
        brier=float(brier_score_loss(y_true, y_prob)),
        expected_calibration_error=_expected_calibration_error(
            y_true, y_prob, calibration_bins
        ),
    )


def select_balanced_accuracy_threshold(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> ThresholdSelection:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Labels and probabilities must have equal nonzero length")
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")

    distinct = np.unique(y_prob)
    midpoints = (distinct[:-1] + distinct[1:]) / 2
    candidates = np.unique(np.concatenate(([0.0], midpoints, [1.0])))
    scored = [
        (
            float(balanced_accuracy_score(y_true, y_prob >= threshold)),
            float(threshold),
        )
        for threshold in candidates
    ]
    score, threshold = max(
        scored,
        key=lambda item: (item[0], -abs(item[1] - 0.5), -item[1]),
    )
    return ThresholdSelection(
        threshold=threshold,
        balanced_accuracy=score,
    )


def evaluate_items(
    items: Sequence[EvaluationItem],
    *,
    threshold: float,
) -> EvaluationReport:
    if not items:
        raise ValueError("Evaluation items cannot be empty")
    scored = [item for item in items if item.probability is not None]
    labels = [item.label for item in scored]
    metrics = None
    if set(labels) == {0, 1}:
        metrics = binary_metrics(
            labels=labels,
            probabilities=[float(item.probability) for item in scored],
            threshold=threshold,
        )
    return EvaluationReport(
        total=len(items),
        scored=len(scored),
        abstained=len(items) - len(scored),
        coverage=len(scored) / len(items),
        metrics=metrics,
    )


def per_method_metrics(
    items: Sequence[EvaluationItem],
    *,
    threshold: float,
) -> dict[str, EvaluationReport]:
    methods = sorted({item.method for item in items if item.label == 1})
    return {
        method: evaluate_items(
            [item for item in items if item.label == 0 or item.method == method],
            threshold=threshold,
        )
        for method in methods
    }


def subgroup_metrics(
    items: Sequence[EvaluationItem],
    *,
    attribute: str,
    threshold: float,
) -> dict[str, EvaluationReport]:
    if attribute not in {"race", "gender"}:
        raise ValueError("Subgroup attribute must be race or gender")
    groups = sorted({str(getattr(item, attribute)) for item in items})
    return {
        group: evaluate_items(
            [item for item in items if str(getattr(item, attribute)) == group],
            threshold=threshold,
        )
        for group in groups
    }
