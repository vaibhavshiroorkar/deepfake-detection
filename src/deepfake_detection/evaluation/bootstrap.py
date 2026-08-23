from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, TypeVar

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import EvaluationItem, binary_metrics


class HasSourceIdentity(Protocol):
    source_identity: str


Item = TypeVar("Item", bound=HasSourceIdentity)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    successful_samples: int


@dataclass(frozen=True, slots=True)
class PairedPrediction:
    label: int
    source_identity: str
    left_probability: float
    right_probability: float


def _clusters(items: Sequence[Item]) -> dict[str, list[Item]]:
    grouped: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        grouped[item.source_identity].append(item)
    return grouped


def cluster_bootstrap_interval(
    items: Sequence[Item],
    statistic: Callable[[Sequence[Item]], float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    if not items or samples <= 0:
        raise ValueError("Bootstrap requires items and a positive sample count")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be in (0, 1)")
    grouped = _clusters(items)
    identities = tuple(sorted(grouped))
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = generator.choice(identities, size=len(identities), replace=True)
        resampled = [item for identity in selected for item in grouped[str(identity)]]
        values.append(float(statistic(resampled)))
    tail = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=float(statistic(items)),
        lower=float(np.quantile(values, tail)),
        upper=float(np.quantile(values, 1 - tail)),
        successful_samples=len(values),
    )


def paired_auc_difference(
    predictions: Sequence[PairedPrediction],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    if set(item.label for item in predictions) != {0, 1}:
        raise ValueError("Paired AUC requires both classes")

    def difference(items: Sequence[PairedPrediction]) -> float:
        labels = [item.label for item in items]
        return float(
            roc_auc_score(labels, [item.left_probability for item in items])
            - roc_auc_score(labels, [item.right_probability for item in items])
        )

    grouped = _clusters(predictions)
    identities = tuple(sorted(grouped))
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = generator.choice(identities, size=len(identities), replace=True)
        resampled = [item for identity in selected for item in grouped[str(identity)]]
        if set(item.label for item in resampled) == {0, 1}:
            values.append(difference(resampled))
    if not values:
        raise ValueError("No valid paired bootstrap samples contained both classes")
    tail = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=difference(predictions),
        lower=float(np.quantile(values, tail)),
        upper=float(np.quantile(values, 1 - tail)),
        successful_samples=len(values),
    )


def bootstrap_binary_metrics(
    items: Sequence[EvaluationItem],
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, BootstrapInterval]:
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")

    def calculate(sample: Sequence[EvaluationItem]) -> dict[str, float]:
        scored = [item for item in sample if item.probability is not None]
        labels = [item.label for item in scored]
        if set(labels) != {0, 1}:
            raise ValueError("Bootstrap sample does not contain both scored classes")
        return asdict(
            binary_metrics(
                labels=labels,
                probabilities=[float(item.probability) for item in scored],
                threshold=threshold,
            )
        )

    estimate = calculate(items)
    grouped = _clusters(items)
    identities = tuple(sorted(grouped))
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in estimate}
    for _ in range(samples):
        selected = generator.choice(identities, size=len(identities), replace=True)
        resampled = [item for identity in selected for item in grouped[str(identity)]]
        try:
            metrics = calculate(resampled)
        except ValueError:
            continue
        for name, value in metrics.items():
            values[name].append(float(value))
    if not values["roc_auc"]:
        raise ValueError("No valid bootstrap samples contained both scored classes")
    tail = (1 - confidence) / 2
    return {
        name: BootstrapInterval(
            estimate=float(estimate[name]),
            lower=float(np.quantile(samples_for_metric, tail)),
            upper=float(np.quantile(samples_for_metric, 1 - tail)),
            successful_samples=len(samples_for_metric),
        )
        for name, samples_for_metric in values.items()
    }
