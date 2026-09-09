"""Datasets that may only ever be evaluated on, enforced rather than documented.

`docs/data-card.md` has said since the project started that MNW is
evaluation-only: it cannot be used for training, validation, model selection, or
threshold selection, both because its license forbids it and because a locked
external test set stops being a test once you tune against it. Until now that
rule lived only in prose, so nothing stopped `ddf train visual --train-manifest
mnw.csv` from quietly running.

The check is on the dataset name, which every manifest row carries, so it fires
no matter which path reaches the data.
"""

from __future__ import annotations

from collections.abc import Iterable

# Held apart from data/mnw.py so importing the guard cannot pull in pandas, and
# so adding a second locked dataset later does not mean touching an adapter.
EVALUATION_ONLY_DATASETS = frozenset({"MNW"})


class EvaluationOnlyDatasetError(RuntimeError):
    """Raised when a locked dataset is used for anything but evaluation."""


def reject_evaluation_only(dataset: str, *, operation: str) -> None:
    """Refuse an operation that would let a locked dataset influence a model.

    `operation` names what was attempted, so the message says which rule broke
    rather than only that one did.
    """
    if dataset in EVALUATION_ONLY_DATASETS:
        raise EvaluationOnlyDatasetError(
            f"{dataset} is evaluation-only and cannot be used for {operation}. "
            "It may only be scored through an external-role evaluation. See "
            "docs/data-card.md."
        )


def reject_evaluation_only_datasets(
    datasets: Iterable[str], *, operation: str
) -> None:
    """The same rule for a mixed set of datasets, so a blend cannot smuggle one in."""
    for dataset in sorted(set(datasets)):
        reject_evaluation_only(dataset, operation=operation)
