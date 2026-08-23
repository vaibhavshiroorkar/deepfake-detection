from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from deepfake_detection.data.manifest import ClipRecord


@dataclass(frozen=True, slots=True)
class GroupFold:
    train_sources: tuple[str, ...]
    holdout_sources: tuple[str, ...]
    train_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]


def build_group_folds(
    records: Sequence[ClipRecord],
    *,
    folds: int,
    seed: int,
) -> tuple[GroupFold, ...]:
    sources = sorted({record.source for record in records})
    if folds < 2 or folds > len(sources):
        raise ValueError("Fold count must be between two and the source count")
    metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        demographics = (record.race, record.gender)
        existing = metadata.setdefault(record.source, demographics)
        if existing != demographics:
            raise ValueError(f"Source {record.source} has conflicting demographics")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source in sources:
        strata[metadata[source]].append(source)

    # This generator makes fold assignment repeatable. It protects no secrets.
    generator = random.Random(seed)  # noqa: S311
    holdout_groups: list[list[str]] = [[] for _ in range(folds)]
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        generator.shuffle(members)
        for source in members:
            smallest = min(len(group) for group in holdout_groups)
            candidates = [
                index
                for index, group in enumerate(holdout_groups)
                if len(group) == smallest
            ]
            holdout_groups[generator.choice(candidates)].append(source)

    result: list[GroupFold] = []
    all_sources = set(sources)
    for holdout in holdout_groups:
        holdout_set = set(holdout)
        train_set = all_sources - holdout_set
        result.append(
            GroupFold(
                train_sources=tuple(sorted(train_set)),
                holdout_sources=tuple(sorted(holdout_set)),
                train_indices=tuple(
                    index
                    for index, record in enumerate(records)
                    if record.source in train_set
                ),
                holdout_indices=tuple(
                    index
                    for index, record in enumerate(records)
                    if record.source in holdout_set
                ),
            )
        )
    return tuple(result)
