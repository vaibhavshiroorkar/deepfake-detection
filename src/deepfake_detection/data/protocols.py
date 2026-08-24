from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from .manifest import ClipRecord

Split = dict[str, tuple[ClipRecord, ...]]


@dataclass(frozen=True, slots=True)
class SplitAudit:
    source_overlaps: dict[tuple[str, str], set[str]]
    all_identity_overlaps: dict[tuple[str, str], set[str]]
    method_counts: dict[str, dict[str, int]]


def build_source_split(
    records: Sequence[ClipRecord],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Split:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")

    sources = sorted({record.source for record in records})
    if len(sources) < 3:
        raise ValueError("At least three source identities are required")

    source_metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        demographics = (record.race, record.gender)
        existing = source_metadata.setdefault(record.source, demographics)
        if existing != demographics:
            raise ValueError(f"Source {record.source} has conflicting demographics")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source in sources:
        groups[source_metadata[source]].append(source)

    names = ("train", "val", "test")
    ideal_counts = [len(sources) * ratio for ratio in ratios]
    count_values = [1, 1, 1]
    while sum(count_values) < len(sources):
        index = max(
            range(3),
            key=lambda candidate: (
                ideal_counts[candidate] - count_values[candidate],
                ratios[candidate],
            ),
        )
        count_values[index] += 1
    target_counts = dict(zip(names, count_values, strict=True))
    assigned: dict[str, list[str]] = {name: [] for name in names}
    leftovers: list[tuple[tuple[str, str], str]] = []
    extra_by_group: Counter[tuple[tuple[str, str], str]] = Counter()
    # Research splits need repeatability, not cryptographic randomness.
    generator = random.Random(seed)  # noqa: S311

    for stratum in sorted(groups):
        members = sorted(groups[stratum])
        generator.shuffle(members)
        cursor = 0
        for name, ratio in zip(names, ratios, strict=True):
            capacity = target_counts[name] - len(assigned[name])
            count = min(int(len(members) * ratio), max(0, capacity))
            assigned[name].extend(members[cursor : cursor + count])
            cursor += count
        leftovers.extend((stratum, source) for source in members[cursor:])

    generator.shuffle(leftovers)
    remaining = {name: target_counts[name] - len(assigned[name]) for name in names}
    for stratum, source in leftovers:
        available = [name for name in names if remaining[name] > 0]
        if not available:
            raise RuntimeError("Split allocation exhausted its capacity")
        stratum_size = len(groups[stratum])
        name = max(
            available,
            key=lambda candidate: (
                stratum_size * ratios[names.index(candidate)]
                - int(stratum_size * ratios[names.index(candidate)])
                - extra_by_group[(stratum, candidate)],
                remaining[candidate],
            ),
        )
        assigned[name].append(source)
        remaining[name] -= 1
        extra_by_group[(stratum, name)] += 1

    train_sources = set(assigned["train"])
    val_sources = set(assigned["val"])

    split: Split = {"train": (), "val": (), "test": ()}
    grouped: dict[str, list[ClipRecord]] = {name: [] for name in split}
    for record in records:
        if record.source in train_sources:
            grouped["train"].append(record)
        elif record.source in val_sources:
            grouped["val"].append(record)
        else:
            grouped["test"].append(record)
    return {name: tuple(rows) for name, rows in grouped.items()}


def audit_split(split: Mapping[str, Sequence[ClipRecord]]) -> SplitAudit:
    sources = {name: {record.source for record in rows} for name, rows in split.items()}
    identities = {
        name: {
            identity for record in rows for identity in (record.source, *record.targets)
        }
        for name, rows in split.items()
    }
    source_overlaps: dict[tuple[str, str], set[str]] = {}
    all_identity_overlaps: dict[tuple[str, str], set[str]] = {}
    for left, right in combinations(split, 2):
        if overlap := sources[left] & sources[right]:
            source_overlaps[(left, right)] = overlap
        if overlap := identities[left] & identities[right]:
            all_identity_overlaps[(left, right)] = overlap
    method_counts = {
        name: dict(Counter(record.method for record in rows))
        for name, rows in split.items()
    }
    return SplitAudit(
        source_overlaps=source_overlaps,
        all_identity_overlaps=all_identity_overlaps,
        method_counts=method_counts,
    )


def identity_strict_subset(
    split: Mapping[str, Sequence[ClipRecord]],
) -> Split:
    source_owner = {
        record.source: name for name, rows in split.items() for record in rows
    }
    strict: dict[str, list[ClipRecord]] = {name: [] for name in split}
    for name, rows in split.items():
        for record in rows:
            if all(source_owner.get(target) == name for target in record.targets):
                strict[name].append(record)
    return {name: tuple(rows) for name, rows in strict.items()}


def split_hash(split: Mapping[str, Sequence[ClipRecord]]) -> str:
    assignments = sorted(
        (
            name,
            record.clip_id,
            record.source,
            record.targets,
        )
        for name, rows in split.items()
        for record in rows
    )
    encoded = json.dumps(assignments, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_method_holdout_protocol(
    split: Mapping[str, Sequence[ClipRecord]],
    *,
    heldout_methods: set[str],
) -> Split:
    if not heldout_methods:
        raise ValueError("At least one method must be held out")
    missing_partitions = {"train", "val", "test"} - set(split)
    if missing_partitions:
        raise ValueError(
            f"Method holdout is missing partitions: {', '.join(sorted(missing_partitions))}"
        )
    return {
        "train": tuple(
            record for record in split["train"] if record.method not in heldout_methods
        ),
        "val": tuple(
            record for record in split["val"] if record.method not in heldout_methods
        ),
        "test": tuple(
            record
            for record in split["test"]
            if not record.clip_fake or record.method in heldout_methods
        ),
    }
