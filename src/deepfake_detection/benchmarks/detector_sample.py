from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, TypeVar

import numpy as np

from deepfake_detection.data.protocols import split_hash as protocol_split_hash

MINIMUM_REVIEW_FRAMES = 625
MINIMUM_REVIEW_CLIPS = 125
MINIMUM_COMPARISON_FRAMES = 500
MINIMUM_COMPARISON_CLIPS = 100
MINIMUM_REVIEW_SOURCES = 5
MINIMUM_DOUBLE_REVIEW_FRACTION = 0.10

SplitRole = Literal["calibration", "comparison"]


class ManifestRecord(Protocol):
    clip_id: str
    dataset: str
    manipulation_type: str
    method: str
    source: str
    targets: tuple[str, ...]
    race: str
    gender: str


@dataclass(frozen=True, slots=True)
class ReviewFrame:
    frame_id: str
    dataset: str
    clip_id: str
    source_hash: str
    split_hash: str
    timestamp_sec: float
    frame_sha256: str
    width: int
    height: int
    split_role: SplitRole
    double_review: bool
    manipulation_type: str
    method: str
    race: str
    gender: str

    def __post_init__(self) -> None:
        for name in (
            "frame_id",
            "dataset",
            "clip_id",
            "manipulation_type",
            "method",
            "race",
            "gender",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be blank")
        if _looks_absolute(self.clip_id):
            raise ValueError("clip_id cannot contain an absolute path")
        _validate_sha256("source_hash", self.source_hash)
        _validate_sha256("split_hash", self.split_hash)
        _validate_sha256("frame_sha256", self.frame_sha256)
        if not math.isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be finite and nonnegative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Review frame dimensions must be positive")
        if self.split_role not in {"calibration", "comparison"}:
            raise ValueError("split_role must be calibration or comparison")
        if not isinstance(self.double_review, bool):
            raise TypeError("double_review must be a boolean")


_T = TypeVar("_T")


def _looks_absolute(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _validate_sha256(name: str, value: str) -> None:
    valid = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _stable_rank(value: str, *, seed: int, namespace: str) -> bytes:
    return hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode()).digest()


def _balanced_order(
    items: Sequence[_T],
    *,
    stratum: Callable[[_T], tuple[str, ...]],
    identity: Callable[[_T], str],
    seed: int,
    namespace: str,
) -> tuple[_T, ...]:
    grouped: dict[tuple[str, ...], list[_T]] = defaultdict(list)
    for item in items:
        grouped[stratum(item)].append(item)
    queues: dict[tuple[str, ...], deque[_T]] = {}
    for key, members in grouped.items():
        members.sort(
            key=lambda item: (
                _stable_rank(
                    identity(item),
                    seed=seed,
                    namespace=f"{namespace}:member",
                ),
                identity(item),
            )
        )
        queues[key] = deque(members)
    strata = sorted(
        queues,
        key=lambda key: (
            _stable_rank(
                json.dumps(key),
                seed=seed,
                namespace=f"{namespace}:stratum",
            ),
            key,
        ),
    )
    ordered: list[_T] = []
    while any(queues.values()):
        for key in strata:
            if queues[key]:
                ordered.append(queues[key].popleft())
    return tuple(ordered)


def _require_record_fields(record: ManifestRecord) -> None:
    for field in (
        "clip_id",
        "dataset",
        "manipulation_type",
        "method",
        "source",
        "race",
        "gender",
    ):
        value = getattr(record, field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Manifest record {field} must be a nonblank string")
    if _looks_absolute(record.clip_id):
        raise ValueError("Manifest clip_id cannot contain an absolute path")


def _frame_hash(frame: np.ndarray) -> str:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame_reader must return a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Review source frames must be nonempty three-channel images")
    contiguous = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_hash(record: ManifestRecord) -> str:
    return hashlib.sha256(f"{record.dataset}\0{record.source}".encode()).hexdigest()


def _source_stratum(records: Sequence[ManifestRecord]) -> tuple[str, ...]:
    first = records[0]
    signatures = sorted(
        {f"{record.manipulation_type}\0{record.method}" for record in records}
    )
    return first.race, first.gender, *signatures


def _select_clips(
    records: Sequence[ManifestRecord],
    *,
    clip_count: int,
    seed: int,
) -> tuple[ManifestRecord, ...]:
    by_source: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)
    source_rows = tuple(
        tuple(sorted(rows, key=lambda row: row.clip_id))
        for _, rows in sorted(by_source.items())
    )
    ordered_sources = _balanced_order(
        source_rows,
        stratum=_source_stratum,
        identity=lambda rows: rows[0].source,
        seed=seed,
        namespace="review-sources",
    )
    selected_sources = ordered_sources[: min(len(ordered_sources), clip_count)]
    selected_names = {rows[0].source for rows in selected_sources}
    candidates = tuple(record for record in records if record.source in selected_names)
    ordered_clips = _balanced_order(
        candidates,
        stratum=lambda row: (
            row.manipulation_type,
            row.method,
            row.race,
            row.gender,
        ),
        identity=lambda row: row.clip_id,
        seed=seed,
        namespace="review-clips",
    )

    selected: list[ManifestRecord] = []
    represented_sources: set[str] = set()
    for record in ordered_clips:
        if record.source not in represented_sources:
            selected.append(record)
            represented_sources.add(record.source)
    selected_ids = {record.clip_id for record in selected}
    for record in ordered_clips:
        if len(selected) >= clip_count:
            break
        if record.clip_id not in selected_ids:
            selected.append(record)
            selected_ids.add(record.clip_id)
    return tuple(selected)


def _timestamp(
    clip_id: str,
    *,
    index: int,
    count: int,
    duration_sec: float,
    seed: int,
) -> float:
    digest = _stable_rank(
        f"{clip_id}\0{index}", seed=seed, namespace="review-timestamps"
    )
    unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    fraction = (index + 0.25 + 0.5 * unit) / count
    return duration_sec * fraction


def _split_roles(
    selected: Sequence[ManifestRecord], *, seed: int
) -> dict[str, SplitRole]:
    sources = sorted({record.source for record in selected})
    ordered = sorted(
        sources,
        key=lambda source: (
            _stable_rank(source, seed=seed, namespace="review-split"),
            source,
        ),
    )
    calibration_count = max(
        1,
        min(len(ordered) - 1, int(len(ordered) * 0.2 + 0.5)),
    )
    calibration = set(ordered[:calibration_count])
    return {
        source: "calibration" if source in calibration else "comparison"
        for source in sources
    }


def build_review_sample(
    records: Sequence[ManifestRecord],
    *,
    partition: str,
    frozen_split: Mapping[str, Sequence[ManifestRecord]],
    expected_split_hash: str,
    duration_reader: Callable[[ManifestRecord], float],
    frame_reader: Callable[[ManifestRecord, float], np.ndarray],
    frame_count: int = MINIMUM_REVIEW_FRAMES,
    clip_count: int = MINIMUM_REVIEW_CLIPS,
    double_review_fraction: float = MINIMUM_DOUBLE_REVIEW_FRACTION,
    seed: int = 17,
) -> tuple[ReviewFrame, ...]:
    """Build a source-first review sample from an explicit training manifest."""

    if partition != "train":
        raise ValueError("Detector review accepts only an explicit training partition")
    _validate_sha256("expected_split_hash", expected_split_hash)
    if set(frozen_split) != {"train", "val", "test"}:
        raise ValueError("Frozen split must contain train, val, and test partitions")
    observed_split_hash = protocol_split_hash(frozen_split)  # type: ignore[arg-type]
    if observed_split_hash != expected_split_hash:
        raise ValueError("Frozen split artifact does not match its expected split hash")
    frozen_train = {(record.clip_id, record.source) for record in frozen_split["train"]}
    frozen_non_train_sources = {
        record.source for name in ("val", "test") for record in frozen_split[name]
    }
    if any(
        (record.clip_id, record.source) not in frozen_train
        or record.source in frozen_non_train_sources
        for record in records
    ):
        raise ValueError(
            "Detector review records are outside the frozen training split"
        )
    if frame_count < MINIMUM_REVIEW_FRAMES:
        raise ValueError(
            f"Detector review requires at least {MINIMUM_REVIEW_FRAMES} frames"
        )
    if clip_count < MINIMUM_REVIEW_CLIPS:
        raise ValueError(
            f"Detector review requires at least {MINIMUM_REVIEW_CLIPS} unique clips"
        )
    if frame_count < clip_count:
        raise ValueError("frame_count must be at least clip_count")
    if (
        not math.isfinite(double_review_fraction)
        or not MINIMUM_DOUBLE_REVIEW_FRACTION <= double_review_fraction <= 1
    ):
        raise ValueError("Double-review fraction must be in [0.10, 1]")
    canonical_records = tuple(records)
    for record in canonical_records:
        _require_record_fields(record)
    clip_ids = [record.clip_id for record in canonical_records]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Training manifest contains duplicate clip identifiers")
    if len(canonical_records) < clip_count:
        raise ValueError(
            f"Detector review requires at least {clip_count} unique clips; "
            f"found {len(canonical_records)}"
        )
    sources = {record.source for record in canonical_records}
    if len(sources) < MINIMUM_REVIEW_SOURCES:
        raise ValueError("Detector review requires at least five source identities")
    source_demographics: dict[str, tuple[str, str]] = {}
    for record in canonical_records:
        demographics = (record.race, record.gender)
        previous = source_demographics.setdefault(record.source, demographics)
        if previous != demographics:
            raise ValueError(
                f"Source {record.source} has conflicting demographic metadata"
            )

    selected = _select_clips(canonical_records, clip_count=clip_count, seed=seed)
    if len(selected) != clip_count:
        raise ValueError(
            f"Detector review selected {len(selected)} clips, expected {clip_count}"
        )
    roles = _split_roles(selected, seed=seed)
    base_count, extra = divmod(frame_count, clip_count)
    built: list[ReviewFrame] = []
    seen_frame_hashes: set[str] = set()
    for clip_index, record in enumerate(selected):
        count = base_count + (1 if clip_index < extra else 0)
        duration_sec = float(duration_reader(record))
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError(f"Clip {record.clip_id} has no positive finite duration")
        source_hash = _source_hash(record)
        for index in range(count):
            timestamp_sec = _timestamp(
                record.clip_id,
                index=index,
                count=count,
                duration_sec=duration_sec,
                seed=seed,
            )
            frame = frame_reader(record, timestamp_sec)
            frame_sha256 = _frame_hash(frame)
            if frame_sha256 in seen_frame_hashes:
                raise ValueError(
                    "Review sample contains a duplicate source frame; "
                    "the sample was not silently shrunk"
                )
            seen_frame_hashes.add(frame_sha256)
            frame_identity = json.dumps(
                (
                    record.dataset,
                    record.clip_id,
                    source_hash,
                    format(timestamp_sec, ".17g"),
                    frame_sha256,
                ),
                separators=(",", ":"),
            )
            frame_id = (
                "frame-"
                + hashlib.sha256(frame_identity.encode("utf-8")).hexdigest()[:24]
            )
            built.append(
                ReviewFrame(
                    frame_id=frame_id,
                    dataset=record.dataset,
                    clip_id=record.clip_id,
                    source_hash=source_hash,
                    split_hash=observed_split_hash,
                    timestamp_sec=timestamp_sec,
                    frame_sha256=frame_sha256,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    split_role=roles[record.source],
                    double_review=False,
                    manipulation_type=record.manipulation_type,
                    method=record.method,
                    race=record.race,
                    gender=record.gender,
                )
            )

    double_count = math.ceil(frame_count * double_review_fraction)
    double_ids = {
        frame.frame_id
        for frame in sorted(
            built,
            key=lambda frame: (
                _stable_rank(
                    frame.frame_id,
                    seed=seed,
                    namespace="double-review",
                ),
                frame.frame_id,
            ),
        )[:double_count]
    }
    result = tuple(
        replace(frame, double_review=frame.frame_id in double_ids) for frame in built
    )
    comparison = tuple(frame for frame in result if frame.split_role == "comparison")
    if len(comparison) < MINIMUM_COMPARISON_FRAMES:
        raise ValueError(
            "Detector review requires at least 500 comparison frames after calibration"
        )
    if len({frame.clip_id for frame in comparison}) < MINIMUM_COMPARISON_CLIPS:
        raise ValueError(
            "Detector review requires at least 100 comparison clips after calibration"
        )
    return result


def _review_frame_from_mapping(row: dict[str, object]) -> ReviewFrame:
    expected = set(ReviewFrame.__dataclass_fields__)
    if set(row) != expected:
        raise ValueError("Review sample row has missing or unexpected fields")
    if not isinstance(row["double_review"], bool):
        raise ValueError("double_review must be a boolean")
    return ReviewFrame(
        frame_id=str(row["frame_id"]),
        dataset=str(row["dataset"]),
        clip_id=str(row["clip_id"]),
        source_hash=str(row["source_hash"]),
        split_hash=str(row["split_hash"]),
        timestamp_sec=float(row["timestamp_sec"]),
        frame_sha256=str(row["frame_sha256"]),
        width=int(row["width"]),
        height=int(row["height"]),
        split_role=str(row["split_role"]),  # type: ignore[arg-type]
        double_review=row["double_review"],
        manipulation_type=str(row["manipulation_type"]),
        method=str(row["method"]),
        race=str(row["race"]),
        gender=str(row["gender"]),
    )


def write_review_sample(frames: Sequence[ReviewFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(
                json.dumps(asdict(frame), sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(path)


def read_review_sample(path: Path) -> tuple[ReviewFrame, ...]:
    frames: list[ReviewFrame] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Review sample line {line_number} is blank")
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
                frames.append(_review_frame_from_mapping(row))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid review sample line {line_number}: {error}"
                ) from error
    return tuple(frames)


def review_sample_sha256(frames: Sequence[ReviewFrame]) -> str:
    canonical = sorted(
        (asdict(frame) for frame in frames),
        key=lambda row: str(row["frame_id"]),
    )
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
