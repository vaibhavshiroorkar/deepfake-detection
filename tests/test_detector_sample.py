from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.benchmarks import detector_sample
from deepfake_detection.benchmarks.detector_sample import (
    build_review_sample,
    read_review_sample,
    write_review_sample,
)
from deepfake_detection.data.protocols import split_hash


@dataclass(frozen=True)
class _Clip:
    clip_id: str
    dataset: str
    video_path: Path
    manipulation_type: str
    method: str
    source: str
    targets: tuple[str, ...]
    race: str
    gender: str


_MANIPULATIONS = (
    "RealVideo-RealAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-FakeAudio",
)


def _clips(count: int = 100) -> tuple[_Clip, ...]:
    return tuple(
        _Clip(
            clip_id=f"clip-{index:03d}",
            dataset="fixture",
            video_path=Path("private") / f"clip-{index:03d}.mp4",
            manipulation_type=_MANIPULATIONS[index % 4],
            method=f"method-{index % 4}",
            source=f"source-{index:03d}",
            targets=(),
            race=f"race-{index % 4}",
            gender=f"gender-{index % 2}",
        )
        for index in range(count)
    )


def _frozen_split(train: tuple[_Clip, ...]) -> dict[str, tuple[_Clip, ...]]:
    validation = replace(
        train[0],
        clip_id="validation-clip",
        source="validation-source",
    )
    test = replace(
        train[1],
        clip_id="test-clip",
        source="test-source",
    )
    return {"train": train, "val": (validation,), "test": (test,)}


def _frame(clip: _Clip, timestamp_sec: float) -> np.ndarray:
    payload = hashlib.sha256(
        f"{clip.clip_id}:{timestamp_sec:.9f}".encode("ascii")
    ).digest()
    return np.frombuffer(payload[:12], dtype=np.uint8).reshape(2, 2, 3)


def _build(clips: tuple[_Clip, ...], **values: object):
    frozen = (
        values.pop("frozen_split") if "frozen_split" in values else _frozen_split(clips)
    )
    expected_hash = values.pop("expected_split_hash", split_hash(frozen))
    return build_review_sample(
        clips,
        frozen_split=frozen,
        expected_split_hash=expected_hash,
        **values,
    )


def test_sample_requires_an_explicit_training_partition() -> None:
    with pytest.raises(ValueError, match="training partition"):
        _build(
            _clips(),
            partition="validation",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
        )


def test_sample_rejects_a_validation_record_relabelled_as_training() -> None:
    train = _clips(125)
    frozen = _frozen_split(train)
    validation = frozen["val"]

    with pytest.raises(ValueError, match="frozen training split"):
        _build(
            validation,
            partition="train",
            frozen_split=frozen,
            expected_split_hash=split_hash(frozen),
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
            frame_count=625,
            clip_count=125,
        )


def test_sample_rejects_a_training_clip_with_a_validation_target_identity() -> None:
    train = _clips(125)
    frozen = _frozen_split(train)
    cross_identity = replace(
        train[0],
        targets=(frozen["val"][0].source,),
    )
    frozen["train"] = (cross_identity, *train[1:])

    with pytest.raises(ValueError, match="identity-strict training"):
        build_review_sample(
            frozen["train"],
            partition="train",
            frozen_split=frozen,
            expected_split_hash=split_hash(frozen),
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
            frame_count=625,
            clip_count=125,
        )


def test_sample_binds_the_frozen_split_and_meets_the_comparison_gate() -> None:
    train = _clips(125)
    frozen = _frozen_split(train)
    expected_hash = split_hash(frozen)

    sample = build_review_sample(
        train,
        partition="train",
        frozen_split=frozen,
        expected_split_hash=expected_hash,
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
        frame_count=625,
        clip_count=125,
    )

    comparison = tuple(row for row in sample if row.split_role == "comparison")
    assert len(comparison) == 500
    assert len({row.clip_id for row in comparison}) == 100
    assert {row.split_hash for row in sample} == {expected_hash}
    assert len({row.identity_strict_split_hash for row in sample}) == 1
    assert len(detector_sample.review_sample_sha256(sample)) == 64


def test_sample_does_not_shrink_the_frame_or_clip_gates() -> None:
    with pytest.raises(ValueError, match="at least 125 unique clips"):
        _build(
            _clips(99),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
        )

    with pytest.raises(ValueError, match="at least 625 frames"):
        _build(
            _clips(125),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
            frame_count=499,
        )


def test_sample_is_deterministic_balanced_and_source_disjoint() -> None:
    clips = _clips(125)

    first = _build(
        clips,
        partition="train",
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
        seed=17,
    )
    reordered = _build(
        tuple(reversed(clips)),
        partition="train",
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
        seed=17,
    )

    assert first == reordered
    assert len(first) == 625
    assert len({frame.frame_id for frame in first}) == 625
    assert len({frame.clip_id for frame in first}) == 125
    assert sum(frame.double_review for frame in first) == 63
    source_roles: dict[str, set[str]] = {}
    for frame in first:
        source_roles.setdefault(frame.source_hash, set()).add(frame.split_role)
    assert all(len(roles) == 1 for roles in source_roles.values())
    assert Counter(next(iter(roles)) for roles in source_roles.values()) == Counter(
        {"calibration": 25, "comparison": 100}
    )
    manipulation_counts = Counter(frame.manipulation_type for frame in first)
    assert max(manipulation_counts.values()) - min(manipulation_counts.values()) <= 5
    assert {frame.method for frame in first} == {
        "method-0",
        "method-1",
        "method-2",
        "method-3",
    }
    assert {frame.race for frame in first} == {
        "race-0",
        "race-1",
        "race-2",
        "race-3",
    }
    assert {frame.gender for frame in first} == {"gender-0", "gender-1"}


def test_sample_jsonl_round_trip_excludes_private_paths(tmp_path: Path) -> None:
    clips = _clips(125)
    absolute_root = tmp_path.resolve()
    private_clips = tuple(
        _Clip(
            **{
                **clip.__dict__,
                "video_path": absolute_root / clip.video_path,
            }
        )
        for clip in clips
    )
    sample = _build(
        private_clips,
        partition="train",
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
    )
    output = tmp_path / "sample.jsonl"

    write_review_sample(sample, output)

    assert read_review_sample(output) == sample
    text = output.read_text(encoding="utf-8")
    assert str(absolute_root) not in text
    assert len(text.splitlines()) == 625
    assert all("video_path" not in json.loads(line) for line in text.splitlines())


def test_sample_jsonl_rejects_non_boolean_review_flags(tmp_path: Path) -> None:
    row = {
        "frame_id": "frame-001",
        "dataset": "fixture",
        "clip_id": "clip-001",
        "source_hash": "a" * 64,
        "split_hash": "c" * 64,
        "identity_strict_split_hash": "d" * 64,
        "timestamp_sec": 1.0,
        "frame_sha256": "b" * 64,
        "width": 100,
        "height": 100,
        "split_role": "calibration",
        "double_review": "false",
        "manipulation_type": "RealVideo-RealAudio",
        "method": "real",
        "race": "unknown",
        "gender": "unknown",
    }
    path = tmp_path / "bad-sample.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="double_review must be a boolean"):
        read_review_sample(path)


def test_sample_rejects_duplicate_source_frames() -> None:
    static = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="duplicate source frame"):
        _build(
            _clips(125),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=lambda _clip, _timestamp: static,
        )
