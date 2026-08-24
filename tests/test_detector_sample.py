from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.benchmarks.detector_sample import (
    build_review_sample,
    read_review_sample,
    write_review_sample,
)


@dataclass(frozen=True)
class _Clip:
    clip_id: str
    dataset: str
    video_path: Path
    manipulation_type: str
    method: str
    source: str
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
            race=f"race-{index % 4}",
            gender=f"gender-{index % 2}",
        )
        for index in range(count)
    )


def _frame(clip: _Clip, timestamp_sec: float) -> np.ndarray:
    payload = hashlib.sha256(
        f"{clip.clip_id}:{timestamp_sec:.9f}".encode("ascii")
    ).digest()
    return np.frombuffer(payload[:12], dtype=np.uint8).reshape(2, 2, 3)


def test_sample_requires_an_explicit_training_partition() -> None:
    with pytest.raises(ValueError, match="training partition"):
        build_review_sample(
            _clips(),
            partition="validation",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
        )


def test_sample_does_not_shrink_the_frame_or_clip_gates() -> None:
    with pytest.raises(ValueError, match="at least 100 unique clips"):
        build_review_sample(
            _clips(99),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
        )

    with pytest.raises(ValueError, match="at least 500 frames"):
        build_review_sample(
            _clips(),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=_frame,
            frame_count=499,
        )


def test_sample_is_deterministic_balanced_and_source_disjoint() -> None:
    clips = _clips(104)

    first = build_review_sample(
        clips,
        partition="train",
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
        seed=17,
    )
    reordered = build_review_sample(
        tuple(reversed(clips)),
        partition="train",
        duration_reader=lambda _: 10.0,
        frame_reader=_frame,
        seed=17,
    )

    assert first == reordered
    assert len(first) == 500
    assert len({frame.frame_id for frame in first}) == 500
    assert len({frame.clip_id for frame in first}) == 100
    assert sum(frame.double_review for frame in first) == 50
    source_roles: dict[str, set[str]] = {}
    for frame in first:
        source_roles.setdefault(frame.source_hash, set()).add(frame.split_role)
    assert all(len(roles) == 1 for roles in source_roles.values())
    assert Counter(next(iter(roles)) for roles in source_roles.values()) == Counter(
        {"calibration": 20, "comparison": 80}
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
    clips = _clips()
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
    sample = build_review_sample(
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
    assert len(text.splitlines()) == 500
    assert all("video_path" not in json.loads(line) for line in text.splitlines())


def test_sample_jsonl_rejects_non_boolean_review_flags(tmp_path: Path) -> None:
    row = {
        "frame_id": "frame-001",
        "dataset": "fixture",
        "clip_id": "clip-001",
        "source_hash": "a" * 64,
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
        build_review_sample(
            _clips(),
            partition="train",
            duration_reader=lambda _: 10.0,
            frame_reader=lambda _clip, _timestamp: static,
        )
