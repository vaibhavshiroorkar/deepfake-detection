from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection.benchmarks.detector_annotations import (
    FaceAnnotation,
    FrameAnnotation,
    read_annotations,
    validate_annotations,
    write_annotations,
)
from deepfake_detection.benchmarks.detector_sample import ReviewFrame
from deepfake_detection.views.tracking import Box, Landmarks5, Point


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _landmarks(offset: float = 0.0) -> Landmarks5:
    return Landmarks5(
        eye_left=Point(30 + offset, 30),
        eye_right=Point(50 + offset, 30),
        nose=Point(40 + offset, 45),
        mouth_left=Point(33 + offset, 60),
        mouth_right=Point(47 + offset, 60),
    )


def _sample() -> tuple[ReviewFrame, ...]:
    frames = []
    for index in range(500):
        source_index = index // 5
        frames.append(
            ReviewFrame(
                frame_id=f"frame-{index:03d}",
                dataset="fixture",
                clip_id=f"clip-{source_index:03d}",
                source_hash=_sha(f"source-{source_index:03d}"),
                timestamp_sec=float(index % 5),
                frame_sha256=_sha(f"frame-{index:03d}"),
                width=100,
                height=100,
                split_role=("calibration" if source_index < 20 else "comparison"),
                double_review=index < 50,
                manipulation_type=f"manipulation-{source_index % 4}",
                method=f"method-{source_index % 5}",
                race=f"race-{source_index % 4}",
                gender=f"gender-{source_index % 2}",
            )
        )
    return tuple(frames)


def _annotation(
    frame: ReviewFrame,
    *,
    reviewer_id: str = "reviewer-a",
    target_offset: float = 0.0,
    pose: str = "frontal",
) -> FrameAnnotation:
    return FrameAnnotation(
        frame_id=frame.frame_id,
        frame_sha256=frame.frame_sha256,
        reviewer_id=reviewer_id,
        faces=(
            FaceAnnotation(
                box=Box(20 + target_offset, 20, 60 + target_offset, 80),
                target=True,
                landmarks=_landmarks(target_offset),
            ),
        ),
        no_suitable_target=False,
        pose=pose,
        lighting="normal",
        multi_person=False,
    )


def _complete_annotations(
    sample: tuple[ReviewFrame, ...],
) -> tuple[FrameAnnotation, ...]:
    rows = [_annotation(frame) for frame in sample]
    rows.extend(
        _annotation(frame, reviewer_id="reviewer-b")
        for frame in sample
        if frame.double_review
    )
    return tuple(rows)


def test_frame_annotation_requires_one_landmarked_target_when_suitable() -> None:
    frame = _sample()[0]
    with pytest.raises(ValueError, match="exactly one target"):
        FrameAnnotation(
            frame_id=frame.frame_id,
            frame_sha256=frame.frame_sha256,
            reviewer_id="reviewer-a",
            faces=(),
            no_suitable_target=False,
            pose="frontal",
            lighting="normal",
            multi_person=False,
        )

    with pytest.raises(ValueError, match="target landmarks"):
        FrameAnnotation(
            frame_id=frame.frame_id,
            frame_sha256=frame.frame_sha256,
            reviewer_id="reviewer-a",
            faces=(
                FaceAnnotation(
                    box=Box(20, 20, 60, 80),
                    target=True,
                    landmarks=None,
                ),
            ),
            no_suitable_target=False,
            pose="frontal",
            lighting="normal",
            multi_person=False,
        )


def test_annotation_supports_all_visible_faces_and_no_target_frames() -> None:
    frame = _sample()[0]
    multi_face = FrameAnnotation(
        frame_id=frame.frame_id,
        frame_sha256=frame.frame_sha256,
        reviewer_id="reviewer-a",
        faces=(
            FaceAnnotation(Box(20, 20, 60, 80), True, _landmarks()),
            FaceAnnotation(Box(65, 10, 95, 55), False, None),
        ),
        no_suitable_target=False,
        pose="frontal",
        lighting="normal",
        multi_person=True,
    )
    no_target = FrameAnnotation(
        frame_id=frame.frame_id,
        frame_sha256=frame.frame_sha256,
        reviewer_id="reviewer-a",
        faces=(FaceAnnotation(Box(5, 5, 25, 30), False, None),),
        no_suitable_target=True,
        pose="profile",
        lighting="dark",
        multi_person=False,
    )

    assert len(multi_face.faces) == 2
    assert not no_target.faces[0].target


def test_annotation_jsonl_round_trip_preserves_multi_face_rows(
    tmp_path: Path,
) -> None:
    frame = _sample()[0]
    annotation = FrameAnnotation(
        frame_id=frame.frame_id,
        frame_sha256=frame.frame_sha256,
        reviewer_id="reviewer-a",
        faces=(
            FaceAnnotation(Box(20, 20, 60, 80), True, _landmarks()),
            FaceAnnotation(Box(65, 10, 95, 55), False, None),
        ),
        no_suitable_target=False,
        pose="frontal",
        lighting="normal",
        multi_person=True,
    )
    output = tmp_path / "annotations.jsonl"

    write_annotations((annotation,), output)

    assert read_annotations(output) == (annotation,)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_validation_enforces_gates_and_reports_review_disagreement() -> None:
    sample = _sample()
    annotations = list(_complete_annotations(sample))
    second_index = next(
        index
        for index, annotation in enumerate(annotations)
        if annotation.frame_id == sample[0].frame_id
        and annotation.reviewer_id == "reviewer-b"
    )
    annotations[second_index] = _annotation(
        sample[0],
        reviewer_id="reviewer-b",
        target_offset=4.0,
        pose="profile",
    )

    audit = validate_annotations(sample, tuple(annotations))

    assert audit.valid
    assert audit.frame_count == 500
    assert audit.clip_count == 100
    assert audit.review_count == 550
    assert audit.double_review_required == 50
    assert audit.double_review_completed == 50
    assert audit.missing_frame_ids == ()
    assert audit.missing_strata == ()
    disagreement = next(
        item for item in audit.disagreements if item.frame_id == sample[0].frame_id
    )
    assert disagreement.face_counts == (1, 1)
    assert disagreement.pose_values == ("frontal", "profile")
    assert disagreement.target_box_min_iou == pytest.approx(36 / 44)
    assert disagreement.target_landmark_max_nme == pytest.approx(0.2)


def test_validation_rejects_missing_or_duplicate_double_reviews() -> None:
    sample = _sample()
    annotations = list(_complete_annotations(sample))
    annotations = [
        annotation
        for annotation in annotations
        if not (
            annotation.frame_id == sample[0].frame_id
            and annotation.reviewer_id == "reviewer-b"
        )
    ]
    annotations.append(_annotation(sample[1]))

    audit = validate_annotations(sample, tuple(annotations))

    assert not audit.valid
    assert sample[0].frame_id in audit.missing_double_review_frame_ids
    assert (sample[1].frame_id, "reviewer-a") in audit.duplicate_reviews
    assert any("double review" in error for error in audit.errors)


def test_validation_detects_duplicate_frames_and_source_split_overlap() -> None:
    sample = _sample()
    duplicate = sample + (sample[0],)
    overlap = list(sample)
    overlap[-1] = replace(overlap[-1], source_hash=sample[0].source_hash)

    duplicate_audit = validate_annotations(
        duplicate,
        _complete_annotations(sample),
    )
    overlap_audit = validate_annotations(
        tuple(overlap),
        _complete_annotations(sample),
    )

    assert not duplicate_audit.valid
    assert sample[0].frame_id in duplicate_audit.duplicate_frame_ids
    assert not overlap_audit.valid
    assert overlap_audit.overlapping_source_hashes == (sample[0].source_hash,)


def test_validation_rejects_a_changed_calibration_comparison_ratio() -> None:
    sample = tuple(
        replace(
            frame,
            split_role=(
                "calibration"
                if int(frame.clip_id.removeprefix("clip-")) < 50
                else "comparison"
            ),
        )
        for frame in _sample()
    )

    audit = validate_annotations(sample, _complete_annotations(sample))

    assert not audit.valid
    assert any("20/80" in error for error in audit.errors)


def test_validation_reports_missing_strata_and_bad_frame_geometry() -> None:
    sample = _sample()
    annotations = list(_complete_annotations(sample))
    omitted = {
        frame.frame_id
        for frame in sample
        if frame.manipulation_type == "manipulation-3"
    }
    annotations = [
        annotation for annotation in annotations if annotation.frame_id not in omitted
    ]
    bad = _annotation(sample[0])
    bad = replace(
        bad,
        faces=(FaceAnnotation(Box(20, 20, 120, 80), True, _landmarks()),),
    )
    annotations = [
        bad if annotation == _annotation(sample[0]) else annotation
        for annotation in annotations
    ]

    audit = validate_annotations(sample, tuple(annotations))

    assert not audit.valid
    assert "manipulation_type=manipulation-3" in audit.missing_strata
    assert any("outside frame bounds" in error for error in audit.errors)


def test_reviewer_identity_and_multi_person_flag_are_required() -> None:
    frame = _sample()[0]
    with pytest.raises(ValueError, match="reviewer_id"):
        replace(_annotation(frame), reviewer_id=" ")
    with pytest.raises(ValueError, match="multi_person"):
        replace(
            _annotation(frame),
            faces=(
                FaceAnnotation(Box(20, 20, 60, 80), True, _landmarks()),
                FaceAnnotation(Box(65, 10, 95, 55), False, None),
            ),
        )
