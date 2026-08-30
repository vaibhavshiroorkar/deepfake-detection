from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection.benchmarks.detector_annotations import (
    FaceAnnotation,
    FrameAnnotation,
    read_annotations,
    resolve_annotations,
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
    for index in range(625):
        source_index = index // 5
        frames.append(
            ReviewFrame(
                frame_id=f"frame-{index:03d}",
                dataset="fixture",
                clip_id=f"clip-{source_index:03d}",
                source_hash=_sha(f"source-{source_index:03d}"),
                split_hash="f" * 64,
                timestamp_sec=float(index % 5),
                frame_sha256=_sha(f"frame-{index:03d}"),
                width=100,
                height=100,
                split_role=("calibration" if source_index < 25 else "comparison"),
                double_review=index < 63,
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
    review_role: str = "review",
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
        review_role=review_role,
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


def _post_split_gate_sample() -> tuple[ReviewFrame, ...]:
    frames = []
    for index in range(625):
        source_index = index // 5
        frames.append(
            ReviewFrame(
                frame_id=f"gate-frame-{index:03d}",
                dataset="fixture",
                clip_id=f"gate-clip-{source_index:03d}",
                source_hash=_sha(f"gate-source-{source_index:03d}"),
                split_hash="f" * 64,
                timestamp_sec=float(index % 5),
                frame_sha256=_sha(f"gate-frame-{index:03d}"),
                width=100,
                height=100,
                split_role=("calibration" if source_index < 25 else "comparison"),
                double_review=index < 63,
                manipulation_type=f"manipulation-{source_index % 4}",
                method=f"method-{source_index % 5}",
                race=f"race-{source_index % 4}",
                gender=f"gender-{source_index % 2}",
            )
        )
    return tuple(frames)


def test_annotation_audit_binds_split_sample_and_post_split_counts() -> None:
    sample = _post_split_gate_sample()

    audit = validate_annotations(sample, _complete_annotations(sample))

    assert audit.valid
    assert audit.comparison_frame_count == 500
    assert audit.comparison_clip_count == 100
    assert len(audit.split_hash) == 64
    assert len(audit.reviewed_sample_sha256) == 64


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


def test_unresolved_disagreement_invalidates_the_audit() -> None:
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

    assert not audit.valid
    assert audit.frame_count == 625
    assert audit.clip_count == 125
    assert audit.review_count == 688
    assert audit.double_review_required == 63
    assert audit.double_review_completed == 63
    assert audit.missing_frame_ids == ()
    assert audit.missing_strata == ()
    assert audit.unresolved_disagreement_frame_ids == (sample[0].frame_id,)
    assert audit.adjudicated_frame_ids == ()
    disagreement = next(
        item for item in audit.disagreements if item.frame_id == sample[0].frame_id
    )
    assert disagreement.face_counts == (1, 1)
    assert disagreement.pose_values == ("frontal", "profile")
    assert disagreement.target_box_min_iou == pytest.approx(36 / 44)
    assert disagreement.target_landmark_max_nme == pytest.approx(0.2)


def test_one_distinct_adjudicator_resolves_disagreement_and_gold_rows() -> None:
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
    )
    annotations.append(
        _annotation(
            sample[0],
            reviewer_id="reviewer-c",
            target_offset=2.0,
            review_role="adjudication",
        )
    )

    audit = validate_annotations(sample, tuple(annotations))
    gold = resolve_annotations(sample, tuple(reversed(annotations)))

    assert audit.valid
    assert audit.review_count == 688
    assert audit.annotation_count == 689
    assert audit.adjudication_count == 1
    assert audit.unresolved_disagreement_frame_ids == ()
    assert audit.adjudicated_frame_ids == (sample[0].frame_id,)
    assert len(audit.disagreements) == 1
    assert len(gold) == 625
    assert gold[0].reviewer_id == "reviewer-c"
    assert gold[0].review_role == "adjudication"


def test_adjudication_does_not_count_as_an_independent_double_review() -> None:
    sample = _sample()
    annotations = [
        annotation
        for annotation in _complete_annotations(sample)
        if not (
            annotation.frame_id == sample[0].frame_id
            and annotation.reviewer_id == "reviewer-b"
        )
    ]
    annotations.append(
        _annotation(
            sample[0],
            reviewer_id="reviewer-c",
            review_role="adjudication",
        )
    )

    audit = validate_annotations(sample, tuple(annotations))

    assert not audit.valid
    assert sample[0].frame_id in audit.missing_double_review_frame_ids


def test_adjudicator_must_be_unique_and_distinct_from_reviewers() -> None:
    sample = _sample()
    annotations = list(_complete_annotations(sample))
    annotations[-50] = _annotation(
        sample[0], reviewer_id="reviewer-b", target_offset=4.0
    )
    annotations.append(
        _annotation(
            sample[0],
            reviewer_id="reviewer-a",
            review_role="adjudication",
        )
    )

    audit = validate_annotations(sample, tuple(annotations))

    assert not audit.valid
    assert audit.unresolved_disagreement_frame_ids == (sample[0].frame_id,)
    with pytest.raises(ValueError, match="invalid annotation audit"):
        resolve_annotations(sample, tuple(annotations))


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


def test_disagreement_compares_changed_and_unmatched_non_target_faces() -> None:
    sample = _sample()
    target = FaceAnnotation(Box(20, 20, 60, 80), True, _landmarks())
    first = replace(
        _annotation(sample[0]),
        faces=(target, FaceAnnotation(Box(65, 10, 95, 55), False, None)),
        multi_person=True,
    )
    changed = replace(
        first,
        reviewer_id="reviewer-b",
        faces=(target, FaceAnnotation(Box(60, 10, 90, 55), False, None)),
    )
    unmatched = replace(
        first,
        reviewer_id="reviewer-b",
        faces=(target,),
        multi_person=False,
    )
    base = [
        annotation
        for annotation in _complete_annotations(sample)
        if annotation.frame_id != sample[0].frame_id
    ]

    changed_audit = validate_annotations(sample, (*base, first, changed))
    unmatched_audit = validate_annotations(sample, (*base, first, unmatched))

    assert not changed_audit.valid
    changed_disagreement = changed_audit.disagreements[0]
    assert changed_disagreement.visible_face_min_iou == pytest.approx(25 / 35)
    assert changed_disagreement.unmatched_face_count == 0
    assert not unmatched_audit.valid
    assert unmatched_audit.disagreements[0].unmatched_face_count == 1


def test_disagreement_detects_target_label_changes() -> None:
    sample = _sample()
    target_box = Box(20, 20, 60, 80)
    other_box = Box(65, 10, 95, 55)
    first = replace(
        _annotation(sample[0]),
        faces=(
            FaceAnnotation(target_box, True, _landmarks()),
            FaceAnnotation(other_box, False, None),
        ),
        multi_person=True,
    )
    second = replace(
        first,
        reviewer_id="reviewer-b",
        faces=(
            FaceAnnotation(target_box, False, _landmarks()),
            FaceAnnotation(other_box, True, _landmarks(35)),
        ),
    )
    base = [
        annotation
        for annotation in _complete_annotations(sample)
        if annotation.frame_id != sample[0].frame_id
    ]

    audit = validate_annotations(sample, (*base, first, second))

    assert not audit.valid
    assert audit.disagreements[0].target_label_mismatch


def test_landmark_nme_is_symmetric_across_reviewer_order() -> None:
    sample = _sample()
    small = _landmarks()
    large = Landmarks5(
        eye_left=Point(25, 30),
        eye_right=Point(65, 30),
        nose=Point(42, 45),
        mouth_left=Point(30, 60),
        mouth_right=Point(50, 60),
    )

    def annotation(reviewer_id: str, landmarks: Landmarks5) -> FrameAnnotation:
        return replace(
            _annotation(sample[0], reviewer_id=reviewer_id),
            faces=(FaceAnnotation(Box(20, 20, 70, 80), True, landmarks),),
        )

    base = [
        row
        for row in _complete_annotations(sample)
        if row.frame_id != sample[0].frame_id
    ]
    first = validate_annotations(
        sample,
        (*base, annotation("reviewer-a", small), annotation("reviewer-b", large)),
    )
    swapped = validate_annotations(
        sample,
        (*base, annotation("reviewer-a", large), annotation("reviewer-b", small)),
    )

    first_nme = first.disagreements[0].target_landmark_max_nme
    swapped_nme = swapped.disagreements[0].target_landmark_max_nme
    assert first_nme == pytest.approx(5.6 / 30)
    assert swapped_nme == pytest.approx(first_nme)


def test_degenerate_landmark_normalization_invalidates_the_audit() -> None:
    sample = _sample()
    degenerate = Landmarks5(
        eye_left=Point(40, 30),
        eye_right=Point(40, 30),
        nose=Point(40, 45),
        mouth_left=Point(33, 60),
        mouth_right=Point(47, 60),
    )
    annotations = list(_complete_annotations(sample))
    annotations[0] = replace(
        annotations[0],
        faces=(FaceAnnotation(Box(20, 20, 60, 80), True, degenerate),),
    )

    audit = validate_annotations(sample, tuple(annotations))

    assert not audit.valid
    assert any("degenerate inter-eye" in error for error in audit.errors)


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("frame_id", None),
        ("frame_sha256", False),
        ("reviewer_id", 7),
        ("pose", []),
        ("lighting", {}),
        ("review_role", None),
    ),
)
def test_annotation_jsonl_rejects_wrong_string_scalar_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "annotation.jsonl"
    write_annotations((_annotation(_sample()[0]),), path)
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field] = value
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{field} must be a nonblank string"):
        read_annotations(path)


def test_frame_annotation_requires_real_strings_and_known_review_role() -> None:
    frame = _sample()[0]
    with pytest.raises(TypeError, match="reviewer_id must be a string"):
        replace(_annotation(frame), reviewer_id=7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="review_role"):
        replace(_annotation(frame), review_role="editor")  # type: ignore[arg-type]


def test_reviewer_whitespace_alias_cannot_complete_double_review() -> None:
    frame = _sample()[0]

    with pytest.raises(ValueError, match="reviewer_id must be canonical"):
        replace(_annotation(frame), reviewer_id="reviewer-a ")


def test_reviewer_whitespace_alias_cannot_act_as_adjudicator() -> None:
    frame = _sample()[0]

    with pytest.raises(ValueError, match="reviewer_id must be canonical"):
        _annotation(
            frame,
            reviewer_id=" reviewer-a",
            review_role="adjudication",
        )


@pytest.mark.parametrize(
    "field",
    ("frame_id", "frame_sha256", "pose", "lighting", "review_role"),
)
def test_required_annotation_strings_reject_outer_whitespace(field: str) -> None:
    annotation = _annotation(_sample()[0])

    with pytest.raises(ValueError, match=f"{field} must be canonical"):
        replace(annotation, **{field: f" {getattr(annotation, field)}"})


def test_adjudication_is_rejected_when_independent_reviews_agree() -> None:
    sample = _sample()
    annotations = (
        *_complete_annotations(sample),
        _annotation(
            sample[0],
            reviewer_id="reviewer-c",
            target_offset=4.0,
            review_role="adjudication",
        ),
    )

    audit = validate_annotations(sample, annotations)

    assert not audit.valid
    assert any(
        sample[0].frame_id in error and "Adjudication" in error
        for error in audit.errors
    )
    with pytest.raises(ValueError, match="invalid annotation audit"):
        resolve_annotations(sample, annotations)


def test_validation_rejects_duplicate_sample_frame_hashes() -> None:
    sample = list(_sample())
    sample[-1] = replace(sample[-1], frame_sha256=sample[0].frame_sha256)

    audit = validate_annotations(tuple(sample), _complete_annotations(tuple(sample)))

    assert not audit.valid
    assert audit.duplicate_frame_sha256s == (sample[0].frame_sha256,)
