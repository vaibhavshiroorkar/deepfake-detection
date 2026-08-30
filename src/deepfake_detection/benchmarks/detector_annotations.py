from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from deepfake_detection.views.tracking import Box, Landmarks5, Point

from .detector_sample import (
    MINIMUM_COMPARISON_CLIPS,
    MINIMUM_COMPARISON_FRAMES,
    MINIMUM_DOUBLE_REVIEW_FRACTION,
    MINIMUM_REVIEW_CLIPS,
    MINIMUM_REVIEW_FRAMES,
    ReviewFrame,
    _validate_sha256,
    review_sample_sha256,
)


@dataclass(frozen=True, slots=True)
class FaceAnnotation:
    box: Box
    target: bool
    landmarks: Landmarks5 | None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise TypeError("Face annotation box must be a Box")
        if self.box.right <= self.box.left or self.box.bottom <= self.box.top:
            raise ValueError("Face annotation box must have positive size")
        if not isinstance(self.target, bool):
            raise TypeError("Face annotation target must be a boolean")
        if self.landmarks is not None and not isinstance(self.landmarks, Landmarks5):
            raise TypeError("Face annotation landmarks must be Landmarks5 or None")


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    frame_id: str
    frame_sha256: str
    reviewer_id: str
    faces: tuple[FaceAnnotation, ...]
    no_suitable_target: bool
    pose: str
    lighting: str
    multi_person: bool
    review_role: Literal["review", "adjudication"] = "review"

    def __post_init__(self) -> None:
        for name in (
            "frame_id",
            "frame_sha256",
            "reviewer_id",
            "pose",
            "lighting",
            "review_role",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} cannot be blank")
            if value != value.strip():
                raise ValueError(f"{name} must be canonical without outer whitespace")
        _validate_sha256("frame_sha256", self.frame_sha256)
        if self.review_role not in {"review", "adjudication"}:
            raise ValueError("review_role must be review or adjudication")
        if not isinstance(self.faces, tuple) or not all(
            isinstance(face, FaceAnnotation) for face in self.faces
        ):
            raise TypeError("faces must be a tuple of FaceAnnotation values")
        if not isinstance(self.no_suitable_target, bool):
            raise TypeError("no_suitable_target must be a boolean")
        if not isinstance(self.multi_person, bool):
            raise TypeError("multi_person must be a boolean")
        target_faces = tuple(face for face in self.faces if face.target)
        if self.no_suitable_target:
            if target_faces:
                raise ValueError("A no-target frame cannot mark a target face")
        elif len(target_faces) != 1:
            raise ValueError("A suitable frame must contain exactly one target")
        elif target_faces[0].landmarks is None:
            raise ValueError("A suitable target requires five target landmarks")
        if self.multi_person != (len(self.faces) > 1):
            raise ValueError("multi_person must match the visible face count")


@dataclass(frozen=True, slots=True)
class AnnotationDisagreement:
    frame_id: str
    reviewer_ids: tuple[str, ...]
    face_counts: tuple[int, ...]
    no_suitable_target_values: tuple[bool, ...]
    pose_values: tuple[str, ...]
    lighting_values: tuple[str, ...]
    multi_person_values: tuple[bool, ...]
    visible_face_min_iou: float | None
    unmatched_face_count: int
    target_label_mismatch: bool
    landmark_presence_mismatch: bool
    visible_landmark_max_nme: float | None
    degenerate_landmark_normalization: bool
    target_box_min_iou: float | None
    target_landmark_max_nme: float | None


@dataclass(frozen=True, slots=True)
class AnnotationAudit:
    valid: bool
    errors: tuple[str, ...]
    split_hash: str
    identity_strict_split_hash: str
    reviewed_sample_sha256: str
    frame_count: int
    comparison_frame_count: int
    clip_count: int
    comparison_clip_count: int
    source_count: int
    annotation_count: int
    review_count: int
    adjudication_count: int
    calibration_source_count: int
    comparison_source_count: int
    double_review_required: int
    double_review_completed: int
    missing_frame_ids: tuple[str, ...]
    missing_double_review_frame_ids: tuple[str, ...]
    duplicate_frame_ids: tuple[str, ...]
    duplicate_frame_sha256s: tuple[str, ...]
    duplicate_reviews: tuple[tuple[str, str], ...]
    overlapping_source_hashes: tuple[str, ...]
    missing_strata: tuple[str, ...]
    disagreements: tuple[AnnotationDisagreement, ...]
    unresolved_disagreement_frame_ids: tuple[str, ...]
    adjudicated_frame_ids: tuple[str, ...]


def _landmark_points(landmarks: Landmarks5) -> tuple[Point, ...]:
    return (
        landmarks.eye_left,
        landmarks.eye_right,
        landmarks.nose,
        landmarks.mouth_left,
        landmarks.mouth_right,
    )


def _target(annotation: FrameAnnotation) -> FaceAnnotation | None:
    return next((face for face in annotation.faces if face.target), None)


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _landmark_nme(
    left: FaceAnnotation,
    right: FaceAnnotation,
) -> tuple[float | None, bool]:
    if left.landmarks is None or right.landmarks is None:
        return None, False
    left_inter_eye = _distance(
        left.landmarks.eye_left,
        left.landmarks.eye_right,
    )
    right_inter_eye = _distance(
        right.landmarks.eye_left,
        right.landmarks.eye_right,
    )
    if left_inter_eye <= 0 or right_inter_eye <= 0:
        return None, True
    normalization = (left_inter_eye + right_inter_eye) / 2
    errors = tuple(
        _distance(left_point, right_point)
        for left_point, right_point in zip(
            _landmark_points(left.landmarks),
            _landmark_points(right.landmarks),
            strict=True,
        )
    )
    return sum(errors) / len(errors) / normalization, False


def _face_key(face: FaceAnnotation) -> tuple[object, ...]:
    landmarks = (
        tuple((point.x, point.y) for point in _landmark_points(face.landmarks))
        if face.landmarks is not None
        else ()
    )
    return (
        face.box.left,
        face.box.top,
        face.box.right,
        face.box.bottom,
        face.target,
        landmarks,
    )


@dataclass(frozen=True, slots=True)
class _FaceComparison:
    minimum_iou: float | None
    unmatched_count: int
    target_label_mismatch: bool
    landmark_presence_mismatch: bool
    landmark_max_nme: float | None
    degenerate_landmark_normalization: bool


def _compare_visible_faces(
    left_faces: Sequence[FaceAnnotation],
    right_faces: Sequence[FaceAnnotation],
) -> _FaceComparison:
    left = tuple(sorted(left_faces, key=_face_key))
    right = tuple(sorted(right_faces, key=_face_key))
    candidates = sorted(
        (
            (-left_face.box.iou(right_face.box), left_index, right_index)
            for left_index, left_face in enumerate(left)
            for right_index, right_face in enumerate(right)
        )
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[FaceAnnotation, FaceAnnotation, float]] = []
    for negative_iou, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left[left_index], right[right_index], -negative_iou))
    landmark_errors: list[float] = []
    landmark_presence_mismatch = False
    degenerate = False
    target_label_mismatch = False
    for left_face, right_face, _ in matches:
        target_label_mismatch |= left_face.target != right_face.target
        landmark_presence_mismatch |= (left_face.landmarks is None) != (
            right_face.landmarks is None
        )
        error, pair_degenerate = _landmark_nme(left_face, right_face)
        degenerate |= pair_degenerate
        if error is not None:
            landmark_errors.append(error)
    return _FaceComparison(
        minimum_iou=min((match[2] for match in matches), default=None),
        unmatched_count=(len(left) - len(used_left)) + (len(right) - len(used_right)),
        target_label_mismatch=target_label_mismatch,
        landmark_presence_mismatch=landmark_presence_mismatch,
        landmark_max_nme=max(landmark_errors, default=None),
        degenerate_landmark_normalization=degenerate,
    )


def _disagreement(
    frame_id: str,
    annotations: Sequence[FrameAnnotation],
) -> AnnotationDisagreement | None:
    ordered = tuple(sorted(annotations, key=lambda item: item.reviewer_id))
    annotation_pairs = tuple(itertools.combinations(ordered, 2))
    face_comparisons = tuple(
        _compare_visible_faces(left.faces, right.faces)
        for left, right in annotation_pairs
    )
    targets = tuple(_target(annotation) for annotation in ordered)
    target_pairs = tuple(
        (left, right)
        for left, right in itertools.combinations(targets, 2)
        if left is not None and right is not None
    )
    target_ious = tuple(left.box.iou(right.box) for left, right in target_pairs)
    target_landmark_results = tuple(
        _landmark_nme(left, right) for left, right in target_pairs
    )
    target_landmark_errors = tuple(
        error for error, _ in target_landmark_results if error is not None
    )
    face_counts = tuple(len(annotation.faces) for annotation in ordered)
    no_target = tuple(annotation.no_suitable_target for annotation in ordered)
    poses = tuple(annotation.pose for annotation in ordered)
    lighting = tuple(annotation.lighting for annotation in ordered)
    multi_person = tuple(annotation.multi_person for annotation in ordered)
    visible_ious = tuple(
        comparison.minimum_iou
        for comparison in face_comparisons
        if comparison.minimum_iou is not None
    )
    visible_landmark_errors = tuple(
        comparison.landmark_max_nme
        for comparison in face_comparisons
        if comparison.landmark_max_nme is not None
    )
    target_iou = min(target_ious) if target_ious else None
    target_nme = max(target_landmark_errors, default=None)
    visible_iou = min(visible_ious, default=None)
    visible_nme = max(visible_landmark_errors, default=None)
    unmatched_count = max(
        (comparison.unmatched_count for comparison in face_comparisons),
        default=0,
    )
    target_label_mismatch = any(
        comparison.target_label_mismatch for comparison in face_comparisons
    )
    landmark_presence_mismatch = any(
        comparison.landmark_presence_mismatch for comparison in face_comparisons
    )
    degenerate = any(
        comparison.degenerate_landmark_normalization for comparison in face_comparisons
    ) or any(pair_degenerate for _, pair_degenerate in target_landmark_results)
    agrees = (
        len(set(face_counts)) == 1
        and len(set(no_target)) == 1
        and len(set(poses)) == 1
        and len(set(lighting)) == 1
        and len(set(multi_person)) == 1
        and unmatched_count == 0
        and not target_label_mismatch
        and not landmark_presence_mismatch
        and not degenerate
        and (visible_iou is None or visible_iou == 1.0)
        and (visible_nme is None or visible_nme == 0.0)
        and (target_iou is None or target_iou == 1.0)
        and (target_nme is None or target_nme == 0.0)
    )
    if agrees:
        return None
    return AnnotationDisagreement(
        frame_id=frame_id,
        reviewer_ids=tuple(annotation.reviewer_id for annotation in ordered),
        face_counts=face_counts,
        no_suitable_target_values=no_target,
        pose_values=tuple(sorted(set(poses))),
        lighting_values=tuple(sorted(set(lighting))),
        multi_person_values=tuple(sorted(set(multi_person))),
        visible_face_min_iou=visible_iou,
        unmatched_face_count=unmatched_count,
        target_label_mismatch=target_label_mismatch,
        landmark_presence_mismatch=landmark_presence_mismatch,
        visible_landmark_max_nme=visible_nme,
        degenerate_landmark_normalization=degenerate,
        target_box_min_iou=target_iou,
        target_landmark_max_nme=target_nme,
    )


def _geometry_errors(
    frame: ReviewFrame,
    annotation: FrameAnnotation,
) -> tuple[str, ...]:
    errors: list[str] = []
    for face_index, face in enumerate(annotation.faces):
        box = face.box
        if (
            box.left < 0
            or box.top < 0
            or box.right > frame.width
            or box.bottom > frame.height
        ):
            errors.append(
                f"Frame {frame.frame_id} face {face_index} box is outside frame bounds"
            )
        if face.landmarks is None:
            continue
        if _distance(face.landmarks.eye_left, face.landmarks.eye_right) <= 0:
            errors.append(
                f"Frame {frame.frame_id} face {face_index} has degenerate "
                "inter-eye landmark normalization"
            )
        for point_index, point in enumerate(_landmark_points(face.landmarks)):
            if not (0 <= point.x < frame.width and 0 <= point.y < frame.height):
                errors.append(
                    f"Frame {frame.frame_id} face {face_index} landmark "
                    f"{point_index} is outside frame bounds"
                )
    return tuple(errors)


def _missing_strata(
    sample: Sequence[ReviewFrame],
    reviewed_frame_ids: set[str],
) -> tuple[str, ...]:
    dimensions = ("manipulation_type", "method", "race", "gender")
    missing: list[str] = []
    reviewed = tuple(frame for frame in sample if frame.frame_id in reviewed_frame_ids)
    for dimension in dimensions:
        expected = {str(getattr(frame, dimension)) for frame in sample}
        present = {str(getattr(frame, dimension)) for frame in reviewed}
        missing.extend(f"{dimension}={value}" for value in sorted(expected - present))
    return tuple(missing)


def validate_annotations(
    sample: Sequence[ReviewFrame],
    annotations: Sequence[FrameAnnotation],
) -> AnnotationAudit:
    """Audit annotation completeness without weakening the frozen review gates."""

    sample_rows = tuple(sample)
    annotation_rows = tuple(annotations)
    errors: list[str] = []
    split_hashes = {frame.split_hash for frame in sample_rows}
    if len(split_hashes) != 1:
        errors.append("Review sample must bind exactly one frozen split hash")
    split_hash = next(iter(split_hashes), "0" * 64)
    identity_strict_hashes = {frame.identity_strict_split_hash for frame in sample_rows}
    if len(identity_strict_hashes) != 1:
        errors.append("Review sample must bind exactly one identity-strict split hash")
    identity_strict_hash = next(iter(identity_strict_hashes), "0" * 64)
    comparison_frames = tuple(
        frame for frame in sample_rows if frame.split_role == "comparison"
    )
    comparison_frame_count = len(comparison_frames)
    comparison_clip_count = len({frame.clip_id for frame in comparison_frames})
    if comparison_frame_count < MINIMUM_COMPARISON_FRAMES:
        errors.append(
            f"Comparison gate requires {MINIMUM_COMPARISON_FRAMES} frames; "
            f"found {comparison_frame_count}"
        )
    if comparison_clip_count < MINIMUM_COMPARISON_CLIPS:
        errors.append(
            f"Comparison gate requires {MINIMUM_COMPARISON_CLIPS} clips; "
            f"found {comparison_clip_count}"
        )
    frame_id_counts = Counter(frame.frame_id for frame in sample_rows)
    duplicate_frame_ids = tuple(
        sorted(frame_id for frame_id, count in frame_id_counts.items() if count > 1)
    )
    if duplicate_frame_ids:
        errors.append("Review sample contains duplicate frame identifiers")
    frame_hash_counts = Counter(frame.frame_sha256 for frame in sample_rows)
    duplicate_frame_sha256s = tuple(
        sorted(
            frame_hash for frame_hash, count in frame_hash_counts.items() if count > 1
        )
    )
    if duplicate_frame_sha256s:
        errors.append("Review sample contains duplicate frame SHA-256 values")
    frame_by_id: dict[str, ReviewFrame] = {}
    for frame in sample_rows:
        frame_by_id.setdefault(frame.frame_id, frame)
    frame_count = len(frame_by_id)
    clip_count = len({frame.clip_id for frame in frame_by_id.values()})
    source_roles: dict[str, set[str]] = defaultdict(set)
    for frame in frame_by_id.values():
        source_roles[frame.source_hash].add(frame.split_role)
    overlapping_sources = tuple(
        sorted(source for source, roles in source_roles.items() if len(roles) > 1)
    )
    if overlapping_sources:
        errors.append("Calibration and comparison source identities overlap")
    calibration_sources = sum(
        roles == {"calibration"} for roles in source_roles.values()
    )
    comparison_sources = sum(roles == {"comparison"} for roles in source_roles.values())
    if calibration_sources == 0 or comparison_sources == 0:
        errors.append("Both calibration and comparison sources are required")
    expected_calibration_sources = int(len(source_roles) * 0.2 + 0.5)
    if calibration_sources != expected_calibration_sources:
        errors.append(
            "Review sources do not follow the frozen 20/80 calibration and "
            "comparison split"
        )
    if frame_count < MINIMUM_REVIEW_FRAMES:
        errors.append(
            f"Review gate requires {MINIMUM_REVIEW_FRAMES} unique frames; "
            f"found {frame_count}"
        )
    if clip_count < MINIMUM_REVIEW_CLIPS:
        errors.append(
            f"Review gate requires {MINIMUM_REVIEW_CLIPS} unique clips; "
            f"found {clip_count}"
        )
    required_double_ids = {
        frame.frame_id for frame in frame_by_id.values() if frame.double_review
    }
    minimum_double = math.ceil(frame_count * MINIMUM_DOUBLE_REVIEW_FRACTION)
    if len(required_double_ids) < minimum_double:
        errors.append(
            f"Double-review gate requires at least {minimum_double} frames; "
            f"sample marks {len(required_double_ids)}"
        )

    by_frame_reviews: dict[str, list[FrameAnnotation]] = defaultdict(list)
    by_frame_adjudications: dict[str, list[FrameAnnotation]] = defaultdict(list)
    review_pairs = Counter(
        (annotation.frame_id, annotation.reviewer_id) for annotation in annotation_rows
    )
    duplicate_reviews = tuple(
        sorted(pair for pair, count in review_pairs.items() if count > 1)
    )
    if duplicate_reviews:
        errors.append("Annotation file contains duplicate frame and reviewer rows")
    invalid_annotation_pairs: set[tuple[str, str]] = set()
    for annotation in annotation_rows:
        frame = frame_by_id.get(annotation.frame_id)
        if frame is None:
            errors.append(f"Annotation references unknown frame {annotation.frame_id}")
            continue
        destination = (
            by_frame_reviews
            if annotation.review_role == "review"
            else by_frame_adjudications
        )
        destination[annotation.frame_id].append(annotation)
        if annotation.frame_sha256 != frame.frame_sha256:
            errors.append(f"Annotation frame hash differs for {annotation.frame_id}")
            invalid_annotation_pairs.add((annotation.frame_id, annotation.reviewer_id))
        geometry_errors = _geometry_errors(frame, annotation)
        if geometry_errors:
            invalid_annotation_pairs.add((annotation.frame_id, annotation.reviewer_id))
        errors.extend(geometry_errors)

    reviewed_ids = set(by_frame_reviews)
    missing_frame_ids = tuple(sorted(set(frame_by_id) - reviewed_ids))
    if missing_frame_ids:
        errors.append(f"Annotations are missing {len(missing_frame_ids)} review frames")
    missing_double = tuple(
        sorted(
            frame_id
            for frame_id in required_double_ids
            if len(
                {
                    annotation.reviewer_id
                    for annotation in by_frame_reviews.get(frame_id, ())
                }
            )
            < 2
        )
    )
    if missing_double:
        errors.append(
            f"Annotations are missing a second double review for "
            f"{len(missing_double)} frames"
        )
    completed_double = len(required_double_ids) - len(missing_double)
    missing_strata = _missing_strata(sample_rows, reviewed_ids)
    if missing_strata:
        errors.append("Annotations are missing required sample strata")
    disagreements = tuple(
        disagreement
        for frame_id, rows in sorted(by_frame_reviews.items())
        if len({row.reviewer_id for row in rows}) >= 2
        if (disagreement := _disagreement(frame_id, rows)) is not None
    )
    disagreement_ids = {item.frame_id for item in disagreements}
    unexpected_adjudications = tuple(
        sorted(set(by_frame_adjudications) - disagreement_ids)
    )
    if unexpected_adjudications:
        errors.append(
            "Adjudication is allowed only for a review disagreement: "
            + ", ".join(unexpected_adjudications)
        )
    unresolved: list[str] = []
    adjudicated: list[str] = []
    for disagreement in disagreements:
        frame_id = disagreement.frame_id
        review_reviewers = {row.reviewer_id for row in by_frame_reviews[frame_id]}
        adjudications = by_frame_adjudications.get(frame_id, ())
        valid_adjudications = tuple(
            row
            for row in adjudications
            if row.reviewer_id not in review_reviewers
            and (row.frame_id, row.reviewer_id) not in invalid_annotation_pairs
        )
        if len(adjudications) == 1 and len(valid_adjudications) == 1:
            adjudicated.append(frame_id)
        else:
            unresolved.append(frame_id)
    if unresolved:
        errors.append(
            "Review disagreements require one distinct valid adjudicator for "
            f"{len(unresolved)} frames"
        )
    unique_errors = tuple(dict.fromkeys(errors))
    return AnnotationAudit(
        valid=not unique_errors,
        errors=unique_errors,
        split_hash=split_hash,
        identity_strict_split_hash=identity_strict_hash,
        reviewed_sample_sha256=review_sample_sha256(sample_rows),
        frame_count=frame_count,
        comparison_frame_count=comparison_frame_count,
        clip_count=clip_count,
        comparison_clip_count=comparison_clip_count,
        source_count=len(source_roles),
        annotation_count=len(annotation_rows),
        review_count=sum(row.review_role == "review" for row in annotation_rows),
        adjudication_count=sum(
            row.review_role == "adjudication" for row in annotation_rows
        ),
        calibration_source_count=calibration_sources,
        comparison_source_count=comparison_sources,
        double_review_required=len(required_double_ids),
        double_review_completed=completed_double,
        missing_frame_ids=missing_frame_ids,
        missing_double_review_frame_ids=missing_double,
        duplicate_frame_ids=duplicate_frame_ids,
        duplicate_frame_sha256s=duplicate_frame_sha256s,
        duplicate_reviews=duplicate_reviews,
        overlapping_source_hashes=overlapping_sources,
        missing_strata=missing_strata,
        disagreements=disagreements,
        unresolved_disagreement_frame_ids=tuple(unresolved),
        adjudicated_frame_ids=tuple(adjudicated),
    )


def annotation_audit_sha256(audit: AnnotationAudit) -> str:
    payload = json.dumps(
        asdict(audit), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_annotations(
    sample: Sequence[ReviewFrame],
    annotations: Sequence[FrameAnnotation],
) -> tuple[FrameAnnotation, ...]:
    """Return one deterministic gold annotation per sample frame."""

    sample_rows = tuple(sample)
    annotation_rows = tuple(annotations)
    audit = validate_annotations(sample_rows, annotation_rows)
    if not audit.valid:
        raise ValueError("Cannot resolve an invalid annotation audit")
    reviews: dict[str, list[FrameAnnotation]] = defaultdict(list)
    adjudications: dict[str, list[FrameAnnotation]] = defaultdict(list)
    for annotation in annotation_rows:
        destination = reviews if annotation.review_role == "review" else adjudications
        destination[annotation.frame_id].append(annotation)
    adjudicated = set(audit.adjudicated_frame_ids)
    resolved: list[FrameAnnotation] = []
    for frame in sample_rows:
        candidates = (
            adjudications[frame.frame_id]
            if frame.frame_id in adjudicated
            else reviews[frame.frame_id]
        )
        resolved.append(min(candidates, key=lambda row: row.reviewer_id))
    return tuple(resolved)


def _point_from_mapping(row: object) -> Point:
    if not isinstance(row, dict) or set(row) != {"x", "y"}:
        raise ValueError("Landmark point must contain x and y")
    return Point(float(row["x"]), float(row["y"]))


def _landmarks_from_mapping(row: object) -> Landmarks5 | None:
    if row is None:
        return None
    names = {"eye_left", "eye_right", "nose", "mouth_left", "mouth_right"}
    if not isinstance(row, dict) or set(row) != names:
        raise ValueError("Landmarks must contain the five canonical points")
    return Landmarks5(
        eye_left=_point_from_mapping(row["eye_left"]),
        eye_right=_point_from_mapping(row["eye_right"]),
        nose=_point_from_mapping(row["nose"]),
        mouth_left=_point_from_mapping(row["mouth_left"]),
        mouth_right=_point_from_mapping(row["mouth_right"]),
    )


def _face_from_mapping(row: object) -> FaceAnnotation:
    if not isinstance(row, dict) or set(row) != {"box", "target", "landmarks"}:
        raise ValueError("Face annotation has missing or unexpected fields")
    box = row["box"]
    if not isinstance(box, dict) or set(box) != {"left", "top", "right", "bottom"}:
        raise ValueError("Face box must contain left, top, right, and bottom")
    if not isinstance(row["target"], bool):
        raise ValueError("Face target must be a boolean")
    return FaceAnnotation(
        box=Box(
            float(box["left"]),
            float(box["top"]),
            float(box["right"]),
            float(box["bottom"]),
        ),
        target=row["target"],
        landmarks=_landmarks_from_mapping(row["landmarks"]),
    )


def _required_string(row: dict[str, object], name: str) -> str:
    value = row[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _annotation_from_mapping(row: dict[str, object]) -> FrameAnnotation:
    expected = set(FrameAnnotation.__dataclass_fields__)
    if set(row) != expected:
        raise ValueError("Frame annotation has missing or unexpected fields")
    faces = row["faces"]
    if not isinstance(faces, list):
        raise ValueError("Frame faces must be a JSON list")
    for name in ("no_suitable_target", "multi_person"):
        if not isinstance(row[name], bool):
            raise ValueError(f"{name} must be a boolean")
    return FrameAnnotation(
        frame_id=_required_string(row, "frame_id"),
        frame_sha256=_required_string(row, "frame_sha256"),
        reviewer_id=_required_string(row, "reviewer_id"),
        faces=tuple(_face_from_mapping(face) for face in faces),
        no_suitable_target=row["no_suitable_target"],
        pose=_required_string(row, "pose"),
        lighting=_required_string(row, "lighting"),
        multi_person=row["multi_person"],
        review_role=_required_string(row, "review_role"),  # type: ignore[arg-type]
    )


def write_annotations(
    annotations: Sequence[FrameAnnotation],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for annotation in annotations:
            handle.write(
                json.dumps(
                    asdict(annotation),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    temporary.replace(path)


def read_annotations(path: Path) -> tuple[FrameAnnotation, ...]:
    annotations: list[FrameAnnotation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Annotation line {line_number} is blank")
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
                annotations.append(_annotation_from_mapping(row))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid annotation line {line_number}: {error}"
                ) from error
    return tuple(annotations)
