from __future__ import annotations

from dataclasses import replace

import pytest

from deepfake_detection.benchmarks.detector_annotations import (
    FaceAnnotation,
    FrameAnnotation,
)
from deepfake_detection.benchmarks.detector_metrics import (
    BOOTSTRAP_SEED,
    FROZEN_DETECTOR_RULE_REVISION,
    CandidateFrame,
    DetectorBenchmarkReport,
    DetectorDecision,
    DetectorLatency,
    DetectorMetrics,
    TrackerMetrics,
    _maximum_iou_assignment,
    calibrate_detector_threshold,
    compare_detectors,
    evaluate_detector,
)
from deepfake_detection.evaluation.bootstrap import BootstrapInterval
from deepfake_detection.views.tracking import Box, Detection, Landmarks5, Point


def _landmarks(offset: float = 0.0) -> Landmarks5:
    return Landmarks5(
        eye_left=Point(3 + offset, 5),
        eye_right=Point(13 + offset, 5),
        nose=Point(8 + offset, 9),
        mouth_left=Point(5 + offset, 13),
        mouth_right=Point(11 + offset, 13),
    )


def _mouth_shift(offset: float) -> Landmarks5:
    base = _landmarks()
    return Landmarks5(
        eye_left=base.eye_left,
        eye_right=base.eye_right,
        nose=base.nose,
        mouth_left=Point(base.mouth_left.x + offset, base.mouth_left.y),
        mouth_right=Point(base.mouth_right.x + offset, base.mouth_right.y),
    )


def _annotation(
    frame_id: str,
    *,
    frame_hash: str,
    extra_face: bool = False,
) -> FrameAnnotation:
    faces = [FaceAnnotation(Box(1, 1, 16, 18), True, _landmarks())]
    if extra_face:
        faces.append(FaceAnnotation(Box(20, 1, 30, 16), False, None))
    return FrameAnnotation(
        frame_id=frame_id,
        frame_sha256=frame_hash,
        reviewer_id="reviewer-a",
        faces=tuple(faces),
        no_suitable_target=False,
        pose="frontal",
        lighting="even",
        multi_person=extra_face,
    )


def _runtime_snapshot(cpu: str = "fixture") -> dict[str, object]:
    return {
        "started_at_utc": "2026-08-25T00:00:00+00:00",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "python_version": "3.13.0",
        "platform": "fixture-platform",
        "packages": {"opencv-python": "5.0.0.93"},
        "cpu": cpu,
        "gpu": None,
        "gpu_memory_mib": None,
        "available_memory_mib": 1024,
        "ffmpeg_version": "fixture-ffmpeg",
    }


def _candidate(
    frame_id: str,
    *,
    source: str,
    split_role: str,
    detections: tuple[Detection, ...],
    clip_id: str = "clip-1",
    timestamp: float = 0.0,
    latency_ms: float = 2.0,
) -> CandidateFrame:
    return CandidateFrame(
        frame_id=frame_id,
        clip_id=clip_id,
        timestamp_sec=timestamp,
        frame_sha256=(frame_id[-1] * 64),
        source_hash=source,
        split_role=split_role,
        detections=detections,
        latency_ms=latency_ms,
        detector_revision="detector-revision-1",
        model_sha256="a" * 64,
        device="cpu",
        thread_count=4,
    )


def _with_calibration(
    records: tuple[CandidateFrame, ...],
    annotations: tuple[FrameAnnotation, ...],
    *,
    threshold: float,
) -> tuple[tuple[CandidateFrame, ...], tuple[FrameAnnotation, ...]]:
    calibration = _candidate(
        "frame-f",
        source="f" * 64,
        split_role="calibration",
        detections=(Detection(Box(1, 1, 16, 18), threshold, _landmarks()),),
        clip_id="calibration-clip",
    )
    gold = _annotation("frame-f", frame_hash=calibration.frame_sha256)
    return (calibration, *records), (gold, *annotations)


def test_calibration_uses_only_calibration_sources_and_frozen_ties() -> None:
    calibration = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="calibration",
        detections=(
            Detection(Box(1, 1, 16, 18), 0.70, _landmarks()),
            Detection(Box(40, 40, 50, 50), 0.60, _landmarks()),
        ),
    )
    comparison = _candidate(
        "frame-2",
        source="2" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 16, 18), 0.95, _landmarks()),),
    )
    annotations = (
        _annotation("frame-1", frame_hash="1" * 64),
        _annotation("frame-2", frame_hash="2" * 64),
    )

    threshold = calibrate_detector_threshold((comparison, calibration), annotations)

    assert threshold == 0.70


def test_calibration_matches_all_visible_faces_before_counting_false_detections() -> (
    None
):
    record = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="calibration",
        detections=(
            Detection(Box(0, 0, 3, 10), 0.80, _landmarks()),
            Detection(Box(0, 0, 6, 10), 0.80, None),
        ),
    )
    annotation = FrameAnnotation(
        frame_id="frame-1",
        frame_sha256="1" * 64,
        reviewer_id="reviewer-a",
        faces=(
            FaceAnnotation(Box(0, 0, 2, 10), True, _landmarks()),
            FaceAnnotation(Box(0, 0, 4, 10), False, None),
        ),
        no_suitable_target=False,
        pose="frontal",
        lighting="even",
        multi_person=True,
    )

    threshold = calibrate_detector_threshold(
        (record,),
        (annotation,),
    )

    assert threshold == 0.80


def test_assignment_filters_matches_below_the_frozen_iou_threshold() -> None:
    detections = (Detection(Box(0, 0, 2, 2), 0.9, None),)
    faces = (FaceAnnotation(Box(10, 10, 12, 12), True, None),)

    assert _maximum_iou_assignment(detections, faces) == ()


def test_assignment_optimizes_all_overlaps_before_applying_the_iou_threshold() -> None:
    class FaceBox:
        def __init__(self, index: int) -> None:
            self.index = index

    class DetectionBox:
        def __init__(self, overlaps: tuple[float, float]) -> None:
            self.overlaps = overlaps

        def iou(self, face: FaceBox) -> float:
            return self.overlaps[face.index]

    class Candidate:
        def __init__(self, box: DetectionBox | FaceBox) -> None:
            self.box = box

    detections = (
        Candidate(DetectionBox((0.55, 0.51))),
        Candidate(DetectionBox((0.49, 0.01))),
    )
    faces = (Candidate(FaceBox(0)), Candidate(FaceBox(1)))

    assert _maximum_iou_assignment(detections, faces) == ((0, 1, 0.51),)


def test_assignment_handles_a_crowded_frame() -> None:
    detections = tuple(
        Detection(Box(index * 3, 0, index * 3 + 2, 2), 0.9, None) for index in range(24)
    )
    faces = tuple(
        FaceAnnotation(Box(index * 3, 0, index * 3 + 2, 2), index == 0, None)
        for index in range(24)
    )

    assignment = _maximum_iou_assignment(detections, faces)

    assert assignment == tuple((index, index, 1.0) for index in range(24))


def test_assignment_uses_stable_indices_for_equal_iou_ties() -> None:
    detections = (
        Detection(Box(0, 0, 2, 2), 0.9, None),
        Detection(Box(0, 0, 2, 2), 0.9, None),
    )
    faces = (
        FaceAnnotation(Box(0, 0, 2, 2), True, None),
        FaceAnnotation(Box(0, 0, 2, 2), False, None),
    )

    assert _maximum_iou_assignment(detections, faces) == (
        (0, 0, 1.0),
        (1, 1, 1.0),
    )


def test_evaluation_is_comparison_only_and_reports_landmark_error() -> None:
    records = (
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="calibration",
            detections=(Detection(Box(1, 1, 16, 18), 0.70, _landmarks()),),
        ),
        _candidate(
            "frame-2",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.80, _landmarks(1)),),
            latency_ms=3.0,
        ),
        _candidate(
            "frame-3",
            source="3" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.75, _landmarks(1)),),
            clip_id="clip-2",
            latency_ms=5.0,
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.70,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    assert report.frame_count == 2
    assert report.source_count == 2
    assert report.metrics.target_recall == 1.0
    assert report.metrics.false_detections_per_frame == 0.0
    assert report.metrics.landmark_nme == pytest.approx(0.1)
    assert report.metrics.landmark_coverage == 1.0
    assert report.latency.median_ms == 4.0
    assert report.latency.p95_ms == pytest.approx(4.9)
    assert report.latency.throughput_fps == 250.0
    assert set(report.intervals) >= {
        "target_recall",
        "false_detections_per_frame",
        "landmark_nme",
        "landmark_coverage",
        "non_target_candidate_count",
        "greedy_iou.target_track_errors_per_1000",
        "constant_velocity.target_track_errors_per_1000",
        "latency.median_ms",
        "latency.p95_ms",
        "latency.throughput_fps",
    }
    assert report.bootstrap_samples == 1000
    assert report.bootstrap_seed == BOOTSTRAP_SEED
    assert all(
        interval.successful_samples == 1000 for interval in report.intervals.values()
    )


def test_evaluation_rejects_defined_metrics_with_undefined_fixed_resamples() -> None:
    records = (
        _candidate(
            "frame-a",
            source="a" * 64,
            split_role="calibration",
            detections=(Detection(Box(1, 1, 16, 18), 0.7, _landmarks()),),
        ),
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
        ),
        _candidate(
            "frame-2",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, None),),
            clip_id="clip-2",
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )

    with pytest.raises(ValueError, match="undefined fixed bootstrap"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )


def test_evaluation_omits_intervals_for_undefined_optional_metrics() -> None:
    records = (
        _candidate(
            "frame-a",
            source="a" * 64,
            split_role="calibration",
            detections=(Detection(Box(1, 1, 16, 18), 0.7, _landmarks()),),
        ),
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, None),),
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.7,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    assert report.metrics.landmark_nme is None
    assert report.metrics.aligned_mouth_jitter is None
    assert "landmark_nme" not in report.intervals
    assert "aligned_mouth_jitter" not in report.intervals


def test_evaluation_tracks_both_association_modes() -> None:
    records = tuple(
        _candidate(
            f"frame-{index + 1}",
            source="2" * 64,
            split_role="comparison",
            detections=(
                Detection(Box(left, 1, left + 10, 11), 0.90, _landmarks(left)),
            ),
            timestamp=float(index),
        )
        for index, left in enumerate((0, 5, 15))
    )
    annotations = tuple(
        FrameAnnotation(
            frame_id=record.frame_id,
            frame_sha256=record.frame_sha256,
            reviewer_id="reviewer-a",
            faces=(
                FaceAnnotation(
                    Box(left, 1, left + 10, 11),
                    True,
                    _landmarks(left),
                ),
            ),
            no_suitable_target=False,
            pose="frontal",
            lighting="even",
            multi_person=False,
        )
        for record, left in zip(records, (0, 5, 15), strict=True)
    )
    records, annotations = _with_calibration(
        records,
        annotations,
        threshold=0.5,
    )

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.5,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    trackers = {tracker.association: tracker for tracker in report.trackers}
    assert set(trackers) == {"greedy_iou", "constant_velocity"}
    assert trackers["greedy_iou"].stable_track_coverage == 0.0
    assert trackers["constant_velocity"].stable_track_coverage == 1.0
    assert trackers["constant_velocity"].target_track_errors == 0


def test_evaluation_measures_mouth_residual_motion_after_face_alignment() -> None:
    records = (
        _candidate(
            "frame-1",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, _mouth_shift(0)),),
            timestamp=0.0,
        ),
        _candidate(
            "frame-2",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, _mouth_shift(1)),),
            timestamp=1.0,
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )
    records, annotations = _with_calibration(
        records,
        annotations,
        threshold=0.7,
    )

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.7,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    assert report.metrics.aligned_mouth_jitter == pytest.approx(0.1)


def test_evaluation_counts_a_stable_non_target_primary_track_as_target_error() -> None:
    records = tuple(
        _candidate(
            f"frame-{index + 1}",
            source="2" * 64,
            split_role="comparison",
            detections=(
                Detection(Box(20, 1, 30, 16), 0.99, _landmarks()),
                Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),
            ),
            timestamp=float(index),
        )
        for index in range(5)
    )
    annotations = tuple(
        _annotation(
            record.frame_id,
            frame_hash=record.frame_sha256,
            extra_face=True,
        )
        for record in records
    )
    records, annotations = _with_calibration(
        records,
        annotations,
        threshold=0.7,
    )

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.7,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    for tracker in report.trackers:
        assert tracker.stable_track_coverage == 0.0
        assert tracker.abstention_rate == 1.0
        assert tracker.tracked_frames == 0
        assert tracker.target_track_errors == 0
        assert tracker.target_track_errors_per_1000 == 0
    assert report.metrics.non_target_candidate_count == 5


def _identity_sequence_report(states: tuple[str, ...]) -> DetectorBenchmarkReport:
    target_box = Box(1, 1, 16, 18)
    other_box = Box(7, 1, 22, 18)
    records = tuple(
        _candidate(
            f"frame-{index + 1}",
            source="2" * 64,
            split_role="comparison",
            detections=(
                Detection(
                    target_box if state == "target" else other_box,
                    0.9,
                    _landmarks(),
                ),
            ),
            timestamp=float(index),
        )
        for index, state in enumerate(states)
    )
    annotations = tuple(
        FrameAnnotation(
            frame_id=record.frame_id,
            frame_sha256=record.frame_sha256,
            reviewer_id="reviewer-a",
            faces=(
                FaceAnnotation(target_box, True, _landmarks()),
                FaceAnnotation(other_box, False, None),
            ),
            no_suitable_target=False,
            pose="frontal",
            lighting="even",
            multi_person=True,
        )
        for record in records
    )
    records, annotations = _with_calibration(
        records,
        annotations,
        threshold=0.7,
    )
    return evaluate_detector(
        records,
        annotations,
        threshold=0.7,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )


def test_stable_wrong_initial_acquisition_is_one_track_error_event() -> None:
    report = _identity_sequence_report(("other",) * 5)

    for tracker in report.trackers:
        assert tracker.stable_track_coverage == 1.0
        assert tracker.tracked_frames == 5
        assert tracker.target_track_errors == 1
        assert tracker.target_track_errors_per_1000 == 200


def test_sustained_wrong_segment_and_recovery_are_two_transition_events() -> None:
    report = _identity_sequence_report(("target", "other", "other", "target"))

    for tracker in report.trackers:
        assert tracker.stable_track_coverage == 1.0
        assert tracker.tracked_frames == 4
        assert tracker.target_track_errors == 2
        assert tracker.target_track_errors_per_1000 == 500


def _report(
    name: str,
    *,
    recall: float,
    nme: float,
    track_errors_per_1000: float,
    latency_ms: float,
    scope: str = "research_evidence",
    evaluation_set_sha256: str = "e" * 64,
) -> DetectorBenchmarkReport:
    metrics = DetectorMetrics(
        target_recall=recall,
        false_detections_per_frame=0.05,
        non_target_detections_per_frame=0.1,
        non_target_candidate_count=40,
        landmark_nme=nme,
        landmark_coverage=1.0,
        aligned_mouth_jitter=0.02,
    )
    tracked_frames = 100_000
    target_track_errors = round(track_errors_per_1000 * tracked_frames / 1000)
    trackers = (
        TrackerMetrics(
            association="greedy_iou",
            stable_track_coverage=0.9,
            abstention_rate=1 - 0.9,
            target_track_errors=target_track_errors,
            tracked_frames=tracked_frames,
            target_track_errors_per_1000=track_errors_per_1000,
        ),
        TrackerMetrics(
            association="constant_velocity",
            stable_track_coverage=0.95,
            abstention_rate=1 - 0.95,
            target_track_errors=target_track_errors,
            tracked_frames=tracked_frames,
            target_track_errors_per_1000=track_errors_per_1000,
        ),
    )
    latency = DetectorLatency(
        timed_frames=500,
        median_ms=latency_ms,
        p95_ms=latency_ms + 1,
        throughput_fps=1000 / latency_ms,
        device="cpu",
        thread_count=4,
    )
    point_estimates = {
        "target_recall": metrics.target_recall,
        "false_detections_per_frame": metrics.false_detections_per_frame,
        "non_target_detections_per_frame": metrics.non_target_detections_per_frame,
        "non_target_candidate_count": float(metrics.non_target_candidate_count),
        "landmark_nme": float(metrics.landmark_nme),
        "landmark_coverage": metrics.landmark_coverage,
        "aligned_mouth_jitter": float(metrics.aligned_mouth_jitter),
        "greedy_iou.stable_track_coverage": trackers[0].stable_track_coverage,
        "greedy_iou.abstention_rate": trackers[0].abstention_rate,
        "greedy_iou.target_track_errors_per_1000": trackers[
            0
        ].target_track_errors_per_1000,
        "constant_velocity.stable_track_coverage": trackers[1].stable_track_coverage,
        "constant_velocity.abstention_rate": trackers[1].abstention_rate,
        "constant_velocity.target_track_errors_per_1000": trackers[
            1
        ].target_track_errors_per_1000,
        "latency.median_ms": latency.median_ms,
        "latency.p95_ms": latency.p95_ms,
        "latency.throughput_fps": latency.throughput_fps,
    }
    intervals = {
        key: BootstrapInterval(value, value, value, 1000)
        for key, value in point_estimates.items()
    }
    return DetectorBenchmarkReport(
        detector_name=name,
        detector_revision=f"{name}-revision",
        model_sha256="a" * 64,
        threshold=0.7,
        collection_threshold=0.1,
        evidence_scope=scope,
        rule_revision=FROZEN_DETECTOR_RULE_REVISION,
        frame_count=500,
        comparison_clip_count=100,
        source_count=100,
        metrics=metrics,
        trackers=trackers,
        latency=latency,
        intervals=intervals,
        runtime_snapshot=_runtime_snapshot(),
        raw_results_sha256="f" * 64,
        evaluation_set_sha256=evaluation_set_sha256,
        split_hash="b" * 64,
        reviewed_sample_sha256="c" * 64,
        annotation_audit_sha256="d" * 64,
        annotation_audit_validated=True,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


def test_research_report_rejects_non_cpu_runtime_metadata() -> None:
    report = _report(
        "candidate",
        recall=0.95,
        nme=0.08,
        track_errors_per_1000=2,
        latency_ms=5,
    )

    with pytest.raises(ValueError, match="CPU runtime metadata"):
        replace(report, latency=replace(report.latency, device="cuda:0"))


def test_research_report_requires_the_post_split_frame_and_clip_gate() -> None:
    report = _report(
        "candidate",
        recall=0.95,
        nme=0.08,
        track_errors_per_1000=2,
        latency_ms=5,
    )

    with pytest.raises(ValueError, match="500 comparison frames"):
        replace(
            report,
            frame_count=499,
            latency=replace(report.latency, timed_frames=499),
        )

    with pytest.raises(ValueError, match="100 comparison clips"):
        replace(report, comparison_clip_count=99)


def test_research_comparison_requires_the_same_split_and_reviewed_audit() -> None:
    first = _report(
        "first", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5
    )
    changed_split = replace(first, detector_name="changed-split", split_hash="d" * 64)
    changed_audit = replace(
        first,
        detector_name="changed-audit",
        annotation_audit_sha256="e" * 64,
    )

    with pytest.raises(ValueError, match="split"):
        compare_detectors((first, changed_split))
    with pytest.raises(ValueError, match="annotation audit"):
        compare_detectors((first, changed_audit))


def test_selection_does_not_prefer_a_zero_frame_tracker() -> None:
    report = _report(
        "candidate", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5
    )
    zero_frame_tracker = replace(
        report.trackers[0],
        stable_track_coverage=0.0,
        abstention_rate=1.0,
        target_track_errors=0,
        tracked_frames=0,
        target_track_errors_per_1000=0.0,
    )
    intervals = dict(report.intervals)
    intervals.update(
        {
            "greedy_iou.stable_track_coverage": BootstrapInterval(0.0, 0.0, 0.0, 1000),
            "greedy_iou.abstention_rate": BootstrapInterval(1.0, 1.0, 1.0, 1000),
            "greedy_iou.target_track_errors_per_1000": BootstrapInterval(
                0.0, 0.0, 0.0, 1000
            ),
        }
    )
    zero_frame_report = replace(
        report,
        trackers=(zero_frame_tracker, report.trackers[1]),
        intervals=intervals,
    )

    decision = compare_detectors((zero_frame_report,))

    assert decision.selected_association == "constant_velocity"


def test_selection_applies_each_frozen_rejection_margin_before_speed() -> None:
    reports = (
        _report("best", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=8),
        _report(
            "lowrecall", recall=0.939, nme=0.05, track_errors_per_1000=1, latency_ms=2
        ),
        _report(
            "badnme", recall=0.95, nme=0.091, track_errors_per_1000=10, latency_ms=3
        ),
        _report(
            "badtrack", recall=0.95, nme=0.08, track_errors_per_1000=3.01, latency_ms=4
        ),
        _report("winner", recall=0.95, nme=0.08, track_errors_per_1000=3, latency_ms=5),
    )

    decision = compare_detectors(reports)

    assert decision.selected_detector == "winner"
    assert decision.selected_association == "constant_velocity"
    assert decision.rejected == {
        "badnme": ("landmark_nme_margin", "target_track_error_margin"),
        "badtrack": ("target_track_error_margin",),
        "lowrecall": ("target_recall_margin",),
    }
    assert decision.eligible_detectors == ("best", "winner")


def test_selection_keeps_candidates_exactly_on_each_frozen_margin() -> None:
    reports = (
        _report(
            "best",
            recall=0.95,
            nme=0.08,
            track_errors_per_1000=2,
            latency_ms=8,
        ),
        _report(
            "boundary",
            recall=0.94,
            nme=0.09,
            track_errors_per_1000=3,
            latency_ms=5,
        ),
    )

    decision = compare_detectors(reports)

    assert decision.selected_detector == "boundary"
    assert decision.eligible_detectors == ("best", "boundary")


def test_selection_exposes_speed_tie_and_uses_only_given_downstream_scores() -> None:
    reports = (
        _report("alpha", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5),
        _report("beta", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5),
    )

    tied = compare_detectors(reports)
    decided = compare_detectors(
        reports,
        downstream_validation={"alpha": 0.80, "beta": 0.82},
    )

    assert tied == DetectorDecision(
        selected_detector=None,
        selected_association=None,
        eligible_detectors=("alpha", "beta"),
        rejected={},
        downstream_tie_candidates=("alpha", "beta"),
        reason="downstream_validation_required",
        rule_revision=FROZEN_DETECTOR_RULE_REVISION,
    )
    assert decided.selected_detector == "beta"
    assert decided.reason == "downstream_validation_tie_break"


def test_selection_cannot_turn_fixture_evidence_into_a_real_choice() -> None:
    report = _report(
        "fixture",
        recall=1.0,
        nme=0.0,
        track_errors_per_1000=0,
        latency_ms=1,
        scope="software_fixture_only",
    )

    decision = compare_detectors((report,))

    assert decision.selected_detector is None
    assert decision.reason == "software_fixture_only"


def test_report_rejects_incomplete_or_inconsistent_bootstrap_intervals() -> None:
    report = _report(
        "candidate", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5
    )
    wrong_estimate = dict(report.intervals)
    wrong_estimate["target_recall"] = replace(
        wrong_estimate["target_recall"], estimate=0.5
    )
    tiny_estimate_change = dict(report.intervals)
    tiny_estimate_change["target_recall"] = replace(
        tiny_estimate_change["target_recall"],
        estimate=report.metrics.target_recall + 5e-13,
    )
    incomplete = dict(report.intervals)
    incomplete.pop("latency.p95_ms")
    unexpected = {
        **report.intervals,
        "hidden_metric": BootstrapInterval(1.0, 1.0, 1.0, 1000),
    }
    short = dict(report.intervals)
    short["target_recall"] = replace(short["target_recall"], successful_samples=999)
    noninteger = dict(report.intervals)
    noninteger["target_recall"] = replace(
        noninteger["target_recall"], successful_samples=1000.0
    )
    nonfinite = dict(report.intervals)
    nonfinite["target_recall"] = replace(nonfinite["target_recall"], lower=float("nan"))

    for intervals in (
        wrong_estimate,
        tiny_estimate_change,
        incomplete,
        unexpected,
        short,
        noninteger,
        nonfinite,
    ):
        with pytest.raises(ValueError, match="interval"):
            replace(report, intervals=intervals)


def test_report_rejects_invalid_cross_field_constraints() -> None:
    report = _report(
        "candidate", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5
    )
    reversed_interval_bounds = dict(report.intervals)
    reversed_interval_bounds["target_recall"] = BootstrapInterval(
        report.metrics.target_recall,
        0.96,
        0.94,
        1000,
    )
    duplicate_tracker_modes = (
        report.trackers[0],
        report.trackers[1],
        report.trackers[0],
    )
    lower_p95_intervals = dict(report.intervals)
    lower_p95_intervals["latency.p95_ms"] = BootstrapInterval(4.0, 4.0, 4.0, 1000)

    with pytest.raises(ValueError):
        replace(report, intervals=reversed_interval_bounds)
    with pytest.raises(ValueError):
        replace(report, trackers=duplicate_tracker_modes)
    with pytest.raises(ValueError):
        replace(report, frame_count=report.frame_count - 1)
    with pytest.raises(ValueError):
        replace(
            report,
            latency=replace(report.latency, p95_ms=4.0),
            intervals=lower_p95_intervals,
        )


def test_metric_contracts_reject_inconsistent_tracker_and_latency_counts() -> None:
    tracker = TrackerMetrics(
        association="greedy_iou",
        stable_track_coverage=0.9,
        abstention_rate=1 - 0.9,
        target_track_errors=2,
        tracked_frames=1000,
        target_track_errors_per_1000=2.0,
    )
    with pytest.raises(ValueError, match="complements"):
        replace(tracker, abstention_rate=0.2)
    with pytest.raises(ValueError, match="complements"):
        replace(tracker, abstention_rate=tracker.abstention_rate + 5e-13)
    with pytest.raises(ValueError, match="exactly match"):
        replace(tracker, target_track_errors_per_1000=3.0)
    with pytest.raises(ValueError, match="exactly match"):
        replace(
            tracker,
            target_track_errors_per_1000=(tracker.target_track_errors_per_1000 + 5e-13),
        )
    with pytest.raises((TypeError, ValueError), match="integer"):
        replace(tracker, tracked_frames=1000.0)
    with pytest.raises((TypeError, ValueError), match="integer"):
        DetectorLatency(
            timed_frames=True,
            median_ms=1.0,
            p95_ms=2.0,
            throughput_fps=500.0,
            device="cpu",
            thread_count=4,
        )


def test_selection_rejects_unpaired_frames_and_unequal_cpu_settings() -> None:
    first = _report(
        "alpha", recall=0.95, nme=0.08, track_errors_per_1000=2, latency_ms=5
    )
    changed_frames = _report(
        "beta",
        recall=0.95,
        nme=0.08,
        track_errors_per_1000=2,
        latency_ms=6,
        evaluation_set_sha256="d" * 64,
    )
    changed_threads = replace(
        changed_frames,
        evaluation_set_sha256=first.evaluation_set_sha256,
        latency=replace(changed_frames.latency, thread_count=8),
    )
    changed_cpu = replace(
        changed_frames,
        evaluation_set_sha256=first.evaluation_set_sha256,
        runtime_snapshot=_runtime_snapshot("different-cpu"),
    )
    changed_packages = replace(
        changed_frames,
        evaluation_set_sha256=first.evaluation_set_sha256,
        runtime_snapshot={
            **_runtime_snapshot(),
            "packages": {"opencv-python": "different-version"},
        },
    )

    with pytest.raises(ValueError, match="comparison frames"):
        compare_detectors((first, changed_frames))
    with pytest.raises(ValueError, match="thread counts"):
        compare_detectors((first, changed_threads))
    with pytest.raises(ValueError, match="CPU hardware"):
        compare_detectors((first, changed_cpu))
    with pytest.raises(ValueError, match="runtime environments"):
        compare_detectors((first, changed_packages))


def test_selection_applies_nme_and_tracking_rejections_to_recall_pool() -> None:
    reports = (
        _report(
            "nme-best", recall=0.95, nme=0.05, track_errors_per_1000=10, latency_ms=5
        ),
        _report(
            "track-best", recall=0.95, nme=0.061, track_errors_per_1000=1, latency_ms=6
        ),
        _report(
            "compromise", recall=0.95, nme=0.05, track_errors_per_1000=2.1, latency_ms=4
        ),
    )

    decision = compare_detectors(reports)

    assert decision.selected_detector is None
    assert decision.eligible_detectors == ()
    assert decision.rejected == {
        "compromise": ("target_track_error_margin",),
        "nme-best": ("target_track_error_margin",),
        "track-best": ("landmark_nme_margin",),
    }
    assert decision.reason == "no_eligible_detector"


def test_evaluation_rejects_hash_mismatch_nonfinite_values_and_rule_changes() -> None:
    record = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )
    annotation = _annotation("frame-1", frame_hash="2" * 64)

    with pytest.raises(ValueError, match="frame hash"):
        evaluate_detector(
            (record,),
            (annotation,),
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )
    with pytest.raises(ValueError, match="finite"):
        replace(record, latency_ms=float("nan"))
    with pytest.raises(ValueError, match="frozen detector rule"):
        evaluate_detector(
            (record,),
            (_annotation("frame-1", frame_hash="1" * 64),),
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            rule_revision="post-hoc-v2",
        )
    with pytest.raises(ValueError, match="runtime snapshot"):
        evaluate_detector(
            (record,),
            (_annotation("frame-1", frame_hash="1" * 64),),
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot={},
            evidence_scope="software_fixture_only",
        )
    valid_records, valid_annotations = _with_calibration(
        (record,),
        (_annotation("frame-1", frame_hash="1" * 64),),
        threshold=0.7,
    )
    with pytest.raises(ValueError, match="valid annotation audit"):
        evaluate_detector(
            valid_records,
            valid_annotations,
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
        )


def test_evaluation_set_hash_covers_the_resolved_gold_labels() -> None:
    record = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )
    first_gold = _annotation("frame-1", frame_hash="1" * 64)
    changed_gold = replace(
        first_gold,
        faces=(FaceAnnotation(Box(2, 1, 17, 18), True, _landmarks()),),
    )

    first_records, first_annotations = _with_calibration(
        (record,),
        (first_gold,),
        threshold=0.7,
    )
    changed_records, changed_annotations = _with_calibration(
        (record,),
        (changed_gold,),
        threshold=0.7,
    )
    first = evaluate_detector(
        first_records,
        first_annotations,
        threshold=0.7,
        detector_name="first",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )
    changed = evaluate_detector(
        changed_records,
        changed_annotations,
        threshold=0.7,
        detector_name="changed",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    assert first.evaluation_set_sha256 != changed.evaluation_set_sha256


def test_evaluation_set_hash_covers_the_threshold_calibration_frames() -> None:
    comparison = _candidate(
        "frame-2",
        source="2" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )
    first_calibration = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="calibration",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )
    changed_calibration = _candidate(
        "frame-3",
        source="3" * 64,
        split_role="calibration",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )

    def report(calibration: CandidateFrame) -> DetectorBenchmarkReport:
        records = (calibration, comparison)
        annotations = tuple(
            _annotation(record.frame_id, frame_hash=record.frame_sha256)
            for record in records
        )
        return evaluate_detector(
            records,
            annotations,
            threshold=0.8,
            detector_name=calibration.frame_id,
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )

    assert (
        report(first_calibration).evaluation_set_sha256
        != report(changed_calibration).evaluation_set_sha256
    )


def test_evaluation_set_hash_covers_clip_and_timestamp_sequence_identity() -> None:
    record = _candidate(
        "frame-1",
        source="1" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
    )
    annotation = _annotation("frame-1", frame_hash=record.frame_sha256)

    def report(candidate: CandidateFrame) -> DetectorBenchmarkReport:
        records, annotations = _with_calibration(
            (candidate,),
            (annotation,),
            threshold=0.7,
        )
        return evaluate_detector(
            records,
            annotations,
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )

    original_hash = report(record).evaluation_set_sha256

    assert (
        report(replace(record, clip_id="clip-2")).evaluation_set_sha256 != original_hash
    )
    assert (
        report(replace(record, timestamp_sec=1.0)).evaluation_set_sha256
        != original_hash
    )


def test_evaluation_recomputes_and_binds_the_frozen_calibration_threshold() -> None:
    records = (
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="calibration",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
        ),
        _candidate(
            "frame-2",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.9, _landmarks()),),
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )

    with pytest.raises(ValueError, match="frozen calibrated threshold"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.7,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )
    with pytest.raises(ValueError, match="frozen calibrated threshold"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.8 + 5e-13,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )
    with pytest.raises(ValueError, match="collection threshold"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.8,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
            collection_threshold=0.85,
        )
    with pytest.raises(ValueError, match="collection threshold"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.8,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
            collection_threshold=0.8 + 5e-13,
        )


def test_evaluation_rejects_bootstrap_sample_or_seed_overrides() -> None:
    records = (
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="calibration",
            detections=(Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),),
        ),
        _candidate(
            "frame-2",
            source="2" * 64,
            split_role="comparison",
            detections=(Detection(Box(1, 1, 16, 18), 0.9, _landmarks()),),
        ),
    )
    annotations = tuple(
        _annotation(record.frame_id, frame_hash=record.frame_sha256)
        for record in records
    )

    with pytest.raises(ValueError, match="1,000 fixed"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.8,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
            bootstrap_samples=999,
        )
    with pytest.raises(ValueError, match="fixed bootstrap seed"):
        evaluate_detector(
            records,
            annotations,
            threshold=0.8,
            detector_name="candidate",
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
            bootstrap_seed=7,
        )
