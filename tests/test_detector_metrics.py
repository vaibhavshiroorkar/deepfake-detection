from __future__ import annotations

from dataclasses import replace

import pytest

from deepfake_detection.benchmarks.detector_annotations import (
    FaceAnnotation,
    FrameAnnotation,
)
from deepfake_detection.benchmarks.detector_metrics import (
    FROZEN_DETECTOR_RULE_REVISION,
    CandidateFrame,
    DetectorBenchmarkReport,
    DetectorDecision,
    DetectorLatency,
    DetectorMetrics,
    TrackerMetrics,
    calibrate_detector_threshold,
    compare_detectors,
    evaluate_detector,
)
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


def test_evaluation_is_comparison_only_and_reports_landmark_error() -> None:
    records = (
        _candidate(
            "frame-1",
            source="1" * 64,
            split_role="calibration",
            detections=(Detection(Box(40, 40, 50, 50), 0.99, None),),
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
            detections=(Detection(Box(1, 1, 16, 18), 0.75, None),),
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
    assert report.metrics.landmark_coverage == 0.5
    assert report.latency.median_ms == 4.0
    assert report.latency.p95_ms == pytest.approx(4.9)
    assert report.latency.throughput_fps == 250.0
    assert set(report.intervals) >= {
        "target_recall",
        "false_detections_per_frame",
        "landmark_nme",
        "landmark_coverage",
        "greedy_iou.target_track_errors_per_1000",
        "constant_velocity.target_track_errors_per_1000",
    }
    assert report.bootstrap_samples == 1000
    assert all(
        0 < interval.successful_samples <= 1000
        for interval in report.intervals.values()
    )


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

    report = evaluate_detector(
        records,
        annotations,
        threshold=0.7,
        detector_name="candidate",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )

    for tracker in report.trackers:
        assert tracker.target_track_errors == 5
        assert tracker.target_track_errors_per_1000 == 1000
    assert report.metrics.non_target_candidate_count == 5


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
    return DetectorBenchmarkReport(
        detector_name=name,
        detector_revision=f"{name}-revision",
        model_sha256="a" * 64,
        threshold=0.7,
        collection_threshold=0.1,
        evidence_scope=scope,
        rule_revision=FROZEN_DETECTOR_RULE_REVISION,
        frame_count=400,
        source_count=80,
        metrics=DetectorMetrics(
            target_recall=recall,
            false_detections_per_frame=0.05,
            non_target_detections_per_frame=0.1,
            non_target_candidate_count=40,
            landmark_nme=nme,
            landmark_coverage=1.0,
            aligned_mouth_jitter=0.02,
        ),
        trackers=(
            TrackerMetrics(
                association="greedy_iou",
                stable_track_coverage=0.9,
                abstention_rate=0.1,
                target_track_errors=2,
                tracked_frames=1000,
                target_track_errors_per_1000=track_errors_per_1000,
            ),
            TrackerMetrics(
                association="constant_velocity",
                stable_track_coverage=0.95,
                abstention_rate=0.05,
                target_track_errors=2,
                tracked_frames=1000,
                target_track_errors_per_1000=track_errors_per_1000,
            ),
        ),
        latency=DetectorLatency(
            timed_frames=400,
            median_ms=latency_ms,
            p95_ms=latency_ms + 1,
            throughput_fps=1000 / latency_ms,
            device="cpu",
            thread_count=4,
        ),
        intervals={},
        runtime_snapshot=_runtime_snapshot(),
        raw_results_sha256="f" * 64,
        evaluation_set_sha256=evaluation_set_sha256,
        annotation_audit_validated=True,
    )


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
    with pytest.raises(ValueError, match="valid annotation audit"):
        evaluate_detector(
            (record,),
            (_annotation("frame-1", frame_hash="1" * 64),),
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

    first = evaluate_detector(
        (record,),
        (first_gold,),
        threshold=0.7,
        detector_name="first",
        runtime_snapshot=_runtime_snapshot(),
        evidence_scope="software_fixture_only",
    )
    changed = evaluate_detector(
        (record,),
        (changed_gold,),
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
            threshold=0.7,
            detector_name=calibration.frame_id,
            runtime_snapshot=_runtime_snapshot(),
            evidence_scope="software_fixture_only",
        )

    assert (
        report(first_calibration).evaluation_set_sha256
        != report(changed_calibration).evaluation_set_sha256
    )
