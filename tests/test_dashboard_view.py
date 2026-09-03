from deepfake_detection.dashboard.view_model import build_view_model
from deepfake_detection.inference.predictor import PredictionResult


def test_dashboard_view_marks_incomplete_evidence_before_showing_scores() -> None:
    result = PredictionResult(
        clip_id="clip-1",
        verdict="indeterminate",
        probability=None,
        branch_logits={"visual": 1.2},
        blockers=("missing_audio", "missing_sync_branch"),
        preprocessing_fingerprint="prep",
    )

    view = build_view_model(result, threshold=0.5)

    assert view.title == "Evidence incomplete"
    assert view.channels == {"visual": "available"}
    assert view.final_score == "Not issued"
    assert view.threshold_label == "Fixed decision threshold: 0.50"


def test_visual_only_view_names_its_limited_evidence_scope() -> None:
    result = PredictionResult(
        clip_id="clip-1",
        verdict="real",
        probability=0.125,
        branch_logits={"visual": -1.946},
        blockers=(),
        preprocessing_fingerprint="prep",
    )

    view = build_view_model(result, threshold=0.5)

    assert view.mode_label == "Visual-only development baseline"
    assert view.channels == {"visual": "available"}
    assert view.final_score == "12.5%"
    assert view.limitations == (
        "Validated on a source-disjoint FakeAVCeleb development split only.",
        "This score does not establish cross-dataset generalization.",
    )


def test_dashboard_view_defaults_to_the_frozen_threshold() -> None:
    result = PredictionResult(
        clip_id="clip-1",
        verdict="fake",
        probability=0.75,
        branch_logits={"visual": 1.099},
        blockers=(),
        preprocessing_fingerprint="prep",
    )

    view = build_view_model(result)

    assert view.threshold_label == "Fixed decision threshold: 0.50"
