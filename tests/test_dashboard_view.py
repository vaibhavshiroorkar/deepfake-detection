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

    view = build_view_model(result)

    assert view.title == "Evidence incomplete"
    assert view.channels == {
        "visual": "available",
        "audio": "missing",
        "sync": "missing",
    }
    assert view.final_score == "Not issued"
