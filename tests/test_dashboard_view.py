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


def _multimodal(kind: str, logits: dict[str, float], threshold: float):
    return PredictionResult(
        clip_id="clip-1",
        verdict="fake",
        probability=0.91,
        branch_logits=logits,
        blockers=(),
        preprocessing_fingerprint="prep",
        media_kind=kind,
        threshold=threshold,
    )


def test_the_view_names_the_streams_that_ran_not_a_hardcoded_one() -> None:
    view = build_view_model(
        _multimodal(
            "video",
            {
                "visual-efficientnet": 1.0,
                "stream-emotion": 0.4,
                "final-audio-seed17": -0.2,
            },
            0.5,
        )
    )

    assert view.mode_label == "Five streams fused"
    # Lip-sync could have run on a video and did not, so the reader is told.
    assert view.channels == {
        "Visual": "available",
        "Audio": "available",
        "Lip-sync": "missing",
        "Emotion": "available",
    }


def test_an_image_verdict_is_marked_limited_and_carries_its_own_threshold() -> None:
    view = build_view_model(_multimodal("image", {"visual-dinov3": 1.4}, 0.7631))

    assert view.channels == {"Visual": "limited"}
    assert view.threshold_label == "Decision threshold for image: 0.76"
    assert "One frame" in view.limitations[0]


def test_an_audio_verdict_reports_only_the_audio_channel() -> None:
    view = build_view_model(_multimodal("audio", {"final-audio-seed17": 0.2}, 0.2619))

    assert view.channels == {"Audio": "available"}
    assert view.mode_label == "Audio stream only"
    assert view.threshold_label == "Decision threshold for audio: 0.26"
