import pytest

from deepfake_detection.fusion.late import FusionArtifact, FusionSample, LateFusion


def make_sample(value: float) -> FusionSample:
    return FusionSample(
        branch_logits={
            "visual": value,
            "audio": value * 0.8,
            "sync": value * 1.2,
        },
        face_coverage=0.95,
        audio_clipped=False,
        av_duration_delta_sec=0.01,
    )


def test_late_fusion_calibrates_each_branch_before_combining() -> None:
    samples = [make_sample(value) for value in (-4, -3, -2, -1, 1, 2, 3, 4)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    fusion = LateFusion(branch_names=("visual", "audio", "sync"))

    fusion.fit(samples, labels)

    probabilities = fusion.predict_proba([make_sample(-2), make_sample(2)])
    assert probabilities[0] < 0.5
    assert probabilities[1] > 0.5


def test_late_fusion_rejects_missing_branches_instead_of_dropping_rows() -> None:
    fusion = LateFusion(branch_names=("visual", "audio", "sync"))
    incomplete = FusionSample(
        branch_logits={"visual": 1.0, "audio": 1.0},
        face_coverage=0.95,
        audio_clipped=False,
        av_duration_delta_sec=0.01,
    )

    with pytest.raises(ValueError, match="Missing branch logits: sync"):
        fusion.fit([incomplete, incomplete], [0, 1])


def test_fusion_artifact_rejects_wrong_split_or_preprocessing() -> None:
    artifact = FusionArtifact(
        model=LateFusion(branch_names=("visual", "audio", "sync")),
        split_hash="split-1",
        preprocessing_hash="prep-1",
    )

    with pytest.raises(ValueError, match="split hash"):
        artifact.validate_provenance(
            split_hash="split-2",
            preprocessing_hash="prep-1",
        )
    with pytest.raises(ValueError, match="preprocessing hash"):
        artifact.validate_provenance(
            split_hash="split-1",
            preprocessing_hash="prep-2",
        )


def test_small_mlp_fusion_ablation_produces_ordered_probabilities() -> None:
    samples = [make_sample(value) for value in (-4, -3, -2, -1, 1, 2, 3, 4)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    fusion = LateFusion(
        branch_names=("visual", "audio", "sync"),
        classifier_kind="mlp",
    ).fit(samples, labels)

    probabilities = fusion.predict_proba([make_sample(-2), make_sample(2)])

    assert probabilities[0] < probabilities[1]
