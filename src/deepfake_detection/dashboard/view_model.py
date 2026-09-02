from __future__ import annotations

from dataclasses import dataclass

from deepfake_detection.inference.predictor import PredictionResult


@dataclass(frozen=True, slots=True)
class DashboardView:
    mode_label: str
    title: str
    verdict: str
    final_score: str
    channels: dict[str, str]
    branch_scores: dict[str, str]
    blockers: tuple[str, ...]
    preprocessing_fingerprint: str
    limitations: tuple[str, ...]
    threshold_label: str


def build_view_model(
    result: PredictionResult,
    *,
    threshold: float,
) -> DashboardView:
    titles = {
        "fake": "Likely manipulated",
        "real": "Likely authentic",
        "indeterminate": "Evidence incomplete",
    }
    return DashboardView(
        mode_label="Visual-only development baseline",
        title=titles[result.verdict],
        verdict=result.verdict,
        final_score=(
            f"{result.probability:.1%}"
            if result.probability is not None
            else "Not issued"
        ),
        channels={
            name: "available" if name in result.branch_logits else "missing"
            for name in ("visual",)
        },
        branch_scores={
            name: f"{value:+.3f}" for name, value in result.branch_logits.items()
        },
        blockers=result.blockers,
        preprocessing_fingerprint=result.preprocessing_fingerprint,
        limitations=(
            "Validated on a source-disjoint FakeAVCeleb development split only.",
            "This score does not establish cross-dataset generalization.",
        ),
        threshold_label=f"Fixed decision threshold: {threshold:.2f}",
    )
