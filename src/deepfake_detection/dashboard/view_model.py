from __future__ import annotations

from dataclasses import dataclass

from deepfake_detection.inference.predictor import PredictionResult


@dataclass(frozen=True, slots=True)
class DashboardView:
    title: str
    verdict: str
    final_score: str
    channels: dict[str, str]
    branch_scores: dict[str, str]
    blockers: tuple[str, ...]
    preprocessing_fingerprint: str


def build_view_model(result: PredictionResult) -> DashboardView:
    titles = {
        "fake": "Likely manipulated",
        "real": "Likely authentic",
        "indeterminate": "Evidence incomplete",
    }
    return DashboardView(
        title=titles[result.verdict],
        verdict=result.verdict,
        final_score=(
            f"{result.probability:.1%}"
            if result.probability is not None
            else "Not issued"
        ),
        channels={
            name: "available" if name in result.branch_logits else "missing"
            for name in ("visual", "audio", "sync")
        },
        branch_scores={
            name: f"{value:+.3f}" for name, value in result.branch_logits.items()
        },
        blockers=result.blockers,
        preprocessing_fingerprint=result.preprocessing_fingerprint,
    )
