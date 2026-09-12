from __future__ import annotations

from dataclasses import dataclass

from deepfake_detection.dashboard.lib import media_kind
from deepfake_detection.inference.multimodal import _modality
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


_MODE_LABELS = {
    media_kind.VIDEO: "Five streams fused",
    media_kind.IMAGE: "Visual stream only, on a single frame",
    media_kind.AUDIO: "Audio stream only",
}


def _channels(result: PredictionResult) -> dict[str, str]:
    """Every stream the media kind could drive, and whether it produced a score.

    Built from the routing table rather than from the result alone, because the
    interesting cell is the one that should have run and did not: a video whose
    face track was unstable shows visual as missing, and that is the explanation
    for the verdict.
    """
    if not result.media_kind:
        # The frozen Design A path, which runs one model and reports one name.
        return {
            name: "available" if name in result.branch_logits else "missing"
            for name in ("visual",)
        }
    ran = {_modality(name) for name in result.branch_logits}
    return {
        media_kind.STREAM_LABELS[item.stream]: (
            "limited"
            if item.stream in ran and item.degraded
            else "available"
            if item.stream in ran
            else "missing"
        )
        for item in media_kind.availability(result.media_kind)
        if item.available
    }


def _threshold_label(result: PredictionResult, fallback: float) -> str:
    if result.threshold is None:
        return f"Fixed decision threshold: {fallback:.2f}"
    kind = result.media_kind or "this input"
    return f"Decision threshold for {kind}: {result.threshold:.2f}"


def _limitations(result: PredictionResult) -> tuple[str, ...]:
    shared = (
        "Validated on a source-disjoint FakeAVCeleb development split only.",
        "This score does not establish cross-dataset generalization.",
    )
    if result.media_kind == media_kind.IMAGE:
        return (
            "One frame. The visual stream was trained on sixteen frames from "
            "across a clip, so part of what it reads is absent here.",
            *shared,
        )
    if result.media_kind == media_kind.AUDIO:
        return (
            "Sound only. The audio stream is the weakest of the five in-domain "
            "and reads close to chance cross-corpus.",
            *shared,
        )
    return shared


def build_view_model(
    result: PredictionResult,
    *,
    threshold: float = 0.5,
) -> DashboardView:
    titles = {
        "fake": "Likely manipulated",
        "real": "Likely authentic",
        "indeterminate": "Evidence incomplete",
    }
    mode = _MODE_LABELS.get(result.media_kind, "Visual-only development baseline")
    return DashboardView(
        mode_label=mode,
        title=titles[result.verdict],
        verdict=result.verdict,
        final_score=(
            f"{result.probability:.1%}"
            if result.probability is not None
            else "Not issued"
        ),
        channels=_channels(result),
        branch_scores={
            name: f"{value:+.3f}" for name, value in result.branch_logits.items()
        },
        blockers=result.blockers,
        preprocessing_fingerprint=result.preprocessing_fingerprint,
        limitations=_limitations(result),
        threshold_label=_threshold_label(result, threshold),
    )
