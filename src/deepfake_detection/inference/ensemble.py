"""Several visual checkpoints scoring one clip, averaged in logit space.

Seed alone moves the DFDC number of one visual branch by 0.0733 (F21), and the
reported seed is the best draw. Averaging every trained checkpoint removes the
choice of seed from the claim. The face crops are prepared once and shared, so
the members differ only in their weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deepfake_detection.branches.registry import load_branch
from deepfake_detection.views.cache import preprocessing_config_hash

from .loading import build_preprocessor
from .predictor import PredictionResult, _InferenceRecord


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    name: str
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class EnsemblePrediction:
    result: PredictionResult
    # Each member's own probability, so a reader can see whether the members
    # agree or one of them carries the verdict.
    member_probabilities: dict[str, float]
    # The normalised face crops the members saw, (frames, 3, H, W), or None
    # when no face was found.
    visual_view: np.ndarray | None


def program_members(checkpoint_dir: Path) -> tuple[EnsembleMember, ...]:
    """The six final visual checkpoints of the program run, seeds 17, 29, 43."""
    members = []
    for architecture, stem in (
        ("efficientnet-b0", "final-visual"),
        ("convnext-tiny", "final-visual-convnext"),
    ):
        for seed in (17, 29, 43):
            members.append(
                EnsembleMember(
                    name=f"{architecture}-seed{seed}",
                    checkpoint=checkpoint_dir / f"{stem}-seed{seed}.pt",
                )
            )
    return tuple(members)


class VisualEnsembleEngine:
    def __init__(
        self,
        *,
        preprocessor: Any,
        models: Sequence[tuple[str, nn.Module]],
        threshold: float,
        device: str,
    ) -> None:
        if not models:
            raise ValueError("An ensemble needs at least one member")
        if not 0 <= threshold <= 1:
            raise ValueError("Prediction threshold must be in [0, 1]")
        self.preprocessor = preprocessor
        self.models = tuple((name, model.to(device).eval()) for name, model in models)
        self.threshold = threshold
        self.device = device

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.models)

    def predict(
        self,
        video_path: Path,
        *,
        threshold: float | None = None,
        members: Sequence[str] | None = None,
    ) -> EnsemblePrediction:
        """Score one clip.

        `members` picks a subset by name, so one loaded ensemble also serves a
        single-model view without loading the weights twice.
        """
        cut = self.threshold if threshold is None else threshold
        if not 0 <= cut <= 1:
            raise ValueError("Prediction threshold must be in [0, 1]")
        chosen = self.models
        if members is not None:
            wanted = set(members)
            unknown = wanted - set(self.member_names)
            if unknown:
                raise ValueError(f"Unknown ensemble members: {sorted(unknown)}")
            chosen = tuple(item for item in self.models if item[0] in wanted)

        record = _InferenceRecord(clip_id=video_path.stem)
        prepared = self.preprocessor.prepare_visual(record, video_path)
        if prepared.visual_view is None:
            return EnsemblePrediction(
                result=PredictionResult(
                    clip_id=prepared.clip_id,
                    verdict="indeterminate",
                    probability=None,
                    branch_logits={},
                    blockers=("missing_visual",),
                    preprocessing_fingerprint=prepared.preprocessing_fingerprint,
                    media_kind="video",
                    threshold=cut,
                ),
                member_probabilities={},
                visual_view=None,
            )

        view = np.asarray(prepared.visual_view, dtype=np.float32)
        values = torch.from_numpy(view).unsqueeze(0).to(self.device)
        logits: dict[str, float] = {}
        with torch.inference_mode():
            for name, model in chosen:
                logits[name] = float(model(values).logits[0].cpu())
        mean_logit = float(np.mean(list(logits.values())))
        probability = float(1 / (1 + np.exp(-mean_logit)))
        return EnsemblePrediction(
            result=PredictionResult(
                clip_id=prepared.clip_id,
                verdict="fake" if probability >= cut else "real",
                probability=probability,
                branch_logits=logits,
                blockers=(),
                preprocessing_fingerprint=prepared.preprocessing_fingerprint,
                media_kind="video",
                threshold=cut,
            ),
            member_probabilities={
                name: float(1 / (1 + np.exp(-logit))) for name, logit in logits.items()
            },
            visual_view=view,
        )


def load_visual_ensemble(
    members: Sequence[EnsembleMember],
    *,
    code_version: str,
    device: str = "cuda",
    threshold: float = 0.5,
) -> VisualEnsembleEngine:
    """Build every member and refuse any that was trained on other crops.

    A member whose preprocessing hash differs from the runtime preprocessor's
    would be scored on inputs it never saw in training, which is the silent
    mismatch the frozen-baseline checks exist to stop.
    """
    if not members:
        raise ValueError("An ensemble needs at least one member")
    preprocessor = build_preprocessor(code_version=code_version, device=device)
    runtime_hash = preprocessing_config_hash(
        config=preprocessor.config, code_version=code_version
    )
    models: list[tuple[str, nn.Module]] = []
    for member in members:
        model, state = load_branch(member.checkpoint, "visual")
        if state.metadata.preprocessing_hash != runtime_hash:
            raise ValueError(
                f"{member.name} was trained on different preprocessing than the "
                "runtime builds"
            )
        models.append((member.name, model))
    return VisualEnsembleEngine(
        preprocessor=preprocessor,
        models=models,
        threshold=threshold,
        device=device,
    )
