from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deepfake_detection.branches.sync_objective import sync_anomaly_logit
from deepfake_detection.fusion.late import FusionSample


@dataclass(frozen=True, slots=True)
class _InferenceRecord:
    clip_id: str
    dataset: str = "inference"
    leading_silence_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class PredictionResult:
    clip_id: str
    verdict: str
    probability: float | None
    branch_logits: dict[str, float]
    blockers: tuple[str, ...]
    preprocessing_fingerprint: str


class PredictionEngine:
    def __init__(
        self,
        *,
        preprocessor: Any,
        visual_model: nn.Module,
        audio_model: nn.Module,
        sync_model: nn.Module,
        fusion: Any,
        threshold: float,
        device: str,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Prediction threshold must be in [0, 1]")
        self.preprocessor = preprocessor
        self.visual_model = visual_model.to(device).eval()
        self.audio_model = audio_model.to(device).eval()
        self.sync_model = sync_model.to(device).eval()
        self.fusion = fusion
        self.threshold = threshold
        self.device = device

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.asarray(values, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

    def predict(self, video_path: Path) -> PredictionResult:
        record = _InferenceRecord(clip_id=video_path.stem)
        prepared = self.preprocessor.prepare(record, video_path)
        branch_logits: dict[str, float] = {}
        with torch.inference_mode():
            if prepared.visual_view is not None:
                visual = self.visual_model(self._tensor(prepared.visual_view))
                branch_logits["visual"] = float(visual.logits[0].cpu())
            if prepared.audio_view is not None:
                audio = self.audio_model(self._tensor(prepared.audio_view))
                branch_logits["audio"] = float(audio.logits[0].cpu())
            if (
                prepared.sync_video_view is not None
                and prepared.sync_audio_view is not None
            ):
                sync = self.sync_model(
                    mouth_video=self._tensor(prepared.sync_video_view),
                    waveform=self._tensor(prepared.sync_audio_view),
                )
                branch_logits["sync"] = float(
                    sync_anomaly_logit(sync.offset_logits)[0].cpu()
                )

        blockers = list(prepared.quality.full_fusion_blockers())
        missing = [
            name for name in ("visual", "audio", "sync") if name not in branch_logits
        ]
        blockers.extend(
            f"missing_{name}_branch"
            for name in missing
            if f"missing_{name}" not in blockers
        )
        if blockers:
            return PredictionResult(
                clip_id=prepared.clip_id,
                verdict="indeterminate",
                probability=None,
                branch_logits=branch_logits,
                blockers=tuple(dict.fromkeys(blockers)),
                preprocessing_fingerprint=prepared.preprocessing_fingerprint,
            )

        probability = float(
            self.fusion.predict_proba(
                [
                    FusionSample(
                        branch_logits=branch_logits,
                        face_coverage=prepared.quality.face_coverage,
                        audio_clipped=prepared.quality.audio_clipped,
                        av_duration_delta_sec=prepared.quality.av_duration_delta_sec,
                    )
                ]
            )[0]
        )
        return PredictionResult(
            clip_id=prepared.clip_id,
            verdict="fake" if probability >= self.threshold else "real",
            probability=probability,
            branch_logits=branch_logits,
            blockers=(),
            preprocessing_fingerprint=prepared.preprocessing_fingerprint,
        )
