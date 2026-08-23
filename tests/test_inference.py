from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepfake_detection.branches.contracts import BranchOutput
from deepfake_detection.branches.sync import SyncOutput
from deepfake_detection.inference.predictor import PredictionEngine
from deepfake_detection.views.contracts import PreparedClip, QualityReport


class FixturePreprocessor:
    def __init__(self, prepared: PreparedClip) -> None:
        self.prepared = prepared

    def prepare(self, record, media_path: Path) -> PreparedClip:
        return self.prepared


class FixtureBranch(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, values: torch.Tensor) -> BranchOutput:
        batch = values.shape[0]
        return BranchOutput(
            logits=torch.full((batch,), self.logit),
            embedding=torch.zeros(batch, 2),
            token_count=1,
        )


class FixtureSync(nn.Module):
    def forward(
        self, *, mouth_video: torch.Tensor, waveform: torch.Tensor
    ) -> SyncOutput:
        batch = mouth_video.shape[0]
        offset_logits = torch.zeros(batch, 8)
        offset_logits[:, 7] = 2.0
        return SyncOutput(
            video_tokens=torch.zeros(batch, 2, 2),
            audio_tokens=torch.zeros(batch, 2, 2),
            offset_logits=offset_logits,
            aligned_similarity=torch.zeros(batch, 2),
        )


class FixtureFusion:
    def predict_proba(self, samples):
        return np.asarray([0.8 for _ in samples])


def prepared_clip(*, complete: bool) -> PreparedClip:
    return PreparedClip(
        clip_id="clip-1",
        visual_view=np.zeros((16, 3, 8, 8), dtype=np.float32),
        audio_view=(np.zeros((64_000,), dtype=np.float32) if complete else None),
        sync_video_view=(
            np.zeros((50, 3, 8, 8), dtype=np.float32) if complete else None
        ),
        sync_audio_view=(np.zeros((32_000,), dtype=np.float32) if complete else None),
        quality=QualityReport(
            face_coverage=1.0,
            stable_face_track=True,
            audio_present=complete,
            audio_clipped=False,
            av_duration_delta_sec=0.0,
        ),
        preprocessing_fingerprint="fixture",
    )


def build_engine(prepared: PreparedClip) -> PredictionEngine:
    return PredictionEngine(
        preprocessor=FixturePreprocessor(prepared),
        visual_model=FixtureBranch(1.0),
        audio_model=FixtureBranch(1.5),
        sync_model=FixtureSync(),
        fusion=FixtureFusion(),
        threshold=0.5,
        device="cpu",
    )


def test_complete_prediction_uses_real_branch_outputs_and_fusion() -> None:
    result = build_engine(prepared_clip(complete=True)).predict(Path("clip.mp4"))

    assert result.verdict == "fake"
    assert result.probability == 0.8
    assert set(result.branch_logits) == {"visual", "audio", "sync"}
    assert result.blockers == ()


def test_missing_audio_returns_indeterminate_without_a_full_model_score() -> None:
    result = build_engine(prepared_clip(complete=False)).predict(Path("clip.mp4"))

    assert result.verdict == "indeterminate"
    assert result.probability is None
    assert "missing_audio" in result.blockers
