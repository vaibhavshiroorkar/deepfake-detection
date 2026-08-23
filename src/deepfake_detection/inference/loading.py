from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from deepfake_detection.branches.audio import build_wav2vec2_audio_branch
from deepfake_detection.branches.sync import build_sync_branch
from deepfake_detection.branches.visual import build_efficientnet_b0
from deepfake_detection.fusion.late import FusionArtifact
from deepfake_detection.training.checkpoints import (
    load_checkpoint,
    validate_branch_states,
)
from deepfake_detection.views.cache import preprocessing_config_hash
from deepfake_detection.views.face_detector import MTCNNFaceDetector
from deepfake_detection.views.media import FFmpegMediaDecoder
from deepfake_detection.views.preprocessor import Preprocessor
from deepfake_detection.views.timeline import ViewConfig

from .predictor import PredictionEngine


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    visual_checkpoint: Path
    audio_checkpoint: Path
    sync_checkpoint: Path
    fusion_model: Path
    code_version: str
    threshold: float = 0.5
    audio_model: str = "facebook/wav2vec2-base"
    device: str = "cuda"


def load_prediction_engine(config: InferenceConfig) -> PredictionEngine:
    visual = build_efficientnet_b0(pretrained=False)
    audio = build_wav2vec2_audio_branch(
        model_name=config.audio_model,
        pretrained=False,
    )
    sync = build_sync_branch(
        audio_model_name=config.audio_model,
        pretrained=False,
    )
    states = {
        "visual": load_checkpoint(config.visual_checkpoint, model=visual),
        "audio": load_checkpoint(config.audio_checkpoint, model=audio),
        "sync": load_checkpoint(config.sync_checkpoint, model=sync),
    }
    provenance = validate_branch_states(states)
    fusion = joblib.load(config.fusion_model)
    if not isinstance(fusion, FusionArtifact):
        raise ValueError("Fusion model does not contain provenance metadata")
    fusion.validate_provenance(
        split_hash=provenance.split_hash,
        preprocessing_hash=provenance.preprocessing_hash,
    )

    view_config = ViewConfig()
    if (
        preprocessing_config_hash(
            config=view_config,
            code_version=config.code_version,
        )
        != provenance.preprocessing_hash
    ):
        raise ValueError("Runtime preprocessing does not match the checkpoints")
    preprocessor = Preprocessor(
        decoder=FFmpegMediaDecoder(),
        detector=MTCNNFaceDetector(
            confidence=view_config.detector_confidence,
            device=config.device,
        ),
        config=view_config,
        code_version=config.code_version,
    )
    return PredictionEngine(
        preprocessor=preprocessor,
        visual_model=visual,
        audio_model=audio,
        sync_model=sync,
        fusion=fusion,
        threshold=config.threshold,
        device=config.device,
    )
