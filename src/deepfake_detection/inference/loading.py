from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from deepfake_detection.views.face_detector import MTCNNFaceDetector, YuNetFaceDetector
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
    detector: Literal["mtcnn", "yunet"] = "mtcnn"
    tracker: Literal["greedy_iou", "constant_velocity"] = "greedy_iou"
    crop_mode: Literal["box", "landmark"] = "box"
    model_path: Path | None = None
    expected_model_hash: str | None = None


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_preprocessor(
    *,
    code_version: str,
    device: str = "cpu",
    detector: Literal["mtcnn", "yunet"] = "mtcnn",
    tracker: Literal["greedy_iou", "constant_velocity"] = "greedy_iou",
    crop_mode: Literal["box", "landmark"] = "box",
    model_path: Path | None = None,
    expected_model_hash: str | None = None,
    detector_confidence: float = 0.80,
    remove_leading_silence: bool = True,
) -> Preprocessor:
    if expected_model_hash is not None:
        if len(expected_model_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_model_hash
        ):
            raise ValueError("Expected model hash must be a lowercase SHA-256")
    if detector == "yunet":
        if model_path is None or expected_model_hash is None:
            raise ValueError("YuNet requires a model path and expected model hash")
        actual_hash = _file_sha256(model_path)
        if actual_hash != expected_model_hash:
            raise ValueError("YuNet model hash does not match the expected hash")
        backend = YuNetFaceDetector(
            model_path=model_path,
            confidence=detector_confidence,
        )
    elif detector == "mtcnn":
        if model_path is not None:
            raise ValueError("MTCNN does not accept a model path")
        if expected_model_hash is not None:
            raise ValueError("MTCNN does not accept an expected model hash")
        backend = MTCNNFaceDetector(
            confidence=detector_confidence,
            device=device,
        )
        actual_hash = backend.model_sha256()
    else:
        raise ValueError("Detector must be 'mtcnn' or 'yunet'")
    view_config = ViewConfig(
        detector_confidence=detector_confidence,
        detector=detector,
        detector_model_sha256=actual_hash,
        mouth_crop_mode=crop_mode,
        track_association=tracker,
        track_max_gap=1 if tracker == "constant_velocity" else 0,
        remove_leading_silence=remove_leading_silence,
    )
    return Preprocessor(
        decoder=FFmpegMediaDecoder(),
        detector=backend,
        config=view_config,
        code_version=code_version,
    )


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

    preprocessor = build_preprocessor(
        code_version=config.code_version,
        device=config.device,
        detector=config.detector,
        tracker=config.tracker,
        crop_mode=config.crop_mode,
        model_path=config.model_path,
        expected_model_hash=config.expected_model_hash,
    )
    view_config = preprocessor.config
    if (
        preprocessing_config_hash(
            config=view_config,
            code_version=config.code_version,
        )
        != provenance.preprocessing_hash
    ):
        raise ValueError("Runtime preprocessing does not match the checkpoints")
    return PredictionEngine(
        preprocessor=preprocessor,
        visual_model=visual,
        audio_model=audio,
        sync_model=sync,
        fusion=fusion,
        threshold=config.threshold,
        device=config.device,
    )
