from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from deepfake_detection.dashboard.configuration import dashboard_defaults
from deepfake_detection.dashboard.state import UploadedClip, temporary_video
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.inference.loading import (
    VisualInferenceConfig,
    build_preprocessor,
    load_visual_prediction_engine,
)
from deepfake_detection.inference.predictor import (
    PredictionResult,
    VisualPredictionEngine,
)
from deepfake_detection.views.contracts import PreparedClip

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@st.cache_resource
def load_frozen_visual_engine() -> VisualPredictionEngine:
    defaults = dashboard_defaults(root=Path.cwd())
    return load_visual_prediction_engine(
        VisualInferenceConfig(
            visual_checkpoint=defaults.visual_checkpoint,
            code_version=defaults.code_version,
            expected_checkpoint_sha256=defaults.checkpoint_sha256,
            expected_run_id=defaults.run_id,
            expected_split_hash=defaults.split_hash,
            expected_git_commit=defaults.git_commit,
            expected_seed=defaults.seed,
            threshold=0.5,
            device="cuda",
        )
    )


def predict_upload(clip: UploadedClip) -> PredictionResult:
    engine = load_frozen_visual_engine()
    with temporary_video(clip) as path:
        return engine.predict(path)


def prepare_uploaded_visual(
    clip: UploadedClip,
    *,
    device: str = "cuda",
) -> PreparedClip:
    defaults = dashboard_defaults(root=_PROJECT_ROOT)
    preprocessor = build_preprocessor(
        code_version=defaults.code_version,
        device=device,
        detector="mtcnn",
        tracker="greedy_iou",
        crop_mode="box",
    )
    with temporary_video(clip) as path:
        record = ClipRecord(
            clip_id=clip.sha256,
            dataset="dashboard",
            video_path=path,
            manipulation_type="RealVideo-RealAudio",
            method="unknown",
            source="upload",
            targets=(),
            clip_fake=False,
            video_fake=False,
            audio_fake=False,
        )
        prepared = preprocessor.prepare_visual(record, path)
    if prepared.preprocessing_config_hash != defaults.preprocessing_hash:
        raise ValueError("Runtime preprocessing does not match the checkpoint metadata")
    return prepared


def display_face_frames(view: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(view, dtype=np.float32)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(
        1, 3, 1, 1
    )
    pixels = np.clip((values * standard_deviation + mean) * 255.0, 0, 255)
    hwc = pixels.transpose(0, 2, 3, 1).astype(np.uint8)
    return tuple(hwc)
