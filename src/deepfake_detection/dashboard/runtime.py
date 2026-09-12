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
from deepfake_detection.inference.multimodal import (
    MultimodalEngine,
    load_multimodal_engine,
)
from deepfake_detection.inference.predictor import (
    PredictionResult,
    VisualPredictionEngine,
)
from deepfake_detection.views.contracts import PreparedClip

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEVICE = "cuda"


def require_cuda() -> None:
    """Fail early, and in a sentence, when this process has no usable CUDA.

    Without this the first `.to("cuda")` raises `AssertionError: Torch not
    compiled with CUDA enabled` from inside a vendored model, which the page's
    handlers do not catch and the reader cannot act on. The two states that
    produce it are a CPU-only install, and a server started before the
    environment was upgraded: a running process keeps the torch it imported, so
    the fix for the second is to restart the server, not to reinstall.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed in this environment. Install it with "
            "`uv sync --extra cu130 --extra ml --extra media --extra dashboard "
            "--extra tracking --group dev`."
        ) from error

    if torch.cuda.is_available():
        return

    built_for_cuda = bool(getattr(torch.version, "cuda", None))
    if not built_for_cuda:
        raise RuntimeError(
            f"CUDA is unavailable: this process is running torch "
            f"{torch.__version__}, a CPU-only build. If the environment was "
            "upgraded after this server started, restart the server. "
            "Otherwise reinstall with `uv sync --extra cu130 --extra ml "
            "--extra media --extra dashboard --extra tracking --group dev`, "
            "and never install the cpu and cu130 extras together."
        )
    raise RuntimeError(
        f"CUDA is unavailable: torch {torch.__version__} is a CUDA build, but "
        "no device is visible. Check the NVIDIA driver with `nvidia-smi`."
    )


@st.cache_resource
def load_frozen_visual_engine() -> VisualPredictionEngine:
    require_cuda()
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
            device=_DEVICE,
        )
    )


@st.cache_resource
def load_multimodal_gate_engine() -> MultimodalEngine:
    """The five-stream engine, or a RuntimeError a reader can act on.

    Cached because loading it means rebuilding five models and reading about
    2.5 GB of checkpoints, which takes long enough that doing it per click would
    look like a hang.
    """
    require_cuda()
    defaults = dashboard_defaults(root=_PROJECT_ROOT)
    if not defaults.gate_fusion_checkpoint.is_file():
        raise RuntimeError(
            "No fusion head is trained for the gate. Build one with "
            "`uv run python scripts/fit_gate_fusion.py --run-dir "
            f"{defaults.stream_run.name}`."
        )
    return load_multimodal_engine(
        run_dir=defaults.stream_run,
        code_version=defaults.stream_code_version,
        fusion_path=defaults.gate_fusion_checkpoint,
        device=_DEVICE,
        root=_PROJECT_ROOT,
    )


def predict_upload(clip: UploadedClip) -> PredictionResult:
    """Score the upload through every stream its media kind can drive.

    Falls back to the frozen visual baseline only when no head is trained, and
    says which one ran through `PredictionResult.media_kind`: a verdict from one
    visual model and a verdict fused from five are different claims.
    """
    defaults = dashboard_defaults(root=_PROJECT_ROOT)
    if not defaults.gate_fusion_checkpoint.is_file():
        # No head trained yet. The frozen visual baseline still answers, and it
        # reports no media kind, which is what makes the page label it as the
        # single-model baseline rather than a fusion.
        return predict_upload_visual_only(clip)
    engine = load_multimodal_gate_engine()
    with temporary_video(clip) as path:
        return engine.predict(path)


def predict_upload_visual_only(clip: UploadedClip) -> PredictionResult:
    engine = load_frozen_visual_engine()
    with temporary_video(clip) as path:
        return engine.predict(path)


def prepare_uploaded_visual(clip: UploadedClip) -> PreparedClip:
    require_cuda()
    defaults = dashboard_defaults(root=_PROJECT_ROOT)
    preprocessor = build_preprocessor(
        code_version=defaults.code_version,
        device=_DEVICE,
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
