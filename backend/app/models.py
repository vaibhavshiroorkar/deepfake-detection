"""Lazy singletons for the heavy ML models.

Models are loaded on first use and cached for the process lifetime.
All loaders are thread-safe and tolerate missing weights / offline envs:
on failure they raise; callers should catch and degrade gracefully.

Phase-2 models
--------------
- **DINOv2-large + head** — primary image classifier (replaces Swin-v2 as default)
- **Swin-v2** — kept as fallback image classifier
- **Whisper-base encoder** — audio feature extractor (unchanged)
- **Whisper classifier head** — trained MLP on Whisper features (new)
- **GPT-2** — text perplexity scoring (new)
- **Temporal Transformer** — video temporal analysis on DINOv2 embeddings (new)
- **MTCNN** — face detection (unchanged)
"""
from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_device: Any = None

# Existing singletons
_ai_classifier: tuple[Any, Any, dict] | None = None
_whisper: tuple[Any, Any] | None = None
_face_detector: Any = None

# Phase-2 singletons
_dinov2_classifier: tuple[Any, Any] | None = None
_whisper_classifier: Any | None = None
_gpt2: tuple[Any, Any] | None = None
_temporal_transformer: Any | None = None

# Allow pinning to CPU explicitly for low-RAM hosts.
_FORCE_CPU = os.getenv("VERITAS_FORCE_CPU", "0") == "1"

# Pretrained AI-image classifier. Defaults to a Swin-v2 model trained on
# SDXL outputs vs. real photographs. Override with VERITAS_AI_IMAGE_MODEL.
_AI_IMAGE_MODEL = os.getenv("VERITAS_AI_IMAGE_MODEL", "Organika/sdxl-detector")

# Phase-2 weight paths (set after training)
_DINOV2_WEIGHTS = os.getenv("VERITAS_DINOV2_WEIGHTS", "")
_WHISPER_CLS_WEIGHTS = os.getenv("VERITAS_WHISPER_WEIGHTS", "")
_TEMPORAL_WEIGHTS = os.getenv("VERITAS_TEMPORAL_WEIGHTS", "")


def get_device():
    global _device
    if _device is None:
        import torch
        if _FORCE_CPU or not torch.cuda.is_available():
            _device = torch.device("cpu")
        else:
            _device = torch.device("cuda")
    return _device


# ---------------------------------------------------------------------------
# Swin-v2  (existing — kept as fallback image classifier)
# ---------------------------------------------------------------------------

def get_ai_image_classifier() -> tuple[Any, Any, dict]:
    """Returns (model, processor, label_map) for the pretrained AI-image
    classifier. label_map is {'ai_index': int, 'real_index': int} so callers
    can pull the right softmax probability without guessing label order."""
    global _ai_classifier
    if _ai_classifier is None:
        with _lock:
            if _ai_classifier is None:
                from transformers import (
                    AutoImageProcessor,
                    AutoModelForImageClassification,
                )
                processor = AutoImageProcessor.from_pretrained(_AI_IMAGE_MODEL)
                model = AutoModelForImageClassification.from_pretrained(
                    _AI_IMAGE_MODEL
                )
                model = model.to(get_device()).eval()

                id2label = getattr(model.config, "id2label", {}) or {}
                ai_idx = real_idx = None
                for idx, label in id2label.items():
                    s = str(label).lower()
                    if any(k in s for k in
                           ("artificial", "fake", "ai", "synth", "generated")):
                        ai_idx = int(idx)
                    elif any(k in s for k in
                             ("human", "real", "authentic", "natural")):
                        real_idx = int(idx)
                # Fall back to binary assumption if labels were ambiguous.
                if ai_idx is None and real_idx is None:
                    ai_idx, real_idx = 0, 1
                elif ai_idx is None:
                    ai_idx = 1 - real_idx
                elif real_idx is None:
                    real_idx = 1 - ai_idx

                _ai_classifier = (
                    model,
                    processor,
                    {"ai_index": ai_idx, "real_index": real_idx,
                     "id2label": id2label, "name": _AI_IMAGE_MODEL},
                )
    return _ai_classifier


# ---------------------------------------------------------------------------
# DINOv2-large + classification head  (Phase 2 — primary image classifier)
# ---------------------------------------------------------------------------

def get_dinov2_classifier() -> tuple[Any, Any]:
    """Returns (DINOv2Classifier, processor).

    The head may be untrained if VERITAS_DINOV2_WEIGHTS is not set —
    callers should check and fall back to Swin-v2 if needed.
    """
    global _dinov2_classifier
    if _dinov2_classifier is None:
        with _lock:
            if _dinov2_classifier is None:
                from .detectors.dinov2_head import load_dinov2_classifier
                model, proc = load_dinov2_classifier(
                    _DINOV2_WEIGHTS or None, get_device()
                )
                _dinov2_classifier = (model, proc)
    return _dinov2_classifier


def dinov2_weights_available() -> bool:
    """True if trained DINOv2 head weights are configured."""
    return bool(_DINOV2_WEIGHTS) and os.path.exists(_DINOV2_WEIGHTS)


# ---------------------------------------------------------------------------
# Whisper-base encoder (unchanged)
# ---------------------------------------------------------------------------

def get_whisper_encoder() -> tuple[Any, Any]:
    """Returns (encoder_module, feature_extractor) for openai/whisper-base."""
    global _whisper
    if _whisper is None:
        with _lock:
            if _whisper is None:
                from transformers import WhisperFeatureExtractor, WhisperModel
                processor = WhisperFeatureExtractor.from_pretrained(
                    "openai/whisper-base"
                )
                full = WhisperModel.from_pretrained("openai/whisper-base")
                encoder = full.encoder.to(get_device()).eval()
                _whisper = (encoder, processor)
    return _whisper


# ---------------------------------------------------------------------------
# Whisper classifier head  (Phase 2 — trained audio classifier)
# ---------------------------------------------------------------------------

def get_whisper_classifier() -> Any:
    """Returns the trained WhisperClassifierHead."""
    global _whisper_classifier
    if _whisper_classifier is None:
        with _lock:
            if _whisper_classifier is None:
                from .detectors.whisper_classifier import load_whisper_classifier
                _whisper_classifier = load_whisper_classifier(
                    _WHISPER_CLS_WEIGHTS or None, get_device()
                )
    return _whisper_classifier


def whisper_classifier_weights_available() -> bool:
    """True if trained Whisper classifier weights are configured."""
    return bool(_WHISPER_CLS_WEIGHTS) and os.path.exists(_WHISPER_CLS_WEIGHTS)


# ---------------------------------------------------------------------------
# GPT-2  (Phase 2 — text perplexity scoring)
# ---------------------------------------------------------------------------

def get_gpt2() -> tuple[Any, Any]:
    """Returns (GPT2LMHeadModel, GPT2Tokenizer), both ready for inference."""
    global _gpt2
    if _gpt2 is None:
        with _lock:
            if _gpt2 is None:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                model = GPT2LMHeadModel.from_pretrained("gpt2")
                model = model.to(get_device()).eval()
                _gpt2 = (model, tokenizer)
    return _gpt2


# ---------------------------------------------------------------------------
# Temporal Transformer  (Phase 2 — video temporal analysis)
# ---------------------------------------------------------------------------

def get_temporal_transformer() -> Any:
    """Returns the TemporalTransformer model."""
    global _temporal_transformer
    if _temporal_transformer is None:
        with _lock:
            if _temporal_transformer is None:
                from .detectors.temporal_transformer import load_temporal_transformer
                _temporal_transformer = load_temporal_transformer(
                    _TEMPORAL_WEIGHTS or None, get_device()
                )
    return _temporal_transformer


def temporal_transformer_weights_available() -> bool:
    """True if trained temporal transformer weights are configured."""
    return bool(_TEMPORAL_WEIGHTS) and os.path.exists(_TEMPORAL_WEIGHTS)


# ---------------------------------------------------------------------------
# MTCNN face detector (unchanged)
# ---------------------------------------------------------------------------

def get_face_detector():
    """Returns an MTCNN instance from facenet-pytorch."""
    global _face_detector
    if _face_detector is None:
        with _lock:
            if _face_detector is None:
                from facenet_pytorch import MTCNN
                _face_detector = MTCNN(
                    image_size=224,
                    margin=20,
                    keep_all=True,
                    post_process=False,
                    device=get_device(),
                )
    return _face_detector
