"""Lazy singletons for the heavy ML models.

Models are loaded on first use and cached for the process lifetime.
All loaders are thread-safe and tolerate missing weights / offline envs:
on failure they raise; callers should catch and degrade gracefully.
"""
from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_device: Any = None
_ai_classifier: tuple[Any, Any, dict] | None = None
_whisper: tuple[Any, Any] | None = None
_face_detector: Any = None

# Allow pinning to CPU explicitly for low-RAM hosts.
_FORCE_CPU = os.getenv("VERITAS_FORCE_CPU", "0") == "1"

# Pretrained AI-image classifier. Defaults to a Swin-v2 model trained on
# SDXL outputs vs. real photographs. Override with VERITAS_AI_IMAGE_MODEL.
_AI_IMAGE_MODEL = os.getenv("VERITAS_AI_IMAGE_MODEL", "Organika/sdxl-detector")


def get_device():
    global _device
    if _device is None:
        import torch
        if _FORCE_CPU or not torch.cuda.is_available():
            _device = torch.device("cpu")
        else:
            _device = torch.device("cuda")
    return _device


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
