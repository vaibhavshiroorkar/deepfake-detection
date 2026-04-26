"""Pre-download model weights at Docker build time.

Running this as a RUN step in the Dockerfile bakes every upstream model
into the container image layer. On HF Spaces the container starts with
weights already on disk, so cold-starts go from "download 5 GB and then
load" to "load." Saves 60 to 80 seconds per cold-start.

If any model is already cached (because HF_HOME is set to a persistent
location) the downloads short-circuit quickly. If a model genuinely
fails to download (offline build, network error) we log and move on so
the build does not fail. The runtime loaders tolerate missing models
and fall back to heuristics.
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("prefetch")

WHISPER_MODEL = os.getenv("VERITAS_WHISPER_MODEL", "openai/whisper-tiny")


def _safe(fn, label: str) -> None:
    try:
        log.info("prefetching %s", label)
        fn()
        log.info("done: %s", label)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not prefetch %s: %s", label, exc)


def main() -> int:
    try:
        from transformers import (
            AutoFeatureExtractor,
            AutoImageProcessor,
            AutoModelForAudioClassification,
            AutoModelForImageClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            GPT2LMHeadModel,
            GPT2TokenizerFast,
            WhisperFeatureExtractor,
            WhisperModel,
        )
    except ImportError as exc:
        log.error("transformers not installed: %s", exc)
        return 0  # do not fail the image build

    image_models = [
        "Organika/sdxl-detector",
        "umm-maybe/AI-image-detector",
    ]
    for name in image_models:
        _safe(lambda n=name: AutoImageProcessor.from_pretrained(n), f"{name} processor")
        _safe(
            lambda n=name: AutoModelForImageClassification.from_pretrained(n),
            f"{name} weights",
        )

    audio_model = os.getenv(
        "VERITAS_AUDIO_DF_MODEL", "motheecreator/Deepfake-audio-detection"
    )
    _safe(
        lambda: AutoFeatureExtractor.from_pretrained(audio_model),
        f"{audio_model} processor",
    )
    _safe(
        lambda: AutoModelForAudioClassification.from_pretrained(audio_model),
        f"{audio_model} weights",
    )

    text_model = os.getenv(
        "VERITAS_TEXT_DF_MODEL", "andreas122001/roberta-mixed-detector"
    )
    _safe(lambda: AutoTokenizer.from_pretrained(text_model), f"{text_model} tokenizer")
    _safe(
        lambda: AutoModelForSequenceClassification.from_pretrained(text_model),
        f"{text_model} weights",
    )

    _safe(
        lambda: WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL),
        f"{WHISPER_MODEL} processor",
    )
    _safe(
        lambda: WhisperModel.from_pretrained(WHISPER_MODEL),
        f"{WHISPER_MODEL} weights",
    )

    _safe(lambda: GPT2TokenizerFast.from_pretrained("gpt2"), "gpt2 tokenizer")
    _safe(lambda: GPT2LMHeadModel.from_pretrained("gpt2"), "gpt2 weights")

    # facenet-pytorch MTCNN pulls its own weights from S3 on first use.
    try:
        from facenet_pytorch import MTCNN

        log.info("prefetching MTCNN weights")
        MTCNN(image_size=224, margin=20, keep_all=True, post_process=False)
        log.info("done: MTCNN")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not prefetch MTCNN: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
