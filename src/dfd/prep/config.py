"""Preprocessing configurations and their hashes, the `preprocess_id` of D-028.

The video id is the hash of the identity of a crop folder (the content of its params.json, written by dfd.prep.pipeline), so a checkpoint
trained on crops made with these parameters has the same id as the folder (dfd.data.loaders.preprocess_id). Image and audio have their
own small configurations. Changing any value gives a new id and the real adapter refuses a checkpoint with another id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from dfd.prep.pipeline import Params


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def video_config(params: Params | None = None) -> dict:
    p = params or Params()
    return {k: v for k, v in asdict(p).items() if k not in ("min_age_s", "model_dir")}  # same identity as pipeline.run


def image_config(size: int = 224) -> dict:
    """Whole-image path (T-I): the encoder's own model-card rule (resize, crop, normalisation) applied to the full-resolution image;
    `size` is the fallback square used only by an encoder without its own processor."""
    return {"mode": "whole_image", "rule": "encoder model-card processor", "fallback_size": size}


def audio_config(sr: int = 16000, window_samples: int = 64_600, trim_normalise: bool = True) -> dict:
    """Speech path (T-A). With `trim_normalise` (the default since E-007) leading and trailing samples at or below 0.005 of full scale are removed and the
    clip is divided by its peak before the window is cut (dfd.prep.audio.trim_and_peak_normalise); without it the configuration is exactly the one of the
    I-0 checkpoint (first window only), kept so that a version 1 checkpoint is still recognised."""
    cfg = {"sr": sr, "window_samples": window_samples, "channels": "mono", "crop": "first", "pad": "repeat"}
    if trim_normalise:
        cfg.update({"trim_silence_abs": 0.005, "peak_normalise": True})
    return cfg


def video_preprocess_id(params: Params | None = None) -> str:
    return config_hash(video_config(params))


def image_preprocess_id(size: int = 224) -> str:
    return config_hash(image_config(size))


def audio_preprocess_id(sr: int = 16000, window_samples: int = 64_600, trim_normalise: bool = True) -> str:
    return config_hash(audio_config(sr, window_samples, trim_normalise))


IN_USE = {"T-V": video_preprocess_id, "T-I": image_preprocess_id, "T-A": audio_preprocess_id}

# Preprocessing the code can still apply exactly as a checkpoint was trained with (D-028 item 4). IN_USE is what a new checkpoint is built with;
# a checkpoint whose id is in ACCEPTED but not in use is scored with the preprocessing it records, never with another one. An id in neither is refused.
ACCEPTED = {
    "T-V": lambda: {video_preprocess_id()},
    "T-I": lambda: {image_preprocess_id()},
    "T-A": lambda: {audio_preprocess_id(), audio_preprocess_id(trim_normalise=False)},
}
