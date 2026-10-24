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
    return {"mode": "whole_image", "size": size, "interpolation": "area"}


def audio_config(sr: int = 16000, seconds: float = 4.0) -> dict:
    return {"sr": sr, "seconds": seconds, "channels": "mono", "crop": "centre", "pad": "zeros"}


def video_preprocess_id(params: Params | None = None) -> str:
    return config_hash(video_config(params))


def image_preprocess_id(size: int = 224) -> str:
    return config_hash(image_config(size))


def audio_preprocess_id(sr: int = 16000, seconds: float = 4.0) -> str:
    return config_hash(audio_config(sr, seconds))


IN_USE = {"T-V": video_preprocess_id, "T-I": image_preprocess_id, "T-A": audio_preprocess_id}
