from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .alignment import LANDMARK_TEMPLATE_REVISION
from .timeline import ViewConfig


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_config_hash(
    *,
    config: ViewConfig,
    code_version: str,
) -> str:
    payload = {
        "code_version": code_version,
        "config": asdict(config),
        "landmark_template_revision": LANDMARK_TEMPLATE_REVISION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_fingerprint(
    media_path: Path,
    *,
    dataset: str,
    config: ViewConfig,
    code_version: str,
    leading_silence_sec: float = 0.0,
    sync_start_sec: float = 0.0,
) -> str:
    payload = {
        "preprocessing_config_hash": preprocessing_config_hash(
            config=config,
            code_version=code_version,
        ),
        "dataset": dataset,
        "leading_silence_sec": leading_silence_sec,
        "media_sha256": _content_hash(media_path),
    }
    # Added to the payload only when a dataset actually overrides the window,
    # which keeps every fingerprint written before this option existed
    # byte-identical. A new key present on every clip would have invalidated the
    # whole cache to record a value that is zero almost everywhere.
    #
    # Two windows cut from one clip must not collide, so the offset has to be
    # part of the key: the same media file, same config and same dataset would
    # otherwise map both to one entry and the second would overwrite the first.
    if sync_start_sec:
        payload["sync_start_sec"] = sync_start_sec
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
