"""Export Design B stream embeddings into the same feature store as the branches.

`export_features` next door handles the three Design A branches. It cannot serve
streams: it is welded to exactly three, each with its own call signature and its
own way of turning an output into a logit. Streams are uniform by construction,
so they get a loop over a list rather than three hardcoded blocks.

The view selection is imported from `data.datasets` rather than restated. If
export chose its views separately from training, a stream could be trained on
the mouth crops and scored on the face crops with nothing failing, and the only
symptom would be a number that looked disappointing.

Rows are written with the stream name in `branch`, so `FeatureStore.assemble`
and `ddf train fusion --model deep --branches ...` read them unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepfake_detection.data.datasets import (
    STREAM_FRAME_LIMITS,
    STREAM_VIEWS,
    _normalize_waveform,
)
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.cache_store import CacheStore

from .store import FeatureRecord, FeatureStore

# A visual stream reads one view and takes no audio, so it is not in
# STREAM_VIEWS. Naming it here keeps the two kinds in one table.
VISUAL_VIEW = "visual_view"


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """One trained stream to export.

    `kind` is "visual" for a `VisualStream`, which takes frames alone and
    returns `(logit, embedding)`, or the name of an audiovisual stream in
    STREAM_VIEWS, which takes video and audio by keyword and returns a
    `StreamOutput`.
    """

    name: str
    model: nn.Module
    checkpoint_hash: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind != "visual" and self.kind not in STREAM_VIEWS:
            raise ValueError(
                f"Stream {self.name!r} has unknown kind {self.kind!r}. Expected "
                f"'visual' or one of {', '.join(sorted(STREAM_VIEWS))}."
            )


@dataclass(frozen=True, slots=True)
class StreamExportReport:
    clips: int
    exported_rows: int
    unavailable_rows: int
    streams: tuple[str, ...] = ()
    failures: dict[str, str] = field(default_factory=dict)


def _tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
    )


def _normalized_audio(values: np.ndarray, device: str) -> torch.Tensor:
    """The same normalisation the dataset applies, imported rather than restated.

    Writing it out again here is how this goes wrong: a first draft of this file
    used peak normalisation while the dataset centres and divides by standard
    deviation. A stream trained on one and scored on the other sees a different
    input distribution, and nothing about that fails loudly.
    """
    return _normalize_waveform(_tensor(values, device))


def _forward(spec: StreamSpec, prepared, device: str) -> tuple[float, tuple[float, ...]]:
    """Run one stream over one prepared clip. Raises if a view is missing."""
    if spec.kind == "visual":
        frames = getattr(prepared, VISUAL_VIEW)
        if frames is None:
            raise LookupError(VISUAL_VIEW)
        logit, embedding = spec.model(_tensor(frames, device))
        return float(logit[0].cpu()), tuple(
            float(value) for value in embedding[0].cpu().flatten()
        )

    video_field, audio_field = STREAM_VIEWS[spec.kind]
    video = getattr(prepared, video_field)
    audio = getattr(prepared, audio_field)
    if video is None:
        raise LookupError(video_field)
    if audio is None:
        raise LookupError(audio_field)
    limit = STREAM_FRAME_LIMITS[spec.kind]
    if limit is not None:
        video = video[:limit]
    output = spec.model(
        video=_tensor(video, device), audio=_normalized_audio(audio, device)
    )
    return float(output.logit[0].cpu()), tuple(
        float(value) for value in output.embedding[0].cpu().flatten()
    )


def export_stream_features(
    *,
    records: Sequence[ClipRecord],
    cache_index: Mapping[str, Path],
    cache_store: CacheStore,
    feature_store: FeatureStore,
    streams: Sequence[StreamSpec],
    split_hash: str,
    preprocessing_hash: str,
    partition_role: str,
    run_id: str,
    device: str,
) -> StreamExportReport:
    if not streams:
        raise ValueError("At least one stream is required")
    names = [spec.name for spec in streams]
    if len(set(names)) != len(names):
        raise ValueError(f"Stream names must be unique: {', '.join(sorted(names))}")
    for spec in streams:
        spec.model.to(device).eval()

    rows: list[FeatureRecord] = []
    unavailable = 0
    failures: dict[str, str] = {}

    def append(
        record: ClipRecord,
        prepared,
        name: str,
        checkpoint_hash: str,
        *,
        available: bool,
        logit: float = 0.0,
        embedding: tuple[float, ...] = (),
    ) -> None:
        nonlocal unavailable
        if not available:
            unavailable += 1
        quality = prepared.quality if prepared is not None else None
        rows.append(
            FeatureRecord(
                dataset=record.dataset,
                clip_id=record.clip_id,
                segment_id="clip",
                branch=name,
                logit=logit,
                embedding=embedding,
                available=available,
                checkpoint_hash=checkpoint_hash,
                preprocessing_hash=preprocessing_hash,
                split_hash=split_hash,
                run_id=run_id,
                label=int(record.clip_fake),
                quality_flags=(
                    quality.full_fusion_blockers() if quality is not None else ()
                ),
                face_coverage=quality.face_coverage if quality is not None else 0.0,
                audio_clipped=quality.audio_clipped if quality is not None else False,
                av_duration_delta_sec=(
                    quality.av_duration_delta_sec if quality is not None else 0.0
                ),
                cache_fingerprint=(
                    prepared.preprocessing_fingerprint if prepared is not None else ""
                ),
                source_identity=record.source,
                method=record.method,
                race=record.race,
                gender=record.gender,
                partition_role=partition_role,
            )
        )

    with torch.inference_mode():
        for record in records:
            prepared = None
            reason = None
            if record.clip_id not in cache_index:
                reason = "missing_cache_entry"
            else:
                try:
                    prepared = cache_store.load(cache_index[record.clip_id])
                except (OSError, ValueError) as error:
                    reason = f"cache_load_failed: {error}"
            if prepared is not None:
                if prepared.preprocessing_config_hash != preprocessing_hash:
                    raise ValueError(
                        f"Cache entry {record.clip_id} uses a different "
                        "preprocessing hash"
                    )
            if reason is not None:
                # The whole clip is unreadable, so every stream abstains on it.
                # The row still goes in: a clip nothing can read is a clip that
                # was seen and could not be judged, which is what the abstention
                # rate is counting.
                failures[record.clip_id] = reason
                for spec in streams:
                    append(
                        record, None, spec.name, spec.checkpoint_hash, available=False
                    )
                continue

            for spec in streams:
                try:
                    logit, embedding = _forward(spec, prepared, device)
                except LookupError as missing:
                    failures[f"{record.clip_id}:{spec.name}"] = (
                        f"missing_view: {missing.args[0]}"
                    )
                    append(
                        record,
                        prepared,
                        spec.name,
                        spec.checkpoint_hash,
                        available=False,
                    )
                    continue
                append(
                    record,
                    prepared,
                    spec.name,
                    spec.checkpoint_hash,
                    available=True,
                    logit=logit,
                    embedding=embedding,
                )

    feature_store.write(rows)
    return StreamExportReport(
        clips=len(records),
        exported_rows=len(rows),
        unavailable_rows=unavailable,
        streams=tuple(names),
        failures=failures,
    )
