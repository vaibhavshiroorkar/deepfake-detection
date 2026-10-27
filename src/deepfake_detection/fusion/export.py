from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepfake_detection.branches.sync_objective import sync_anomaly_logit
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.cache_store import CacheStore

from .store import FeatureRecord, FeatureStore


@dataclass(frozen=True, slots=True)
class ExportReport:
    clips: int
    exported_rows: int
    unavailable_rows: int
    failures: dict[str, str] = field(default_factory=dict)


def _tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
    )


def verify_out_of_fold(
    *,
    records: Sequence[ClipRecord],
    partition_role: str,
    trained_source_identities: Mapping[str, Sequence[str]] | None,
    selection_source_identities: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Keep optimizer training and checkpoint selection out of OOF predictions."""
    if partition_role != "oof":
        return
    if not trained_source_identities or any(
        not values for values in trained_source_identities.values()
    ):
        raise ValueError("Out-of-fold export requires recorded training provenance")
    exported = {record.source for record in records}
    for branch, trained in sorted(trained_source_identities.items()):
        if not trained:
            continue
        leaked = sorted(exported & set(trained))
        if leaked:
            raise ValueError(
                f"Out-of-fold export is contaminated: {len(leaked)} source "
                f"identities appear both in this manifest and in the {branch} "
                f"checkpoint's training set, starting with {leaked[0]}. "
                "A fusion head fitted on these rows would read branch scores "
                "for clips that branch trained on."
            )

    if selection_source_identities is None or set(selection_source_identities) != set(
        trained_source_identities
    ):
        raise ValueError(
            "Out-of-fold export requires checkpoint selection provenance; retrain legacy checkpoints"
        )
    for branch, selected in selection_source_identities.items():
        if selected is None:
            raise ValueError(f"Missing checkpoint selection provenance for {branch}")
        leaked = sorted(exported & set(selected))
        if leaked:
            raise ValueError(
                f"Out-of-fold selection is contaminated for {branch}: {leaked[0]}"
            )


def export_features(
    *,
    records: Sequence[ClipRecord],
    cache_index: Mapping[str, Path],
    cache_store: CacheStore,
    feature_store: FeatureStore,
    visual_model: nn.Module,
    audio_model: nn.Module,
    sync_model: nn.Module,
    checkpoint_hashes: Mapping[str, str],
    split_hash: str,
    preprocessing_hash: str,
    partition_role: str,
    run_id: str,
    device: str,
    trained_source_identities: Mapping[str, Sequence[str]] | None = None,
    selection_source_identities: Mapping[str, Sequence[str]] | None = None,
) -> ExportReport:
    for name in ("visual", "audio", "sync"):
        if name not in checkpoint_hashes:
            raise ValueError(f"Missing checkpoint hash for {name}")
    verify_out_of_fold(
        records=records,
        partition_role=partition_role,
        trained_source_identities=trained_source_identities,
        selection_source_identities=selection_source_identities,
    )
    visual_model.to(device).eval()
    audio_model.to(device).eval()
    sync_model.to(device).eval()
    rows: list[FeatureRecord] = []
    unavailable = 0
    failures: dict[str, str] = {}

    def append_unavailable_clip(record: ClipRecord, reason: str) -> None:
        nonlocal unavailable
        failures[record.clip_id] = reason
        for branch in ("visual", "audio", "sync"):
            unavailable += 1
            rows.append(
                FeatureRecord(
                    dataset=record.dataset,
                    clip_id=record.clip_id,
                    segment_id="clip",
                    branch=branch,
                    logit=0.0,
                    embedding=(),
                    available=False,
                    checkpoint_hash=checkpoint_hashes[branch],
                    preprocessing_hash=preprocessing_hash,
                    split_hash=split_hash,
                    run_id=run_id,
                    label=int(record.clip_fake),
                    quality_flags=("missing_cache",),
                    face_coverage=0.0,
                    source_identity=record.source,
                    method=record.method,
                    race=record.race,
                    gender=record.gender,
                    partition_role=partition_role,
                )
            )

    def append_row(
        record: ClipRecord,
        prepared,
        branch: str,
        *,
        available: bool,
        logit: float = 0.0,
        embedding: tuple[float, ...] = (),
    ) -> None:
        nonlocal unavailable
        if not available:
            unavailable += 1
        rows.append(
            FeatureRecord(
                dataset=record.dataset,
                clip_id=record.clip_id,
                segment_id="clip",
                branch=branch,
                logit=logit,
                embedding=embedding,
                available=available,
                checkpoint_hash=checkpoint_hashes[branch],
                preprocessing_hash=preprocessing_hash,
                split_hash=split_hash,
                run_id=run_id,
                label=int(record.clip_fake),
                quality_flags=prepared.quality.full_fusion_blockers(),
                face_coverage=prepared.quality.face_coverage,
                audio_clipped=prepared.quality.audio_clipped,
                av_duration_delta_sec=prepared.quality.av_duration_delta_sec,
                cache_fingerprint=prepared.preprocessing_fingerprint,
                source_identity=record.source,
                method=record.method,
                race=record.race,
                gender=record.gender,
                partition_role=partition_role,
            )
        )

    with torch.inference_mode():
        for record in records:
            if record.clip_id not in cache_index:
                append_unavailable_clip(record, "missing_cache_entry")
                continue
            try:
                prepared = cache_store.load(cache_index[record.clip_id])
            except (OSError, ValueError) as error:
                append_unavailable_clip(record, f"cache_load_failed: {error}")
                continue
            if prepared.preprocessing_config_hash != preprocessing_hash:
                raise ValueError(
                    f"Cache entry {record.clip_id} uses a different preprocessing hash"
                )
            if prepared.visual_view is None:
                append_row(record, prepared, "visual", available=False)
            else:
                output = visual_model(_tensor(prepared.visual_view, device))
                append_row(
                    record,
                    prepared,
                    "visual",
                    available=True,
                    logit=float(output.logits[0].cpu()),
                    embedding=tuple(
                        float(value) for value in output.embedding[0].cpu().flatten()
                    ),
                )
            if prepared.audio_view is None:
                append_row(record, prepared, "audio", available=False)
            else:
                from deepfake_detection.training.input_contract import audio_forward

                output = audio_forward(
                    audio_model,
                    _tensor(prepared.audio_view, device),
                    getattr(prepared, "audio_valid_samples", None),
                )
                append_row(
                    record,
                    prepared,
                    "audio",
                    available=True,
                    logit=float(output.logits[0].cpu()),
                    embedding=tuple(
                        float(value) for value in output.embedding[0].cpu().flatten()
                    ),
                )
            if prepared.sync_video_view is None or prepared.sync_audio_view is None:
                append_row(record, prepared, "sync", available=False)
            else:
                output = sync_model(
                    mouth_video=_tensor(prepared.sync_video_view, device),
                    waveform=_tensor(prepared.sync_audio_view, device),
                )
                embedding = torch.cat(
                    (
                        output.video_tokens.mean(dim=1),
                        output.audio_tokens.mean(dim=1),
                    ),
                    dim=-1,
                )
                append_row(
                    record,
                    prepared,
                    "sync",
                    available=True,
                    logit=float(sync_anomaly_logit(output.offset_logits)[0].cpu()),
                    embedding=tuple(float(value) for value in embedding[0].cpu()),
                )
    feature_store.write(rows)
    return ExportReport(
        clips=len(records),
        exported_rows=len(rows),
        unavailable_rows=unavailable,
        failures=failures,
    )
