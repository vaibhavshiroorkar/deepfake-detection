from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from deepfake_detection.branches.sync_objective import (
    MISMATCH_CLASS_INDEX,
    OFFSET_MILLISECONDS,
    crop_audio_context,
)
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import QualityReport

from .manifest import ClipRecord


@dataclass(frozen=True, slots=True)
class BranchItem:
    clip_id: str
    values: Tensor
    label: Tensor
    quality: QualityReport


@dataclass(frozen=True, slots=True)
class BranchBatch:
    clip_ids: tuple[str, ...]
    values: Tensor
    labels: Tensor


def collate_branch_items(items: Sequence[BranchItem]) -> BranchBatch:
    if not items:
        raise ValueError("Cannot collate an empty branch batch")
    return BranchBatch(
        clip_ids=tuple(item.clip_id for item in items),
        values=torch.stack([item.values for item in items]),
        labels=torch.stack([item.label for item in items]),
    )


class CachedBranchDataset(Dataset[BranchItem]):
    def __init__(
        self,
        *,
        records: Sequence[ClipRecord],
        cache_index: Mapping[str, Path],
        cache_store: CacheStore,
        branch: str,
        preprocessing_hash: str | None = None,
    ) -> None:
        if branch not in {"visual", "audio"}:
            raise ValueError(f"Unsupported supervised branch: {branch}")
        missing = sorted(
            record.clip_id for record in records if record.clip_id not in cache_index
        )
        if missing:
            raise ValueError(f"Missing cache entries: {', '.join(missing)}")
        self.records = tuple(records)
        self.cache_index = dict(cache_index)
        self.cache_store = cache_store
        self.branch = branch
        self.preprocessing_hash = preprocessing_hash

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BranchItem:
        record = self.records[index]
        prepared = self.cache_store.load(self.cache_index[record.clip_id])
        if (
            self.preprocessing_hash is not None
            and prepared.preprocessing_config_hash != self.preprocessing_hash
        ):
            raise ValueError(
                f"Cache entry {record.clip_id} uses a different preprocessing hash"
            )
        if self.branch == "visual":
            values = prepared.visual_view
            label = record.video_fake
        else:
            values = prepared.audio_view
            label = record.audio_fake
        if values is None:
            raise ValueError(f"Cache entry {record.clip_id} has no {self.branch} view")
        return BranchItem(
            clip_id=record.clip_id,
            values=torch.from_numpy(values).float(),
            label=torch.tensor(float(label), dtype=torch.float32),
            quality=prepared.quality,
        )


@dataclass(frozen=True, slots=True)
class SyncItem:
    clip_id: str
    mouth_video: Tensor
    waveform: Tensor
    offset_class: Tensor


@dataclass(frozen=True, slots=True)
class SyncBatch:
    clip_ids: tuple[str, ...]
    mouth_video: Tensor
    waveform: Tensor
    offset_classes: Tensor


def collate_sync_items(items: Sequence[SyncItem]) -> SyncBatch:
    if not items:
        raise ValueError("Cannot collate an empty sync batch")
    return SyncBatch(
        clip_ids=tuple(item.clip_id for item in items),
        mouth_video=torch.stack([item.mouth_video for item in items]),
        waveform=torch.stack([item.waveform for item in items]),
        offset_classes=torch.stack([item.offset_class for item in items]),
    )


def _normalize_waveform(waveform: Tensor) -> Tensor:
    centered = waveform - waveform.mean()
    standard_deviation = centered.std(unbiased=False)
    if float(standard_deviation) <= 1e-7:
        return torch.zeros_like(waveform)
    return centered / standard_deviation


class CachedSyncDataset(Dataset[SyncItem]):
    def __init__(
        self,
        *,
        records: Sequence[ClipRecord],
        cache_index: Mapping[str, Path],
        cache_store: CacheStore,
        sample_rate: int,
        preprocessing_hash: str | None = None,
    ) -> None:
        authentic = tuple(record for record in records if not record.clip_fake)
        if len({record.source for record in authentic}) < 2:
            raise ValueError("Sync mismatch training requires two authentic identities")
        missing = sorted(
            record.clip_id for record in authentic if record.clip_id not in cache_index
        )
        if missing:
            raise ValueError(f"Missing cache entries: {', '.join(missing)}")
        self.records = authentic
        self.cache_index = dict(cache_index)
        self.cache_store = cache_store
        self.sample_rate = sample_rate
        self.preprocessing_hash = preprocessing_hash
        self.variants = len(OFFSET_MILLISECONDS) + 1

    def __len__(self) -> int:
        return len(self.records) * self.variants

    def _mismatch_record(self, base_index: int) -> ClipRecord:
        source = self.records[base_index].source
        for distance in range(1, len(self.records)):
            candidate = self.records[(base_index + distance) % len(self.records)]
            if candidate.source != source:
                return candidate
        raise RuntimeError("No cross-identity mismatch record is available")

    def _load_prepared(self, record: ClipRecord):
        prepared = self.cache_store.load(self.cache_index[record.clip_id])
        if (
            self.preprocessing_hash is not None
            and prepared.preprocessing_config_hash != self.preprocessing_hash
        ):
            raise ValueError(
                f"Cache entry {record.clip_id} uses a different preprocessing hash"
            )
        return prepared

    def __getitem__(self, index: int) -> SyncItem:
        base_index, variant = divmod(index, self.variants)
        record = self.records[base_index]
        prepared = self._load_prepared(record)
        if prepared.sync_video_view is None or prepared.sync_audio_view is None:
            raise ValueError(f"Cache entry {record.clip_id} has no sync view")
        waveform = torch.from_numpy(prepared.sync_audio_view).float()
        if variant == MISMATCH_CLASS_INDEX:
            mismatch = self._mismatch_record(base_index)
            mismatch_prepared = self._load_prepared(mismatch)
            if mismatch_prepared.sync_audio_view is None:
                raise ValueError(f"Cache entry {mismatch.clip_id} has no sync audio")
            waveform = torch.from_numpy(mismatch_prepared.sync_audio_view).float()
        else:
            if prepared.sync_audio_context is None:
                raise ValueError(
                    f"Cache entry {record.clip_id} has no offset audio context"
                )
            waveform = crop_audio_context(
                torch.from_numpy(prepared.sync_audio_context).float().unsqueeze(0),
                output_samples=len(waveform),
                offset_ms=OFFSET_MILLISECONDS[variant],
                sample_rate=self.sample_rate,
            ).squeeze(0)
        waveform = _normalize_waveform(waveform)
        return SyncItem(
            clip_id=record.clip_id,
            mouth_video=torch.from_numpy(prepared.sync_video_view).float(),
            waveform=waveform,
            offset_class=torch.tensor(variant, dtype=torch.long),
        )


class CachedGlobalSyncDataset(Dataset[SyncItem]):
    def __init__(
        self,
        *,
        records: Sequence[ClipRecord],
        cache_index: Mapping[str, Path],
        cache_store: CacheStore,
        preprocessing_hash: str | None = None,
    ) -> None:
        missing = sorted(
            record.clip_id for record in records if record.clip_id not in cache_index
        )
        if missing:
            raise ValueError(f"Missing cache entries: {', '.join(missing)}")
        self.records = tuple(records)
        self.cache_index = dict(cache_index)
        self.cache_store = cache_store
        self.preprocessing_hash = preprocessing_hash

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SyncItem:
        record = self.records[index]
        prepared = self.cache_store.load(self.cache_index[record.clip_id])
        if (
            self.preprocessing_hash is not None
            and prepared.preprocessing_config_hash != self.preprocessing_hash
        ):
            raise ValueError(
                f"Cache entry {record.clip_id} uses a different preprocessing hash"
            )
        if prepared.sync_video_view is None or prepared.sync_audio_view is None:
            raise ValueError(f"Cache entry {record.clip_id} has no sync view")
        offset_class = (
            MISMATCH_CLASS_INDEX if record.clip_fake else OFFSET_MILLISECONDS.index(0)
        )
        return SyncItem(
            clip_id=record.clip_id,
            mouth_video=torch.from_numpy(prepared.sync_video_view).float(),
            waveform=_normalize_waveform(
                torch.from_numpy(prepared.sync_audio_view).float()
            ),
            offset_class=torch.tensor(offset_class, dtype=torch.long),
        )
