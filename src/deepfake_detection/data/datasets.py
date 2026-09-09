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


# Which cached views each audiovisual stream compares. Lip-sync reads the mouth
# crops and the audio cut to the same 2.0 second window, so the two modalities
# are aligned by construction. Emotion reads the face crops against the clip's
# audio window; those two do not share a span, which is why the emotion stream
# has to trim itself to the overlapping prefix.
STREAM_VIEWS = {
    "lipsync": ("sync_video_view", "sync_audio_view"),
    "emotion": ("visual_view", "audio_view"),
}

# How many leading video frames of each stream's view are covered by its audio.
# Lip-sync needs no trim: both of its views are exactly 2.0 seconds from the same
# start, so they are aligned by construction.
#
# Emotion does. `visual_view` spreads 16 frames across the whole clip while
# `audio_view` is a fixed 4.0 second window, so on a longer clip the later face
# frames have no audio to be compared against. Measured on this corpus the
# window covers at worst 51 percent of a clip, so the first 8 frames are inside
# it for every clip; taking more would pair a face against silence that came
# from a different moment, which is exactly the comparison the stream claims to
# make.
STREAM_FRAME_LIMITS = {"lipsync": None, "emotion": 8}


@dataclass(frozen=True, slots=True)
class AVPairItem:
    clip_id: str
    video: Tensor
    audio: Tensor
    label: Tensor


@dataclass(frozen=True, slots=True)
class AVPairBatch:
    clip_ids: tuple[str, ...]
    video: Tensor
    audio: Tensor
    labels: Tensor


def collate_av_pair_items(items: Sequence[AVPairItem]) -> AVPairBatch:
    if not items:
        raise ValueError("Cannot collate an empty audiovisual batch")
    return AVPairBatch(
        clip_ids=tuple(item.clip_id for item in items),
        video=torch.stack([item.video for item in items]),
        audio=torch.stack([item.audio for item in items]),
        labels=torch.stack([item.label for item in items]),
    )


class CachedAVPairDataset(Dataset[AVPairItem]):
    """Paired video and audio for one cross-attention stream, with a clip label.

    Distinct from `CachedGlobalSyncDataset`, which exists for the eight-class
    offset objective and yields an offset class rather than a fake label, and
    hardcodes the mouth views. A stream is a binary classifier over whichever
    pair of views it compares, so the views are a parameter here.

    The label is `clip_fake` rather than `video_fake` or `audio_fake`: a
    cross-modal mismatch can be produced by either side being manipulated, so
    the cue-specific labels do not describe what this stream can see.
    """

    def __init__(
        self,
        *,
        records: Sequence[ClipRecord],
        cache_index: Mapping[str, Path],
        cache_store: CacheStore,
        stream: str = "lipsync",
        preprocessing_hash: str | None = None,
    ) -> None:
        if stream not in STREAM_VIEWS:
            raise ValueError(
                f"Unsupported stream {stream!r}. Expected one of "
                f"{', '.join(sorted(STREAM_VIEWS))}."
            )
        missing = sorted(
            record.clip_id for record in records if record.clip_id not in cache_index
        )
        if missing:
            raise ValueError(f"Missing cache entries: {', '.join(missing)}")
        self.records = tuple(records)
        self.cache_index = dict(cache_index)
        self.cache_store = cache_store
        self.stream = stream
        self.video_field, self.audio_field = STREAM_VIEWS[stream]
        self.frame_limit = STREAM_FRAME_LIMITS[stream]
        self.preprocessing_hash = preprocessing_hash

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AVPairItem:
        record = self.records[index]
        prepared = self.cache_store.load(self.cache_index[record.clip_id])
        if (
            self.preprocessing_hash is not None
            and prepared.preprocessing_config_hash != self.preprocessing_hash
        ):
            raise ValueError(
                f"Cache entry {record.clip_id} uses a different preprocessing hash"
            )
        video = getattr(prepared, self.video_field)
        audio = getattr(prepared, self.audio_field)
        if video is None or audio is None:
            # Abstention, not a fault. A clip with no stable primary face has no
            # video view at all, and a clip with no audio track has no audio
            # view; `ddf manifest usable` filters these out ahead of training so
            # the abstention rate stays reportable.
            raise ValueError(
                f"Cache entry {record.clip_id} has no {self.stream} view "
                f"({self.video_field} or {self.audio_field} is missing)"
            )
        if self.frame_limit is not None:
            video = video[: self.frame_limit]
        return AVPairItem(
            clip_id=record.clip_id,
            video=torch.from_numpy(video).float(),
            audio=_normalize_waveform(torch.from_numpy(audio).float()),
            label=torch.tensor(float(record.clip_fake), dtype=torch.float32),
        )
