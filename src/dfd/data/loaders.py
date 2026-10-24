"""Model-input datasets built from a manifest and a split file (workstream S1).

Every dataset yields a dict with the keys ``x`` (the model input), ``label``, ``group_id`` and
``sample_id``, in manifest order restricted to one split, so a run is reproducible from
(manifest, split file, split name, preprocessing). Nothing here is random except the optional
training-time audio crop, which is seeded by (seed, epoch, index).

    FaceClipDataset   16 face crops per video written by dfd.prep.pipeline (D-034). ``x`` is a uint8
                      tensor [T, 3, H, W] (RGB) and ``mask`` a bool tensor [T]: frames with no
                      detected face are zero and masked out, never dropped silently (D-034 item 3);
                      ``no_face`` is True when no frame has a face (warning NO_FACE_DETECTED, D-027).
    ImageDataset      whole-image input, resized to ``size`` (D-033, the no-face path). The face path
                      of D-033 is not built here.
    AudioDataset      mono float32 at 16 kHz cut or zero-padded to ``seconds`` (waveform ``x`` [L] and
                      ``length``, the number of real samples). Videos use the audio track.

The test splits are refused unless ``final=True`` (CLAUDE.md rule 3); this is the loader-side twin of
the guard in dfd.eval.guard.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dfd.data.manifest import Record, default_data_root, read_manifest
from dfd.data.splits import SPLIT_NAMES, read_split
from dfd.prep.pipeline import folder_name

FINAL_ONLY = ("test_in", "test_x")


class LoaderError(ValueError):
    pass


def preprocess_id(crops_dir: str | Path) -> str:
    """First 12 hex digits of the SHA-256 of the crop folder's params.json (D-028 `preprocess_id`)."""
    return hashlib.sha256(Path(crops_dir, "params.json").read_bytes()).hexdigest()[:12]


def select_records(manifest: str | Path, split_csv: str | Path, split: str, modality: str | None = None,
                   final: bool = False) -> list[Record]:
    """Records of one split in manifest order. Refuses test splits unless final=True."""
    if split not in SPLIT_NAMES:
        raise LoaderError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if split in FINAL_ONLY and not final:
        raise LoaderError(f"split {split!r} is final-only: pass final=True from a frozen iteration (CLAUDE.md rule 3)")
    records = read_manifest(manifest)
    assignment = read_split(split_csv)
    missing = [r.sample_id for r in records if r.sample_id not in assignment]
    if missing and len(missing) == len(records):
        raise LoaderError("split file shares no sample_id with the manifest")
    out = [r for r in records if assignment.get(r.sample_id) == split and (modality is None or r.modality == modality)]
    if not out:
        raise LoaderError(f"no records in split {split!r}" + (f" with modality {modality!r}" if modality else ""))
    return out


class _Base(Dataset):
    def __init__(self, records: Sequence[Record], data_root: str | Path | None):
        self.records = list(records)
        self.data_root = Path(data_root) if data_root is not None else default_data_root()

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records], dtype=np.int64)

    def _meta(self, r: Record) -> dict:
        return {"label": r.label, "group_id": r.group_id, "sample_id": r.sample_id}


class FaceClipDataset(_Base):
    """Face crops of a video, read from the output folder of dfd.prep.pipeline."""

    def __init__(self, records: Sequence[Record], crops_dir: str | Path, n_frames: int = 16, size: int = 224,
                 data_root: str | Path | None = None, require_crops: bool = True):
        super().__init__([r for r in records if r.modality == "video"], data_root)
        self.crops_dir = Path(crops_dir)
        self.n_frames, self.size = n_frames, size
        self.preprocess_id = preprocess_id(self.crops_dir)
        if not self.records:
            raise LoaderError("no video records")
        if require_crops:
            absent = [r.sample_id for r in self.records
                      if not (self.crops_dir / folder_name(r.sample_id) / "_done.json").is_file()]
            if absent:
                raise LoaderError(f"{len(absent)} of {len(self.records)} clips have no finished crops, first: {absent[0]}")

    def _read_index(self, sample_id: str) -> list[dict]:
        f = self.crops_dir / folder_name(sample_id) / "index.csv"
        with open(f, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def __getitem__(self, i: int) -> dict:
        import cv2
        r = self.records[i]
        x = torch.zeros(self.n_frames, 3, self.size, self.size, dtype=torch.uint8)
        mask = torch.zeros(self.n_frames, dtype=torch.bool)
        for row in self._read_index(r.sample_id):
            k = int(row["frame_index"])
            if row["detected"] != "1" or not row["crop_path"] or k >= self.n_frames:
                continue
            img = cv2.imread(str(self.crops_dir / row["crop_path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise LoaderError(f"unreadable crop {row['crop_path']} of {r.sample_id}")
            if img.shape[:2] != (self.size, self.size):
                img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            x[k] = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
            mask[k] = True
        return {"x": x, "mask": mask, "no_face": not bool(mask.any()), **self._meta(r)}


class ImageDataset(_Base):
    """Whole image resized to size x size (D-033 whole-image path)."""

    def __init__(self, records: Sequence[Record], size: int = 224, data_root: str | Path | None = None):
        super().__init__([r for r in records if r.modality == "image"], data_root)
        self.size = size
        if not self.records:
            raise LoaderError("no image records")

    def __getitem__(self, i: int) -> dict:
        import cv2
        r = self.records[i]
        img = cv2.imread(str(self.data_root / r.path), cv2.IMREAD_COLOR)
        if img is None:
            raise LoaderError(f"unreadable image {r.path}")
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return {"x": x, **self._meta(r)}


class AudioDataset(_Base):
    """Mono 16 kHz waveform of an audio file or of the audio track of a video, fixed length."""

    def __init__(self, records: Sequence[Record], seconds: float = 4.0, sr: int = 16000,
                 data_root: str | Path | None = None, train_crop: bool = False, seed: int = 0):
        super().__init__([r for r in records if r.modality in ("audio", "video")], data_root)
        if not self.records:
            raise LoaderError("no audio or video records")
        self.sr, self.n = sr, int(round(seconds * sr))
        self.train_crop, self.seed, self.epoch = train_crop, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, i: int) -> dict:
        from dfd.prep import audio as A
        r = self.records[i]
        path = self.data_root / r.path
        wav = A.extract_audio(path, self.sr) if r.modality == "video" else A.load_audio(path, self.sr)
        if wav.size > self.n:
            if self.train_crop:
                rng = np.random.default_rng([self.seed, self.epoch, i])
                start = int(rng.integers(0, wav.size - self.n + 1))
            else:
                start = (wav.size - self.n) // 2  # deterministic centre crop for evaluation
            wav = wav[start:start + self.n]
        length = int(wav.size)
        x = np.zeros(self.n, np.float32)
        x[:length] = wav
        return {"x": torch.from_numpy(x), "length": length, "silent": length == 0, **self._meta(r)}
