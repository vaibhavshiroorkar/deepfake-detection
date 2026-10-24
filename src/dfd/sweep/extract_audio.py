"""E-003 feature extraction: frozen speech encoders on speech deepfakes (docs/O007_SWEEP.md, D-036).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_audio --cand AU-1

Audio is decoded to mono 16 kHz, cut to the first 64,600 samples (4 seconds) and, when shorter, repeat-padded (the clip is tiled
until it fills the window). Each encoder applies its own feature extractor; the Whisper encoder gets its fixed 30 second window with
the 4 seconds first and only the 202 frames of those 4 seconds are pooled. The feature is the mean and standard deviation over time of
the last hidden layer (2 x hidden size). Jobs: ASVspoof 2019 LA train and dev, In-the-Wild dev_x (12 of 54 speakers, D-035).
Decoded windows are cached as int16 in data/_cache/audio_64600/ so that every encoder reads the same samples. Features go to
data/_features/<cand>/T-A/<dataset>_<split>.npz with the .json sidecar written last (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dfd.data import loaders as L
from dfd.prep import audio as A
from dfd.sweep import encoders

WINDOW = encoders.WINDOW
CACHE = Path("data/_cache/audio_64600")
JOBS = [
    ("asvspoof_train", "data/_manifests/asvspoof2019_la.manifest.csv", "data/_splits/asvspoof2019_la-875bf337c379.csv", "train"),
    ("asvspoof_dev", "data/_manifests/asvspoof2019_la.manifest.csv", "data/_splits/asvspoof2019_la-875bf337c379.csv", "val"),
    ("itw_dev_x", "data/_manifests/in_the_wild.manifest.csv", "data/_splits/in_the_wild-5cf2ea54a724.csv", "dev_x"),
]
OUT = Path("data/_features")


def window_of(wav: np.ndarray) -> np.ndarray:
    """First WINDOW samples; a shorter clip is tiled until it fills the window; an empty clip becomes silence."""
    if wav.size == 0:
        return np.zeros(WINDOW, np.float32)
    if wav.size < WINDOW:
        wav = np.tile(wav, -(-WINDOW // wav.size))
    return wav[:WINDOW].astype(np.float32)


class AudioSet(Dataset):
    def __init__(self, records, root="data", cache: Path | None = CACHE):
        self.records, self.root, self.cache = records, Path(root), cache

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        f = None
        if self.cache is not None:
            f = self.cache / (hashlib.sha1(r.sample_id.encode()).hexdigest() + ".npy")
            if f.is_file():
                return torch.from_numpy(np.load(f).astype(np.float32) / 32767.0), i
        p = self.root / r.path
        wav = window_of(A.extract_audio(p) if r.modality == "video" else A.load_audio(p))
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, np.clip(np.round(wav * 32767.0), -32768, 32767).astype(np.int16))
            wav = np.round(wav * 32767.0).astype(np.float32) / 32767.0  # the same quantised samples whether cached or not
        return torch.from_numpy(wav), i


def run_job(enc, name, manifest, split_csv, split, workers, batch, out_root=OUT, cache=CACHE):
    out = out_root / enc.cid / "T-A" / f"{name}.npz"
    if out.is_file() and out.with_suffix(".json").is_file():
        print("skip", out, flush=True)
        return
    recs = L.select_records(manifest, split_csv, split, modality="audio")
    dl = DataLoader(AudioSet(recs, cache=cache), batch_size=batch, num_workers=workers, shuffle=False)
    feats = np.zeros((len(recs), 2 * enc.dim), np.float16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for x, idx in dl:
        feats[idx.numpy()] = enc(x.numpy()).cpu().numpy().astype(np.float16)
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=feats, label=np.array([r.label for r in recs]), group_id=np.array([r.group_id for r in recs]),
             sample_id=np.array([r.sample_id for r in recs]))
    meta = dict(cand=enc.cid, encoder=enc.name, revision=enc.revision, dim=2 * enc.dim, params=enc.n_params, input_size=WINDOW, dataset=name,
                split=split, n_items=len(recs), seconds=round(dt, 1), items_per_s=round(len(recs) / dt, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                preprocessing="first 64,600 samples at 16 kHz, repeat-padded; own feature extractor; mean and std over time of the last hidden layer"
                              + ("; Whisper: 30 s input, 202 frames pooled" if enc.kind == "whisper" else ""),
                note="end-to-end incl. decoding (first encoder run fills the sample cache, later runs read it), one run, not the section 7 cost record")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-jobs", type=int, default=None)
    ap.add_argument("--out-root", default=str(OUT), help="use another folder for a test run")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    enc = encoders.load_audio(a.cand)
    for job in JOBS[:a.limit_jobs]:
        run_job(enc, *job, a.workers, a.batch, Path(a.out_root), None if a.no_cache else CACHE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
