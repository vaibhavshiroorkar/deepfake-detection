"""Shared helpers of the freeze scoring harness (docs/_agents/FREEZE.md).

Datasets, record selection behind the final-only guard, feature and score caches, the frozen-feature and fine-tuned scorers, the head wrapper.
Nothing here reads a sealed split unless the caller passes final=True, and only score_final.py passes it (and only under --final).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dfd.data import loaders as L

FINAL_ROOT = Path("data/_final")
DRY_ROOT = Path("data/_final_dry")
SEALED = ("test_in", "test_x")
SEEDS = (0, 1, 2)

VIDEO_SETS = [
    dict(key="ffpp", manifest="data/_manifests/ffpp.manifest.csv", split_csv="data/_splits/ffpp-037e2976fade.csv", crops="data/_crops/ffpp_c23_yunet_n16",
         splits=("val", "test_in")),
    dict(key="celebdf_v2", manifest="data/_manifests/celebdf_v2.manifest.csv", split_csv="data/_splits/celebdf_v2-82a3a70594e9.csv",
         crops="data/_crops/celebdf_v2_yunet_n16", splits=("dev_x", "test_x")),
    dict(key="dfd", manifest="data/_manifests/dfd.manifest.csv", split_csv="data/_splits/dfd-f5ff5576814a.csv", crops="data/_crops/dfd_yunet_n16",
         splits=("test_x",)),
    dict(key="celebdf_pp", manifest="data/_manifests/celebdf_pp.manifest.csv", split_csv="data/_splits/celebdf_pp-3f9df077b9eb.csv",
         crops="data/_crops/celebdf_pp_yunet_n16", splits=("test_x",)),
]
IMAGE_SETS = [
    dict(key="forensynths_progan_val", manifest="data/_manifests/forensynths_progan_val.manifest.csv", split_csv="data/_splits/progan_val-b42dc9619141.csv",
         splits=("val",)),
    dict(key="forensynths_progan_test", manifest="data/_manifests/forensynths_progan_test.manifest.csv", split_csv="data/_splits/progan_test-7720c87f1b53.csv",
         splits=("test_in",)),
    dict(key="forensynths_cnn_subset", manifest="data/_manifests/forensynths_cnn_subset.manifest.csv", split_csv="data/_splits/cnn_subset-4e973bc4da0f.csv",
         splits=("dev_x", "test_x")),
    dict(key="diffusion_test", manifest="data/_manifests/diffusion_test.manifest.csv", split_csv="data/_splits/diffusion_test-67dd057f5042.csv",
         splits=("test_x",)),
]
AUDIO_SETS = [
    dict(key="asvspoof2019_la", manifest="data/_manifests/asvspoof2019_la.manifest.csv", split_csv="data/_splits/asvspoof2019_la-875bf337c379.csv",
         splits=("val", "test_in")),
    dict(key="in_the_wild", manifest="data/_manifests/in_the_wild.manifest.csv", split_csv="data/_splits/in_the_wild-5cf2ea54a724.csv",
         splits=("dev_x", "test_x")),
]
LAVDF = dict(key="lavdf", manifest="data/_manifests/lavdf_subset.manifest.csv", split_csv="data/_splits/lavdf_subset-e9d12decaf1f.csv",
             crops="data/_crops/lavdf_subset_yunet_n16")
# feature files of the sweep (E-001 to E-003, E-007) that already hold the non-sealed splits: (dataset key, split) -> file stem
LEGACY = {("ffpp", "val"): "ffpp_val", ("celebdf_v2", "dev_x"): "celebdf_dev_x", ("forensynths_progan_val", "val"): "progan_val",
          ("forensynths_cnn_subset", "dev_x"): "cnn_dev_x", ("asvspoof2019_la", "val"): "asvspoof_dev", ("in_the_wild", "dev_x"): "itw_dev_x"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def select(ds: dict, split: str, final: bool, modality: str | None, limit: int | None = None):
    """Records of a split. A sealed split needs final=True (the loader refuses otherwise); `limit` takes an evenly strided subset (smoke tests)."""
    if split in SEALED and not final:
        raise RuntimeError(f"{ds['key']}/{split} is sealed: this call needs --final")
    recs = L.select_records(ds["manifest"], ds["split_csv"], split, modality=modality, final=final)
    if limit and len(recs) > limit:
        recs = [recs[i] for i in np.unique(np.linspace(0, len(recs) - 1, limit).astype(int))]
    return recs


def splits_of(ds: dict, dry: bool):
    return [s for s in ds["splits"] if not (dry and s in SEALED)]


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


class NpHead:
    """Linear head from the arrays mean, std, w, b (dfd.sweep.heads / dfd.infer.checkpoint format)."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std, self.w, self.b = z["mean"].astype(np.float32), z["std"].astype(np.float32), z["w"].astype(np.float32), float(z["b"])
        self.path = str(path)

    def logit_rows(self, X):
        return ((np.asarray(X, np.float32) - self.mean) / self.std) @ self.w + self.b

    def prob_rows(self, X):
        return 1.0 / (1.0 + np.exp(-self.logit_rows(X)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def clip_mean(p_rows, clip_of, n):
    s = np.bincount(clip_of, weights=p_rows, minlength=n)
    c = np.bincount(clip_of, minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


@dataclass
class Block:
    """Scores of one run on one (dataset, split): clip or image level, raw probabilities, NaN where no score exists."""
    modality: str
    dataset: str
    split: str
    seed: int
    sample_id: np.ndarray
    group_id: np.ndarray
    label: np.ndarray
    p: np.ndarray
    listed: int = 0           # records of the split before any were dropped (missing crops, no face)


# ------------------------------------------------------------------ feature access (frozen encoders)
def _align(z: dict, sample_ids):
    pos = {str(s): i for i, s in enumerate(z["sample_id"])}
    idx = np.array([pos[s] for s in sample_ids if s in pos], dtype=int)
    return idx


_ENC = {}


def get_encoder(kind: str, cid: str):
    """One encoder in memory at a time (one GPU job)."""
    key = (kind, cid)
    if key in _ENC:
        return _ENC[key]
    release()
    from dfd.sweep import encoders
    _ENC[key] = encoders.load(cid) if kind == "image" else encoders.load_audio(cid)
    return _ENC[key]


def release():
    import gc
    _ENC.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def video_features(cand: str, ds: dict, split: str, final: bool, limit, root: Path, workers: int = 8):
    """Frozen-encoder features of every frame of every clip: dict feats [N,T,D] float16, mask [N,T], label, group_id, sample_id, n_listed."""
    recs = select(ds, split, final, "video", limit)
    legacy = LEGACY.get((ds["key"], split))
    lf = Path("data/_features") / cand / "T-V" / f"{legacy}.npz" if legacy else None
    if lf is not None and lf.is_file():
        z = dict(np.load(lf))
        idx = _align(z, [r.sample_id for r in recs])
        return {k: z[k][idx] for k in ("feats", "mask", "label", "group_id", "sample_id")} | {"n_listed": len(recs)}
    out = root / "features" / cand / "video" / f"{ds['key']}_{split}.npz"
    if out.is_file():
        z = dict(np.load(out))
        return {k: z[k] for k in ("feats", "mask", "label", "group_id", "sample_id")} | {"n_listed": int(z["n_listed"])}
    import torch
    from torch.utils.data import DataLoader
    from dfd.prep.pipeline import folder_name
    enc = get_encoder("image", cand)
    have = [r for r in recs if (Path(ds["crops"]) / folder_name(r.sample_id) / "_done.json").is_file()]
    dset = L.FaceClipDataset(have, ds["crops"], data_root="data")
    dl = DataLoader(dset, batch_size=16, num_workers=workers, shuffle=False)
    N, T = len(dset), dset.n_frames
    feats = np.zeros((N, T, enc.dim), np.float16)
    mask = np.zeros((N, T), bool)
    lab, grp, sid, i = [], [], [], 0
    for b in dl:
        x, m = b["x"], b["mask"]
        B = x.shape[0]
        flat = x[m]
        if flat.shape[0]:
            f = torch.cat([enc(flat[k:k + 256].cuda(non_blocking=True)) for k in range(0, flat.shape[0], 256)]).cpu()
            tmp = torch.zeros(B, T, enc.dim)
            tmp[m] = f
            feats[i:i + B] = tmp.numpy().astype(np.float16)
        mask[i:i + B] = m.numpy()
        lab += [int(v) for v in b["label"]]
        grp += list(b["group_id"])
        sid += list(b["sample_id"])
        i += B
    save_npz(out, feats=feats, mask=mask, label=np.array(lab), group_id=np.array(grp), sample_id=np.array(sid), n_listed=len(recs))
    return dict(feats=feats, mask=mask, label=np.array(lab), group_id=np.array(grp), sample_id=np.array(sid), n_listed=len(recs))


def image_features(cand: str, ds: dict, split: str, final: bool, limit, root: Path, workers: int = 8):
    recs = select(ds, split, final, "image", limit)
    legacy = LEGACY.get((ds["key"], split))
    lf = Path("data/_features") / cand / "T-I" / f"{legacy}.npz" if legacy else None
    if lf is not None and lf.is_file():
        z = dict(np.load(lf))
        idx = _align(z, [r.sample_id for r in recs])
        return {k: z[k][idx] for k in ("feats", "label", "group_id", "sample_id")} | {"n_listed": len(recs)}
    out = root / "features" / cand / "image" / f"{ds['key']}_{split}.npz"
    if out.is_file():
        z = dict(np.load(out))
        return {k: z[k] for k in ("feats", "label", "group_id", "sample_id")} | {"n_listed": int(z["n_listed"])}
    from torch.utils.data import DataLoader
    from dfd.sweep.extract_images import ImgSet
    enc = get_encoder("image", cand)
    dl = DataLoader(ImgSet(recs, enc.processor), batch_size=64, num_workers=workers, shuffle=False)
    feats = np.zeros((len(recs), enc.dim), np.float16)
    for x, idx in dl:
        feats[idx.numpy()] = enc.encode_pixels(x.cuda(non_blocking=True)).cpu().numpy().astype(np.float16)
    d = dict(feats=feats, label=np.array([r.label for r in recs]), group_id=np.array([r.group_id for r in recs]), sample_id=np.array([r.sample_id for r in recs]))
    save_npz(out, **d, n_listed=len(recs))
    return d | {"n_listed": len(recs)}


def audio_features(cand: str, variant: str, ds: dict, split: str, final: bool, limit, root: Path, workers: int = 8, modality: str = "audio", tag: str | None = None):
    """Mean and std over time of the last hidden layer for the 4 s window ("first": E-003 window, "trim_norm": E-007 window)."""
    recs = select(ds, split, final, modality, limit)
    legacy = LEGACY.get((ds["key"], split))
    base = {"first": "data/_features", "trim_norm": "data/_features_trimnorm"}[variant]
    lf = Path(base) / cand / "T-A" / f"{legacy}.npz" if legacy else None
    if lf is not None and lf.is_file():
        z = dict(np.load(lf))
        idx = _align(z, [r.sample_id for r in recs])
        return {k: z[k][idx] for k in ("feats", "label", "group_id", "sample_id")} | {"n_listed": len(recs)}
    out = root / "features" / cand / f"audio_{variant}{('_' + tag) if tag else ''}" / f"{ds['key']}_{split}.npz"
    if out.is_file():
        z = dict(np.load(out))
        return {k: z[k] for k in ("feats", "label", "group_id", "sample_id")} | {"n_listed": int(z["n_listed"])}
    from torch.utils.data import DataLoader
    from dfd.sweep import extract_audio
    enc = get_encoder("audio", cand)
    cache = extract_audio.CACHE if variant == "first" else extract_audio.CACHE_TRIMNORM
    if modality == "video":
        cache = None      # LAV-DF audio tracks are decoded from the video each time (train_audio_lavdf.py does the same)
    dl = DataLoader(extract_audio.AudioSet(recs, cache=cache, variant=variant), batch_size=32, num_workers=workers, shuffle=False)
    X = np.zeros((len(recs), 2 * enc.dim), np.float16)
    for x, idx in dl:
        X[idx.numpy()] = enc(x.numpy()).cpu().numpy().astype(np.float16)
    d = dict(feats=X, label=np.array([r.label for r in recs]), group_id=np.array([r.group_id for r in recs]), sample_id=np.array([r.sample_id for r in recs]))
    save_npz(out, **d, n_listed=len(recs))
    return d | {"n_listed": len(recs)}


def frozen_video_p(feats, mask, head: NpHead):
    N, T, _ = feats.shape
    clip_of = np.repeat(np.arange(N), T).reshape(N, T)[mask]
    return clip_mean(head.prob_rows(feats[mask]), clip_of, N)


# ------------------------------------------------------------------ fine-tuned image encoders (E-005, E-011, E-012, E-013)
def frames_of(recs, crops):
    """(paths, clip_of, index of the records that have finished crops)."""
    from dfd.prep.pipeline import folder_name
    paths, clip_of, have = [], [], []
    for i, r in enumerate(recs):
        d = Path(crops) / folder_name(r.sample_id)
        if not (d / "_done.json").is_file():
            continue
        have.append(i)
        with open(d / "index.csv", "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["detected"] == "1" and row["crop_path"] and int(row["frame_index"]) < 16:
                    paths.append(str(Path(crops) / row["crop_path"]))
                    clip_of.append(i)
    return np.array(paths), np.array(clip_of, dtype=int), np.array(have, dtype=int)


def load_finetuned(cid: str, weights: str):
    """Encoder with the fine-tuned parameters (and BatchNorm statistics when the file has them) and the linear head."""
    import torch
    release()
    from dfd.sweep import encoders
    enc = encoders.load(cid)
    w = torch.load(weights, map_location="cpu")
    names = dict(enc.model.named_parameters())
    with torch.no_grad():
        for n, v in w["params"].items():
            names[n].copy_(v.to(names[n].device, names[n].dtype))
        if "buffers" in w:
            bufs = dict(enc.model.named_buffers())
            for n, v in w["buffers"].items():
                bufs[n].copy_(v.to(bufs[n].device, bufs[n].dtype))
        elif cid == "IMG-04":
            raise RuntimeError(f"{weights} has no BatchNorm statistics: the Xception baseline cannot be scored from it (src/dfd/train/finetune.py saved only parameters). "
                               "Rerun the baseline run with the current finetune.py.")
    head = torch.nn.Linear(enc.dim, 1).cuda()
    head.load_state_dict(w["head"])
    enc.model.eval()
    return enc, head


def ft_scores(model_key: str, cid: str, weights: str, jobs, root: Path, workers: int = 10):
    """Clip probabilities (mean of the frame probabilities, D-024) of a fine-tuned encoder for [(ds, split, recs)]; cached per (model, dataset, split).

    Returns {(ds_key, split): dict(sample_id, label, group_id, p, n_listed)}; p is NaN for a clip without a finished crop or without a face frame.
    """
    from dfd.train import finetune as FT
    out, todo = {}, []
    for ds, split, recs in jobs:
        f = root / "ftcache" / model_key / f"{ds['key']}_{split}.npz"
        if f.is_file():
            z = dict(np.load(f))
            out[(ds["key"], split)] = {k: z[k] for k in ("sample_id", "label", "group_id", "p")} | {"n_listed": int(z["n_listed"])}
        else:
            todo.append((ds, split, recs, f))
    if todo:
        enc, head = load_finetuned(cid, weights)
        for ds, split, recs, f in todo:
            paths, clip_of, have = frames_of(recs, ds["crops"])
            p = np.full(len(recs), np.nan)
            if len(paths):
                fr = FT.score_frames(enc, head, FT.Frames(paths, clip_of, np.zeros(len(recs))), workers)
                p = FT.clip_mean(fr, clip_of, len(recs))
            d = dict(sample_id=np.array([r.sample_id for r in recs]), label=np.array([r.label for r in recs]), group_id=np.array([r.group_id for r in recs]), p=p)
            save_npz(f, **d, n_listed=len(recs))
            out[(ds["key"], split)] = d | {"n_listed": len(recs)}
            print(f"  {model_key} {ds['key']}/{split}: {int(np.isfinite(p).sum())} of {len(recs)} clips scored", flush=True)
        release()
    return out
