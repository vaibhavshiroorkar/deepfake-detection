"""E-011, audio part: linear heads on AU-8, AU-4, AU-6 features of the LAV-DF audio track (label modify_audio), E-007 window, head recipe of
dfd.sweep.heads (seeds 0, 1, 2). Early stopping on the AUC against modify_audio on LAV-DF `val` (as every head of this project stops on its
validation split). Scores the fusion-fit subset `train` and `val` (and `test_in` only with --final).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/train_audio_lavdf.py [--final]

Output: data/_features_lavdf/<cand>/<name>.npz (features), data/_results/I-6/audio_lavdf.npz (raw clip probabilities per member and seed), heads
data/_results/I-6/audio_head_<cand>_s<seed>.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfd.data import loaders as L
from dfd.sweep import encoders, extract_audio, heads

HT = ("data/_manifests/lavdf_headtrain.manifest.csv", "data/_splits/lavdf_headtrain-b5544b99f6bf.csv", "train")
SUB = ("data/_manifests/lavdf_subset.manifest.csv", "data/_splits/lavdf_subset-e9d12decaf1f.csv")
OUT = Path("data/_results/I-6")
FEAT = Path("data/_features_lavdf")
MEMBERS = ["AU-8", "AU-4", "AU-6"]


def get(split_name, cfg, final=False):
    manifest, split_csv, split = cfg[0], cfg[1], (cfg[2] if len(cfg) > 2 else split_name)
    return L.select_records(manifest, split_csv, split, modality="video", final=final)


def features(enc, cid, name, recs, workers):
    f = FEAT / cid / f"{name}.npz"
    if f.is_file():
        z = np.load(f)
        assert [str(s) for s in z["sample_id"]] == [r.sample_id for r in recs]
        return z["feats"].astype(np.float32)
    dl = DataLoader(extract_audio.AudioSet(recs, cache=None, variant="trim_norm"), batch_size=32, num_workers=workers, shuffle=False)
    X = np.zeros((len(recs), 2 * enc.dim), np.float16)
    for x, idx in dl:
        X[idx.numpy()] = enc(x.numpy()).cpu().numpy().astype(np.float16)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, feats=X, sample_id=np.array([r.sample_id for r in recs]))
    return X.astype(np.float32)


def labels(recs):
    return np.array([int(dict(kv.split("=") for kv in r.notes.split(";"))["modify_audio"]) for r in recs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"headtrain": get("train", HT), "train": get("train", SUB), "val": get("val", SUB)}
    if a.final:
        sets["test_in"] = get("test_in", SUB, True)
    res = {"members": np.array(MEMBERS), **{f"{k}_sample_id": np.array([r.sample_id for r in v]) for k, v in sets.items() if k != "headtrain"}}
    P = {k: np.zeros((3, 3, len(sets[k]))) for k in sets if k != "headtrain"}
    for m, cid in enumerate(MEMBERS):
        enc = encoders.load_audio(cid)
        X = {k: features(enc, cid, k, v, a.workers) for k, v in sets.items()}
        del enc
        torch.cuda.empty_cache()
        y = {k: labels(v) for k, v in sets.items()}
        g = {k: np.array([r.group_id for r in v]) for k, v in sets.items()}
        mk = lambda k: heads.Split(X[k], np.arange(len(y[k])), y[k], g[k], np.array([r.sample_id for r in sets[k]]))   # noqa: E731
        for s in (0, 1, 2):
            h = heads.train_head(mk("headtrain"), mk("val"), s)
            np.savez(OUT / f"audio_head_{cid}_s{s}.npz", mean=h.mean, std=h.std, w=h.w, b=np.float64(h.b), best_epoch=h.best_epoch, val_auc=h.val_auc)
            for k in P:
                P[k][m, s] = h.prob_rows(X[k])
            print(cid, "seed", s, "best epoch", h.best_epoch, "val auc (stopping split)", round(h.val_auc, 4), flush=True)
    for k in P:
        res[f"{k}_p"] = P[k]
    np.savez(OUT / "audio_lavdf.npz", **res)


if __name__ == "__main__":
    main()
