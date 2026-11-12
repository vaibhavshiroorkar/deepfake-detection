"""Measurement M-004 (diagnosis only, CPU, no selection change; D-049 follow-up).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/m004.py

(a) The from-scratch T-A reference on ASVspoof 2019 LA dev (in-domain) from data/_results/T-A/reference.json, and the label mapping of
    the In-the-Wild dev_x features and the ASVspoof train and dev features against meta.csv and the cm protocol files.
(b) AUC of trivial features of the audio (leading silence, clip duration, RMS energy, repeat-padded fraction of the 4 s window) on
    ASVspoof 2019 LA train, dev and on In-the-Wild dev_x. AUC of the raw feature as the fake-score: 0.5 carries no information, below 0.5
    means the fake class has the smaller value. Group bootstrap (speaker), 2,000 resamples.
(c) Per-generator AUC of IMG-13 and IMG-15 on the two T-I dev_x units. Each unit is one group, so the interval resamples images
    inside the unit (not groups) and says nothing about other generators.
Only train, validation and dev_x data are read. Output: data/_results/M-004/m004.json and m004.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dfd.data import loaders as L
from dfd.eval import bootstrap as B
from dfd.eval.metrics import roc_auc
from dfd.sweep import extract_audio

WINDOW, SR, THR = 64600, 16000, 0.005   # leading silence: first sample whose |x| exceeds 0.005 of full scale (about -46 dB)
OUT = Path("data/_results/M-004")
CACHE = extract_audio.CACHE


def audio_features(recs):
    import hashlib
    rows = []
    for r in recs:
        f = CACHE / (hashlib.sha1(r.sample_id.encode()).hexdigest() + ".npy")
        w = np.load(f).astype(np.float32) / 32767.0
        loud = np.nonzero(np.abs(w) > THR)[0]
        lead = (loud[0] if loud.size else w.size) / SR
        info = sf.info(str(Path("data") / r.path))
        n_orig = info.frames * SR / info.samplerate
        rows.append((lead, info.frames / info.samplerate, float(np.sqrt(np.mean(w ** 2))), max(0.0, 1.0 - n_orig / WINDOW)))
    return np.array(rows)


def auc_ci(y, s, g):
    b = B.cluster_bootstrap(y, np.asarray(s, float)[None], g, "auc", n_boot=B.N_BOOT)
    return dict(auc=float(b.estimate), low=float(b.low), high=float(b.high))


def image_ci(y, s, n_boot=2000, seed=20261024):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s, float)
    est = roc_auc(y, s)
    reps = []
    i0, i1 = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i0, i0.size), rng.choice(i1, i1.size)])
        reps.append(roc_auc(y[idx], s[idx]))
    return dict(auc=float(est), low=float(np.quantile(reps, 0.025)), high=float(np.quantile(reps, 0.975)))


def main():
    res = {}
    # ---------------------------------------------------------------- (a)
    ref = json.loads(Path("data/_results/T-A/reference.json").read_text())
    res["a_reference"] = [dict(seed=p["seed"], asvspoof_dev_auc=p["val_auc"], itw_dev_x_auc=p["devx_auc"]) for p in ref["per_seed"]]
    meta = {r["file"]: r for r in csv.DictReader(open("data/in_the_wild/release_in_the_wild/meta.csv", encoding="utf-8"))}
    z = np.load("data/_features/AU-8/T-A/itw_dev_x.npz", allow_pickle=False)
    bad = sum((0 if meta[str(s).split(":", 1)[1]]["label"] == "bona-fide" else 1) != int(l) for s, l in zip(z["sample_id"], z["label"]))
    res["a_labels"] = dict(itw_dev_x_items=int(len(z["label"])), itw_label_mismatches=int(bad))
    P = "data/asvspoof2019_la/LA/ASVspoof2019_LA_cm_protocols/"
    for split, fn in (("asvspoof_train", "ASVspoof2019.LA.cm.train.trn.txt"), ("asvspoof_dev", "ASVspoof2019.LA.cm.dev.trl.txt")):
        proto = {l.split()[1]: l.split()[4] for l in open(P + fn)}
        zz = np.load(f"data/_features/AU-8/T-A/{split}.npz", allow_pickle=False)
        m = sum((0 if proto[str(s).split(":", 1)[1]] == "bonafide" else 1) != int(l) for s, l in zip(zz["sample_id"], zz["label"]))
        res["a_labels"][split] = dict(items=int(len(zz["label"])), mismatches=int(m))
    # ---------------------------------------------------------------- (b)
    names = ["leading_silence_s", "duration_s", "rms", "repeat_pad_fraction"]
    res["b_trivial"] = {}
    for name, (_, manifest, split_csv, split) in zip(("asvspoof_train", "asvspoof_dev", "itw_dev_x"), extract_audio.JOBS):
        recs = L.select_records(manifest, split_csv, split, modality="audio")
        X = audio_features(recs)
        y = np.array([r.label for r in recs])
        g = np.array([r.group_id for r in recs])
        res["b_trivial"][name] = dict(n=int(y.size), fake=int(y.sum()), groups=int(np.unique(g).size),
                                      features={n: auc_ci(y, X[:, k], g) for k, n in enumerate(names)},
                                      medians={n: dict(real=float(np.median(X[y == 0, k])), fake=float(np.median(X[y == 1, k]))) for k, n in enumerate(names)})
        print(name, json.dumps(res["b_trivial"][name]["features"]), flush=True)
    # ---------------------------------------------------------------- (c)
    z = np.load("data/_results/T-I/scores.npz", allow_pickle=False)
    man = {r["sample_id"]: r for r in csv.DictReader(open("data/_manifests/forensynths_cnn_subset.manifest.csv"))}
    res["c_per_generator"] = {}
    for c in ("IMG-13", "IMG-15"):
        s = z[f"dev_{c}"].mean(0)
        per = {}
        for gname in np.unique(z["group"]):
            k = z["group"] == gname
            per[str(gname).split(":", 1)[1]] = dict(n=int(k.sum()), real=int((z["y"][k] == 0).sum()), fake=int((z["y"][k] == 1).sum()), **image_ci(z["y"][k], s[k]))
        per["pooled_2_units"] = dict(n=int(len(s)), **image_ci(z["y"], s))
        res["c_per_generator"][c] = per
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m004.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["c_per_generator"], indent=1))
    print(json.dumps(res["a_reference"]), json.dumps(res["a_labels"]))


if __name__ == "__main__":
    main()
