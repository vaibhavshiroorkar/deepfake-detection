"""Retrain and save the linear heads of the staircase iterations that have no stored head files (CPU only, deterministic).

    PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/Scripts/python.exe scripts/freeze/prepare_heads.py

I-0 (T-V IMG-07, T-I IMG-13, T-A AU-8 on the first 4 s), I-2 (AU-8 on the E-007 window) and I-3 (AU-8 with the three augmented copies, E-008) were
trained inside scripts that did not keep the head parameters. The recipe is the one logged (dfd.sweep.heads, seeds 0, 1, 2, nothing else changes) and
the result is checked against what was logged: I-0 against data/_results/I-0/<task>/scores.npz (validation and dev_x clip scores, element by
element), I-2 and I-3 against the AUC estimates in data/_results/I-2/ha1.json and I-3/ha2.json. The differences go to data/_final/heads/verification.json;
a difference above the tolerance stops the script (a head that does not reproduce the logged one is not the frozen head).
Heads of I-4 (data/_results/I-4/head_*.npz) and of the LAV-DF iterations (data/_results/I-6/audio_head_*.npz) already exist and are not touched.
Only train, validation and dev_x features are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfd.eval import metrics as M
from dfd.sweep import heads, rank, tasks

OUT = Path("data/_final/heads")
TOL_SCORES = 1e-4
TOL_AUC = 5e-4


def save(path, h):
    np.savez(path, mean=h.mean, std=h.std, w=h.w, b=np.float64(h.b), best_epoch=h.best_epoch, val_auc=h.val_auc)


def clip_p(h, sp):
    return heads.clip_scores(h.prob_rows(sp.X), sp.clip_of, sp.y.size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ver = {}
    # ---- I-0
    tasks.FEATURES = Path("data/_features")
    for tid, cand in (("T-V", "IMG-07"), ("T-I", "IMG-13"), ("T-A", "AU-8")):
        t = tasks.TASKS[tid]
        tr, _ = rank.load_split(cand, t, t.train)
        va, _ = rank.load_split(cand, t, t.val)
        dv, _ = rank.load_split(cand, t, t.devx)
        hs = [heads.train_head(tr, va, s) for s in (0, 1, 2)]
        for s, h in enumerate(hs):
            save(OUT / f"I-0_{tid}_s{s}.npz", h)
        ref = np.load(Path("data/_results/I-0") / tid / "scores.npz", allow_pickle=True)
        assert np.array_equal(ref["val_sample_id"], va.sample_id) and np.array_equal(ref["dev_sample_id"], dv.sample_id)
        dv_ = max(float(np.nanmax(np.abs(np.array([clip_p(h, sp) for h in hs]) - ref[f"{k}_scores"]))) for k, sp in (("val", va), ("dev", dv)))
        ver[f"I-0_{tid}"] = dict(max_abs_score_difference=dv_, tolerance=TOL_SCORES, ok=bool(dv_ <= TOL_SCORES), candidate=cand)
        print("I-0", tid, cand, "max abs score difference to the logged scores", dv_, flush=True)
    # ---- I-2 and I-3 (AU-8, E-007 window)
    tasks.FEATURES = Path("data/_features_trimnorm")
    t = tasks.TASKS["T-A"]
    tr, _ = rank.load_split("AU-8", t, t.train)
    va, _ = rank.load_split("AU-8", t, t.val)
    dv, _ = rank.load_split("AU-8", t, t.devx)
    n = tr.y.size
    aug = [np.load(f"data/_features_aug/AU-8/T-A/asvspoof_train_aug{k}.npz")["feats"].astype(np.float32) for k in (1, 2, 3)]
    tr_aug = heads.Split(np.concatenate([tr.X] + aug), np.concatenate([tr.clip_of] + [np.arange(n)] * 3), tr.y, tr.groups, tr.sample_id)
    for tag, split_tr, logged in (("I-2", tr, "data/_results/I-2/ha1.json"), ("I-3", tr_aug, "data/_results/I-3/ha2.json")):
        hs = [heads.train_head(split_tr, va, s) for s in (0, 1, 2)]
        for s, h in enumerate(hs):
            save(OUT / f"{tag}_T-A_s{s}.npz", h)
        j = json.loads(Path(logged).read_text())
        key = "new"
        got = {name: float(np.mean([M.roc_auc(sp.y, clip_p(h, sp)) for h in hs])) for name, sp in (("validation", va), ("dev_x", dv))}
        want = {name: j[name][key]["estimate"] for name in got}
        d = max(abs(got[k] - want[k]) for k in got)
        ver[f"{tag}_T-A"] = dict(got=got, logged=want, max_abs_difference=d, tolerance=TOL_AUC, ok=bool(d <= TOL_AUC))
        print(tag, "AU-8", got, "logged", want, flush=True)
    (OUT / "verification.json").write_text(json.dumps(ver, indent=1))
    bad = [k for k, v in ver.items() if not v["ok"]]
    if bad:
        raise SystemExit(f"heads that do not reproduce the logged result: {bad}")
    print("all heads reproduce the logged results; written to", OUT)


if __name__ == "__main__":
    main()
