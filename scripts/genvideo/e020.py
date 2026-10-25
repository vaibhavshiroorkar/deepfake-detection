"""E-020: development protocol and held-out evaluation of the generated-video detector. Protocol: docs/EXPERIMENTS.md E-020.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/genvideo/e020.py dev        # DeepAction only: LOGO selection, freeze heads
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/genvideo/e020.py df26 --final   # one-time evaluation of the frozen heads on DF26

``dev`` never reads DF26. ``df26`` refuses without --final and a second time (ledger); it changes no head, threshold or choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from dfd.eval.bootstrap import cluster_bootstrap
from dfd.eval.ledger import append_entries, read_ledger
from dfd.eval.metrics import roc_auc
from dfd.platform.genvideo.model import load_head, save_head

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "_results" / "genvideo"
OUT = FEAT
CKPT = ROOT / "checkpoints" / "genvideo-E020"
LEDGER = ROOT / "ledger" / "final_ledger.jsonl"
SEEDS = (0, 1, 2)
N_BOOT = 2000
CANDIDATES = {"C1": "restrav21", "C2": "cls_mean", "C3": "stats"}      # cost order for ties: C3 < C2 < C1
FOLDS = 5


def fit(cand: str, X, y, seed: int):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    if cand == "C1":
        reps = int(round((y == 1).sum() / max((y == 0).sum(), 1)))
        idx = np.concatenate([np.arange(len(y))] + [np.where(y == 0)[0]] * max(reps - 1, 0))
        m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, random_state=seed).fit(Z[idx], y[idx])
        return mu, sd, list(m.coefs_), list(m.intercepts_)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Z, y)
    return mu, sd, [m.coef_.T], [m.intercept_]


def predict(model, X):
    mu, sd, W, b = model
    h = (X - mu) / sd
    for i in range(len(W) - 1):
        h = np.maximum(h @ W[i] + b[i], 0.0)
    return 1.0 / (1.0 + np.exp(-(h @ W[-1] + b[-1]).ravel()))


def folds_of(groups, seed=0):
    u = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    f = {g: i % FOLDS for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])


def ci(a):
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_dev(args) -> int:
    z = np.load(FEAT / "features_deepaction.npz", allow_pickle=False)
    y, gen, grp = z["label"].astype(int), z["generator"], z["group"]
    gens = sorted(set(gen[y == 1]))
    names = list(CANDIDATES)
    fold = folds_of(grp)
    n = len(y)
    # S[c][s][gi, i]: out-of-fold score of clip i in the leave-one-generator-out run that holds generator gi out. Generator clips of gi and the reals
    # of the held-out prompt fold are scored by a model that saw neither that generator nor those prompt groups.
    S = {c: {s: np.full((len(gens), n), np.nan) for s in (SEEDS if c == "C1" else (0,))} for c in names}
    for c in names:
        X = z[CANDIDATES[c]].astype(np.float64)
        for s in S[c]:
            for gi, g in enumerate(gens):
                for f in range(FOLDS):
                    tr = (gen != g) & (fold != f)
                    te = ((gen == g) | (y == 0)) & (fold == f)
                    if te.any():
                        S[c][s][gi, te] = predict(fit(c, X[tr], y[tr], s), X[te])
    tab = {c: np.mean(list(S[c].values()), axis=0) for c in names}

    def macro(t, ii):
        a = []
        for gi, g in enumerate(gens):
            m = (gen[ii] == g) | (y[ii] == 0)
            a.append(roc_auc(y[ii][m], t[gi][ii][m]))
        return float(np.mean(a))

    allidx = np.arange(n)
    ug = np.array(sorted(set(grp)))
    byg = {g: np.where(grp == g)[0] for g in ug}
    rng = np.random.default_rng(20261026)
    point = {c: macro(tab[c], allidx) for c in names}
    reps = {c: [] for c in names}
    for _ in range(N_BOOT):
        ii = np.concatenate([byg[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        for c in names:
            reps[c].append(macro(tab[c], ii))
    reps = {c: np.array(v) for c, v in reps.items()}
    res = {"experiment": "E-020", "date": time.strftime("%Y-%m-%d"), "features": "features_deepaction.npz", "n_clips": int(n),
           "n_real": int((y == 0).sum()), "generators": gens, "candidates": {}}
    for c in names:
        per_seed = [macro(S[c][s], allidx) for s in S[c]]
        per_gen = {g: round(float(roc_auc(y[(gen == g) | (y == 0)], tab[c][gi][(gen == g) | (y == 0)])), 4) for gi, g in enumerate(gens)}
        res["candidates"][c] = {"feature": CANDIDATES[c], "macro_logo_auc": round(point[c], 4), "ci95": ci(reps[c]),
                                "seed_values": [round(v, 4) for v in per_seed],
                                "seed_sd": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else None, "per_generator_auc": per_gen}
    diffs = {f"{a}-{b}": {"estimate": round(point[a] - point[b], 4), "ci95": ci(reps[a] - reps[b])} for a, b in (("C1", "C3"), ("C1", "C2"), ("C2", "C3"))}
    res["paired_differences"] = diffs
    eligible = [c for c in ("C1", "C2") if res["candidates"][c]["ci95"][0] > 0.70 and diffs[f"{c}-C3"]["ci95"][0] > 0]
    chosen = None
    if eligible:
        chosen = max(eligible, key=lambda c: point[c])
        if chosen == "C1" and "C2" in eligible and diffs["C1-C2"]["ci95"][0] <= 0:
            chosen = "C2"
    res["selection"] = {"eligible": eligible, "chosen": chosen,
                        "rule": "eligible = macro LOGO AUC lower bound > 0.70 and paired difference to C3 lower bound > 0 (C3 is the pixel-statistics shortcut "
                                "control and is never deployed); best point estimate; if C1 is best and its paired difference to C2 has lower bound <= 0, the "
                                "simpler C2 is chosen"}
    if chosen:
        sc, yy = [], []
        for gi, g in enumerate(gens):
            m = (gen == g) | (y == 0)
            sc.append(tab[chosen][gi][m]); yy.append(y[m])
        sc, yy = np.concatenate(sc), np.concatenate(yy)
        w = np.where(yy == 1, 0.5 / (yy == 1).sum(), 0.5 / (yy == 0).sum())
        ths = np.unique(sc)
        ba = [float(((sc >= t) & (yy == 1)).astype(float).dot(w) + ((sc < t) & (yy == 0)).astype(float).dot(w)) for t in ths]
        res["selection"]["threshold"] = float(ths[int(np.argmax(ba))])
        res["selection"]["threshold_balanced_accuracy_oof"] = round(max(ba), 4)
    CKPT.mkdir(parents=True, exist_ok=True)
    meta_common = {"experiment": "E-020", "protocol": "canonical-v1", "trained_on": "DeepAction canonical clips, all 2,605 (100 real Pexels, 2,505 generated)",
                   "dino_revision": str(z["dino_revision"][0])}
    for c in names:
        model = fit(c, z[CANDIDATES[c]].astype(np.float64), y, 0)
        thr_c = res["selection"].get("threshold", 0.5) if c == chosen else 0.5
        save_head(CKPT / f"head_{c}.npz", model[0], model[1], model[2], model[3], thr_c,
                  {**meta_common, "head_id": f"E020-{c}-seed0", "feature": CANDIDATES[c], "candidate": c, "evaluation": "not evaluated"})
    res["head_files"] = {c: sha256(CKPT / f"head_{c}.npz") for c in names}
    (OUT / "e020_dev_results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("candidates", "paired_differences", "selection")}, indent=1))
    return 0


def cmd_df26(args) -> int:
    if any(e.get("iteration") == "E-020" for e in read_ledger(LEDGER)):
        print("E-020 is already in the ledger: refused", file=sys.stderr)
        return 2
    dev = json.loads((OUT / "e020_dev_results.json").read_text())
    z = np.load(FEAT / "features_df26.npz", allow_pickle=False)
    y, gen, grp, acc = z["label"].astype(int), z["generator"], z["group"], z["access"]
    out = {"experiment": "E-020", "dataset": "df26", "n": int(len(y)), "n_real": int((y == 0).sum()), "chosen_on_development": dev["selection"]["chosen"],
           "head_hashes_at_dev": dev["head_files"], "heads": {}}
    for c in CANDIDATES:
        head = load_head(CKPT / f"head_{c}.npz")
        assert sha256(CKPT / f"head_{c}.npz") == dev["head_files"][c], "head changed since development"
        thr = dev["selection"].get("threshold", 0.5) if c == dev["selection"]["chosen"] else None
        s = head.score(z[CANDIDATES[c]].astype(np.float64))
        r = {}
        for name, mask in (("all", np.ones(len(y), bool)), ("closed", (acc == "closed") | (y == 0)), ("open", (acc == "open") | (y == 0))):
            b = cluster_bootstrap(y[mask], s[mask], grp[mask], metric="auc", n_boot=N_BOOT)
            r[name] = {"auc": round(b.estimate, 4), "ci95": [round(b.low, 4), round(b.high, 4)], "n": int(mask.sum()), "n_fake": int(y[mask].sum())}
        r["per_generator"] = {}
        for g in sorted(set(gen[y == 1])):
            m = (gen == g) | (y == 0)
            b = cluster_bootstrap(y[m], s[m], grp[m], metric="auc", n_boot=N_BOOT)
            r["per_generator"][g] = {"auc": round(b.estimate, 4), "ci95": [round(b.low, 4), round(b.high, 4)], "n_fake": int((gen == g).sum())}
        if thr is not None:
            r["threshold"] = thr
            r["sensitivity"] = round(float((s[y == 1] >= thr).mean()), 4)
            r["specificity"] = round(float((s[y == 0] < thr).mean()), 4)
        out["heads"][c] = r
    ch = dev["selection"]["chosen"]
    if ch:
        lo, hi = out["heads"][ch]["closed"]["ci95"]
        out["verdict_H_G1"] = "supported" if lo > 0.70 else ("not supported" if hi < 0.60 else "inconclusive")
    (OUT / "e020_df26_results.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    append_entries([{"date": time.strftime("%Y-%m-%d"), "iteration": "E-020", "dataset": "df26", "split": "test_x", "run_ids": list(dev["head_files"].values()),
                     "modalities": ["visual"], "score_file_sha256": sha256(FEAT / "features_df26.npz")}], LEDGER)
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dev", "df26"])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "df26":
        if not a.final:
            print("refused: DF26 evaluation needs --final", file=sys.stderr)
            return 2
        return cmd_df26(a)
    return cmd_dev(a)


if __name__ == "__main__":
    raise SystemExit(main())
