"""Shared definitions of E-027 (docs/EXPERIMENTS.md, preregistered 2026-10-27): DF26 metadata, source groups, folds, feature views,
heads, macro-AUC bootstrap and Wilson intervals. The closed subset is never returned by `dev_rows`; only e027_closed.py asks for it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
DF26 = ROOT / "data" / "df26"
CANON = ROOT / "data" / "_canonical_v3" / "df26"
OUT = ROOT / "data" / "_results" / "modern" / "e027"
CKPT = ROOT / "checkpoints" / "modern-E027"
E020_FEAT = ROOT / "data" / "_results" / "genvideo" / "features_df26.npz"
E020_HEAD = ROOT / "checkpoints" / "genvideo-E020" / "head.npz"
E020_HEAD_SHA = "b67d01ac0001ec69f53ce659c88ab8b4415bc3fb810ce30041651de3bfa7fafb"
OPEN_GENS = ["HunyuanVideo_1.5_14b_i2v", "HunyuanVideo_1.5_14b_t2v", "LTX_2.3_distilled_i2v", "LTX_2.3_distilled_t2v",
             "Wan_2.2_14b_i2v", "Wan_2.2_14b_t2v"]
CLOSED_GENS = ["Grok_Imagine", "Kling_3.0", "Veo_3.1", "Wan_2.6"]
FAMILY = {g: g.split("_")[0] for g in OPEN_GENS}          # HunyuanVideo, LTX, Wan
FOLDS = 5
SEEDS = (0, 1, 2)
BOOT_SEED = 20261027
N_BOOT = 2000
ENCODERS = {"IMG-07": "DINOv2-small", "IMG-08": "DINOv2-base", "IMG-17": "SigLIP-base", "IMG-13": "CLIP-ViT-L/14"}
COST = ["IMG-07", "IMG-08", "IMG-17", "IMG-13"]              # vision parameters 22 M < 87 M < 93 M < 304 M
PICK = np.array([0, 2, 4, 6, 7, 9, 11, 13])
TC = re.compile(r"^[0-9]{3}(_[0-9]+)?$")


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_group(clip_id: str) -> str:
    return ("tc:" + clip_id[:3]) if TC.match(clip_id) else ("yt:" + clip_id[:11])


def videos() -> list[dict]:
    with open(DF26 / "metadata" / "videos.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    real_by_path = {r["video_path"]: r for r in rows if r["label"] == "real"}
    for r in rows:
        pr = real_by_path[r["paired_real_path"]]
        r["group"] = source_group(pr["clip_id"])
        r["y"] = 0 if r["label"] == "real" else 1
    return rows


def dev_rows() -> list[dict]:
    """Real clips and the six open generators only (development pool)."""
    return [r for r in videos() if r["y"] == 0 or (r["generator_access"] == "open" and r["generator"] in OPEN_GENS)]


def closed_rows_FINAL_ONLY() -> list[dict]:
    """Closed-generator clips. Called only by e027_closed.py after its guard."""
    return [r for r in videos() if r["y"] == 1 and r["generator_access"] == "closed"]


def fold_of_groups(groups, seed: int) -> dict:
    u = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return {g: i % FOLDS for i, g in enumerate(u)}


# ------------------------------------------------------------------ feature views from per-frame embeddings E (n, T, D)
def view(E: np.ndarray, name: str, ids=None) -> np.ndarray:
    E = E.astype(np.float64)
    if name == "P8":
        return E[:, PICK].mean(1)
    if name in ("P14", "P16"):
        return E.mean(1)
    if name in ("PT14", "PT16"):
        return np.concatenate([E.mean(1), E.std(1), np.abs(np.diff(E, axis=1)).mean(1)], 1)
    if name == "PT14s":
        out = []
        for k in range(len(E)):
            perm = np.random.default_rng(BOOT_SEED + k).permutation(E.shape[1])
            e = E[k, perm]
            out.append(np.concatenate([e.mean(0), e.std(0), np.abs(np.diff(e, axis=0)).mean(0)]))
        return np.stack(out)
    if name == "D14":
        d = np.diff(E, axis=1)
        return np.concatenate([np.abs(d).mean(1), d.std(1)], 1)
    raise ValueError(name)


# ------------------------------------------------------------------ heads
def fit(kind: str, X, y, seed: int):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    if kind == "MLP":
        reps = int(round((y == 1).sum() / max((y == 0).sum(), 1)))
        idx = np.concatenate([np.arange(len(y))] + [np.where(y == 0)[0]] * max(reps - 1, 0))
        m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, random_state=seed).fit(Z[idx], y[idx])
        return mu, sd, list(m.coefs_), list(m.intercepts_)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Z, y)
    return mu, sd, [m.coef_.T], [m.intercept_]


def predict(model, X):
    mu, sd, W, b = model
    h = (X - mu) / sd
    for i in range(len(W) - 1):
        h = np.maximum(h @ W[i] + b[i], 0.0)
    z = np.clip((h @ W[-1] + b[-1]).ravel(), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def units(protocol: str) -> dict:
    """held-out unit -> list of generators."""
    if protocol == "LOFO":
        fams = sorted(set(FAMILY.values()))
        return {f: [g for g in OPEN_GENS if FAMILY[g] == f] for f in fams}
    return {g: [g] for g in OPEN_GENS}


def oof_scores(X, y, gen, grp, protocol: str, seed: int, kind: str = "LR") -> dict:
    """Out-of-fold scores. Returns {unit: score vector (nan where not scored)}; reals are scored under every unit."""
    fmap = fold_of_groups(grp, seed)
    fold = np.array([fmap[g] for g in grp])
    out = {}
    for u, gl in units(protocol).items():
        s = np.full(len(y), np.nan)
        inu = np.isin(gen, gl)
        for f in range(FOLDS):
            tr = (~inu) & (fold != f)
            te = (inu | (y == 0)) & (fold == f)
            if te.any():
                s[te] = predict(fit(kind, X[tr], y[tr], seed), X[te])
        out[u] = s
    return out


def gen_unit(protocol: str) -> dict:
    return {g: u for u, gl in units(protocol).items() for g in gl}


# ------------------------------------------------------------------ bootstrap of the macro AUC (seed-averaged), paired across candidates
def weighted_auc(y, s, w):
    from dfd.eval.metrics import roc_auc
    return roc_auc(y, s, weights=w)


class MacroBoot:
    """Shared resamples over source groups of the development pool (reals + open generated clips)."""

    def __init__(self, y, gen, grp, n_boot=N_BOOT, seed=BOOT_SEED):
        from dfd.eval.bootstrap import ClusterResampler
        self.y, self.gen, self.grp = y, gen, grp
        rs = ClusterResampler(y, grp, seed)
        self.W = np.stack(list(rs.weights(n_boot)))          # (B, n)

    def per_gen_evals(self, S_by_seed: list[dict], protocol: str, gens=OPEN_GENS):
        """S_by_seed: list over seeds of {unit: scores}. Returns per-gen evaluator lists."""
        from dfd.eval.metrics import make_evaluator
        gu = gen_unit(protocol) if protocol in ("LOFO", "LOGO") else None
        ev = {}
        for g in gens:
            m = (self.gen == g) | (self.y == 0)
            idx = np.where(m)[0]
            fs = []
            for S in S_by_seed:
                sc = S[gu[g]] if gu else S
                fs.append((idx, make_evaluator("auc", self.y[idx], sc[idx])))
            ev[g] = fs
        return ev

    def macro(self, ev) -> dict:
        """Point (seed mean of per-seed macro), per-seed values, replicate vector, per-gen point."""
        gens = list(ev)
        n_seed = len(ev[gens[0]])
        ones = np.ones(len(self.y))
        per_seed = [np.mean([ev[g][j][1](ones[ev[g][j][0]]) for g in gens]) for j in range(n_seed)]
        per_gen = {g: float(np.mean([ev[g][j][1](ones[ev[g][j][0]]) for j in range(n_seed)])) for g in gens}
        reps = np.empty(len(self.W))
        reps_gen = {g: np.empty(len(self.W)) for g in gens}
        for b, w in enumerate(self.W):
            acc = 0.0
            for g in gens:
                v = np.mean([f(w[idx]) for idx, f in ev[g]])
                reps_gen[g][b] = v
                acc += v
            reps[b] = acc / len(gens)
        return {"point": float(np.mean(per_seed)), "per_seed": [float(v) for v in per_seed],
                "seed_sd": float(np.std(per_seed, ddof=1)) if n_seed > 1 else None, "reps": reps,
                "per_gen": per_gen, "reps_gen": reps_gen}


def ci(a):
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def ba_threshold(sc, yy):
    w = np.where(yy == 1, 0.5 / (yy == 1).sum(), 0.5 / (yy == 0).sum())
    ths = np.unique(sc)
    pos = (yy == 1).astype(float) * w
    neg = (yy == 0).astype(float) * w
    order = np.argsort(sc)
    s_sorted = sc[order]
    # balanced accuracy for threshold t: sum pos w with s >= t + sum neg w with s < t
    cpos = np.cumsum(pos[order][::-1])[::-1]               # pos weight with s >= s_sorted[i]
    cneg = np.cumsum(neg[order]) - neg[order]              # neg weight with s < s_sorted[i]
    first = np.searchsorted(s_sorted, ths, side="left")
    ba = cpos[first] + cneg[first]
    k = int(np.argmax(ba))
    return float(ths[k]), float(ba[k])


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")
