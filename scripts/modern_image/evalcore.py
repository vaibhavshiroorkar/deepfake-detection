"""Evaluation core shared by E-028 and E-029: Eval rows, strata, sets (generator, architecture, pooled, masks), metrics with a joint stratified
bootstrap (reals and fakes resampled separately within each stratum; the same draws for every scorer, so differences are paired).

Strata: (model_name, label, real_source) for every Eval cell, except that all RAISE reals form one stratum (the pooled RAISE cell of E-028).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lib
import _env

GENS = None
RAISE_GENS = ["DeciDiffusionV2", "GALIP", "kandinsky_2_2", "LCM_lora_sdv15", "LCM_lora_sdxl", "LCM_lora_ssd1b", "kvikontent_midjourney_v6",
              "stable_cascade"]


def load_eval():
    """All extracted Eval rows joined with the index: dict of arrays (one entry per scored image) and the IMG-13/IMG-15 features."""
    rows = {(r["shard"], int(r["row"])): r for r in lib.subset_rows("eval")}
    cache = _env.RES / "e028" / "cache"
    c15 = _env.RES / "e029" / "cache_eval15"
    out = {k: [] for k in ["shard", "row", "label", "model_name", "real_source", "architecture", "format", "resolution", "h", "w", "n_faces",
                           "ti", "tv"]}
    f13, f15, errs, re = [], [], [], []
    for f in sorted(cache.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        shard = "data/" + f.stem + ".parquet"
        for k, rw in enumerate(z["row"]):
            r = rows[(shard, int(rw))]
            out["shard"].append(shard)
            out["row"].append(int(rw))
            out["label"].append(int(r["label"]))
            for c in ["model_name", "real_source", "architecture", "format", "resolution"]:
                out[c].append(r[c])
            out["h"].append(int(z["h"][k]))
            out["w"].append(int(z["w"][k]))
            out["n_faces"].append(int(z["n_faces"][k]))
            out["ti"].append(float(z["ti"][k]))
            out["tv"].append(float(z["tv"][k]))
        f13.append(z["f13"])
        f15.append(np.load(c15 / (f.stem + ".npy")))
        errs += [dict(e, shard=shard) for e in json.loads(str(z["err"]))]
        if "re_row" in z.files:
            r15 = np.load(c15 / (f.stem + "_re.npy"))
            for k, rw in enumerate(z["re_row"]):
                re.append({"shard": shard, "row": int(rw), "ti": float(z["re_ti"][k]), "f13": z["re_f13"][k], "f15": r15[k]})
    E = {k: np.array(v) for k, v in out.items()}
    E["f13"] = np.concatenate(f13)
    E["f15"] = np.concatenate(f15)
    E["errors"] = errs
    E["re"] = re
    E["key"] = np.array([f"{s}#{r}" for s, r in zip(E["shard"], E["row"])])
    return E


def strata_of(E):
    st = []
    for lab, m, rs in zip(E["label"], E["model_name"], E["real_source"]):
        st.append("RAISE|0|RAISE" if (lab == 0 and rs == "RAISE") else f"{m}|{lab}|{rs}")
    return np.array(st)


def define_sets(E):
    """name -> boolean row mask. Generators: fakes of g plus its paired reals (pooled RAISE reals for the RAISE-paired ones)."""
    st = strata_of(E)
    lab, mn, rs, arch = E["label"], E["model_name"], E["real_source"], E["architecture"]
    gens = sorted(set(mn[lab == 1]))
    sets = {}
    raise_real = (lab == 0) & (rs == "RAISE")
    for g in gens:
        fake = (lab == 1) & (mn == g)
        real = raise_real if g in RAISE_GENS else ((lab == 0) & (mn == g))
        sets[f"gen:{g}"] = fake | real
    for src in ["ffhq", "imagenet"]:
        sets[f"cell:Hourglass|{src}"] = (mn == "Hourglass") & (rs == src)
    arch_of = {g: arch[(lab == 1) & (mn == g)][0] for g in gens}
    for a in sorted(set(arch_of.values())):
        m = np.zeros(len(lab), bool)
        for g, aa in arch_of.items():
            if aa == a:
                m |= sets[f"gen:{g}"]
        sets[f"arch:{a}"] = m
    sets["pooled"] = np.ones(len(lab), bool)
    return st, gens, arch_of, sets


class JointBoot:
    def __init__(self, strata, n_boot=lib.N_BOOT, seed=lib.SEED):
        rng = np.random.default_rng(seed)
        self.n = len(strata)
        self.groups = {s: np.flatnonzero(strata == s) for s in sorted(set(strata))}
        self.n_boot = n_boot
        # W[b, i] = how many times row i is drawn in resample b (multinomial counts = resampling with replacement within the stratum)
        W = np.zeros((n_boot, self.n), np.int16)
        for s, idx in self.groups.items():
            W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
        self.W = W


def metrics_matrix(y, s, W, t):
    """Metrics for every row of the weight matrix W [B, n] at once (W = bootstrap multiplicities; a row of ones = the point estimate).
    AUC: Mann-Whitney with ties counted one half. PR-AUC: average precision over distinct thresholds (sklearn's definition). The rest at
    threshold t: FPR, FNR, precision, recall, F1, balanced accuracy. Returns [B, 8]."""
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    W = W[:, order].astype(np.float32)
    first = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos = (y == 1)
    Wp, Wn = W * pos, W * (~pos)
    gp, gn = np.add.reduceat(Wp, first, axis=1), np.add.reduceat(Wn, first, axis=1)
    P, N = gp.sum(1), gn.sum(1)
    cn = np.cumsum(gn, 1) - gn
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (gp * (cn + 0.5 * gn)).sum(1) / (P * N)
        tp, fp = np.cumsum(gp[:, ::-1], 1), np.cumsum(gn[:, ::-1], 1)
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / P[:, None]
        ap = (np.diff(np.concatenate([np.zeros((len(W), 1), np.float32), rec], 1), axis=1) * prec).sum(1)
        above = s >= t
        TP, FP = Wp[:, above].sum(1), Wn[:, above].sum(1)
        FN, TN = P - TP, N - FP
        fpr, fnr = FP / (FP + TN), FN / (FN + TP)
        pr = np.where(TP + FP > 0, TP / (TP + FP), np.nan)
        rc = 1 - fnr
        f1 = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), np.nan)
        ba = 1 - (fpr + fnr) / 2
    return np.stack([auc, ap, fpr, fnr, pr, rc, f1, ba], 1)


def metrics_w(y, s, w, t):
    return metrics_matrix(y, s, w[None, :], t)[0]


KEYS = ["auc", "pr_auc", "fpr", "fnr", "precision", "recall", "f1", "balanced_accuracy"]


def evaluate(E, scores: dict, thresholds: dict, sets: dict, boot: JointBoot, gens, pairs=(), masks=None, macro_name="macro"):
    """Point estimates and 95 percent intervals for every scorer and set (optionally intersected with a mask), the macro mean over gens,
    and paired differences for (a, b) in pairs. Returns a nested dict."""
    y = E["label"]
    masks = masks or {"all": None}
    res = {}
    for mname, mk in masks.items():
        for setname, m in sets.items():
            mm = m if mk is None else (m & mk)
            idx = np.flatnonzero(mm)
            if len(idx) == 0 or len(set(y[idx])) < 2:
                continue
            for sc, s in scores.items():
                ok = idx[~np.isnan(s[idx])]
                if len(ok) == 0 or len(set(y[ok])) < 2:
                    continue
                pt = metrics_w(y[ok], s[ok], np.ones(len(ok)), thresholds[sc])
                reps = metrics_matrix(y[ok], s[ok], boot.W[:, ok], thresholds[sc])
                res[(mname, setname, sc)] = (pt, reps, int((y[ok] == 1).sum()), int((y[ok] == 0).sum()))
    out = {}
    for (mname, setname, sc), (pt, reps, n1, n0) in res.items():
        out.setdefault(mname, {}).setdefault(setname, {})[sc] = {
            "n_fake": n1, "n_real": n0,
            **{k: {"est": lib.r4(pt[i]), "ci": [lib.r4(x) for x in lib.pct(reps[:, i])]} for i, k in enumerate(KEYS)}}
    # macro over generators
    for mname in masks:
        for sc in scores:
            keys = [(mname, f"gen:{g}", sc) for g in gens]
            if not all(k in res for k in keys):
                continue
            pt = np.nanmean([res[k][0] for k in keys], axis=0)
            reps = np.nanmean(np.stack([res[k][1] for k in keys]), axis=0)
            out[mname].setdefault(macro_name, {})[sc] = {k: {"est": lib.r4(pt[i]), "ci": [lib.r4(x) for x in lib.pct(reps[:, i])]}
                                                        for i, k in enumerate(KEYS)}
            res[(mname, macro_name, sc)] = (pt, reps, None, None)
    # paired differences
    diffs = {}
    for a, b in pairs:
        for (mname, setname, sc), v in list(res.items()):
            if sc != a or (mname, setname, b) not in res:
                continue
            pa, ra = v[0], v[1]
            pb, rb = res[(mname, setname, b)][0], res[(mname, setname, b)][1]
            d = ra - rb
            diffs.setdefault(mname, {}).setdefault(setname, {})[f"{a} minus {b}"] = {
                k: {"est": lib.r4(pa[i] - pb[i]), "ci": [lib.r4(x) for x in lib.pct(d[:, i])]} for i, k in enumerate(KEYS)}
    return out, diffs, res
