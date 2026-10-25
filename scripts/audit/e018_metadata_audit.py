"""E-018: metadata-only shortcut audit on DeepAction v1 (see docs/EXPERIMENTS.md, E-018).

Reads only container/stream headers and the picture-type sequence of the first 60 video frames
(ffprobe). No pixel or audio sample is decoded. Not a detector; nothing here trains on content.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/audit/e018_metadata_audit.py
"""
import csv
import json
import os
import platform
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sklearn
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "deepaction_v1"
OUT = ROOT / "data" / "_results" / "audit"
GENS = ["BDAnimateDiffLightning", "CogVideoX5B", "RunwayML", "StableDiffusion", "VideoPoet", "Veo"]
REAL = "Pexels"
NB_BOOT = 2000
N_FOLDS = 5
SEED = 0

NUM = ["width", "height", "fps", "duration", "nb_frames", "bit_rate", "file_size", "gop_mean_i_dist",
       "gop_n_i_in_60", "gop_frac_b", "n_audio_streams", "has_audio", "has_creation_time", "n_frames_probed"]
CAT = ["container", "codec", "profile", "pix_fmt", "encoder_tag", "color_space", "color_range",
       "color_transfer", "color_primaries"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout


def rate(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else np.nan
    except Exception:
        return np.nan


def num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def probe(path):
    j = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))
    fmt, streams = j.get("format", {}), j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    ftags = {k.lower(): val for k, val in fmt.get("tags", {}).items()}
    vtags = {k.lower(): val for k, val in v.get("tags", {}).items()}
    enc = ftags.get("encoder") or vtags.get("encoder") or vtags.get("handler_name") or "none"
    frames = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#60",
                  "-show_entries", "frame=pict_type", "-of", "csv=p=0", str(path)]).split()
    types = [t.strip(",") for t in frames if t.strip(",")]
    idx = [i for i, t in enumerate(types) if t == "I"]
    gop = float(np.mean(np.diff(idx))) if len(idx) >= 2 else 60.0
    return dict(
        container=fmt.get("format_name", "none"), codec=v.get("codec_name", "none"),
        profile=v.get("profile", "none"), pix_fmt=v.get("pix_fmt", "none"),
        width=num(v.get("width")), height=num(v.get("height")), fps=rate(v.get("avg_frame_rate", "")),
        duration=num(fmt.get("duration")), nb_frames=num(v.get("nb_frames")), bit_rate=num(fmt.get("bit_rate")),
        file_size=float(os.path.getsize(path)), gop_mean_i_dist=gop, gop_n_i_in_60=float(len(idx)),
        gop_frac_b=(types.count("B") / len(types)) if types else np.nan, n_frames_probed=float(len(types)),
        n_audio_streams=float(len(audio)), has_audio=float(len(audio) > 0), encoder_tag=str(enc),
        has_creation_time=float("creation_time" in ftags or "creation_time" in vtags),
        color_space=v.get("color_space", "none"), color_range=v.get("color_range", "none"),
        color_transfer=v.get("color_transfer", "none"), color_primaries=v.get("color_primaries", "none"),
    )


def extract():
    files = sorted(DATA.glob("*/*/*.mp4"))
    files = [f for f in files if f.parent.parent.name in GENS + [REAL]]
    with ThreadPoolExecutor(max_workers=12) as ex:
        feats = list(ex.map(probe, files))
    rows = []
    for f, ft in zip(files, feats):
        rows.append(dict(source=f.parent.parent.name, prompt_id=int(f.parent.name), variant=f.stem, **ft))
    return rows


# ---------------------------------------------------------------- encoding
class Encoder:
    def __init__(self, num_cols, cat_cols):
        self.num_cols, self.cat_cols = num_cols, cat_cols

    def _blocks(self, R):
        N = np.array([[r[c] for c in self.num_cols] for r in R], dtype=float).reshape(len(R), -1)
        C = np.array([[str(r[c]) for c in self.cat_cols] for r in R], dtype=object).reshape(len(R), -1)
        return N, C

    def fit(self, R):
        N, C = self._blocks(R)
        self.imp = SimpleImputer(strategy="median").fit(N)
        self.sc = StandardScaler().fit(self.imp.transform(N))
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(C) if self.cat_cols else None
        self.groups = list(self.num_cols)
        if self.ohe is not None:
            for c, cats in zip(self.cat_cols, self.ohe.categories_):
                self.groups += [c] * len(cats)
        self.names = list(self.num_cols)
        if self.ohe is not None:
            for c, cats in zip(self.cat_cols, self.ohe.categories_):
                self.names += [f"{c}={v}" for v in cats]
        return self

    def transform(self, R):
        N, C = self._blocks(R)
        parts = [self.sc.transform(self.imp.transform(N))]
        if self.ohe is not None:
            parts.append(self.ohe.transform(C))
        return np.hstack(parts)


def make_folds(group_ids):
    rng = np.random.default_rng(SEED)
    ug = np.unique(group_ids)
    perm = rng.permutation(ug)
    fold_of = {g: i % N_FOLDS for i, g in enumerate(perm)}
    return np.array([fold_of[g] for g in group_ids])


def build(name, params):
    if name == "lr":
        return LogisticRegression(C=params, max_iter=2000, class_weight="balanced")
    return GradientBoostingClassifier(max_depth=2, n_estimators=100, learning_rate=params, subsample=1.0,
                                      random_state=SEED)


GRID = {"lr": [0.1, 1.0, 10.0], "gbt": [0.05, 0.1]}


def fit_model(name, X, y, g, multiclass=False):
    """Inner 3 group folds select the grid value; refit on the whole training set."""
    sw = compute_sample_weight("balanced", y)
    rng = np.random.default_rng(SEED + 1)
    ug = np.unique(g)
    perm = rng.permutation(ug)
    inner = {gg: i % 3 for i, gg in enumerate(perm)}
    fi = np.array([inner[gg] for gg in g])
    best, bs = None, -np.inf
    for p in GRID[name]:
        sc = []
        for k in range(3):
            tr, te = fi != k, fi == k
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = build(name, p)
            if name == "gbt":
                m.fit(X[tr], y[tr], sample_weight=sw[tr])
            else:
                m.fit(X[tr], y[tr])
            pr = m.predict_proba(X[te])
            if multiclass:
                sc.append(-log_loss(y[te], pr, labels=m.classes_) if len(m.classes_) == len(np.unique(y)) else -np.inf)
            else:
                sc.append(roc_auc_score(y[te], pr[:, 1]))
        s = np.mean(sc) if sc else -np.inf
        if s > bs:
            best, bs = p, s
    m = build(name, best if best is not None else GRID[name][0])
    if name == "gbt":
        m.fit(X, y, sample_weight=sw)
    else:
        m.fit(X, y)
    return m, best


# ---------------------------------------------------------------- bootstrap
def boot_ci(stat_fn, gids, rows_by_group, ug, rng):
    vals = []
    for _ in range(NB_BOOT):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([rows_by_group[p] for p in pick])
        try:
            vals.append(stat_fn(idx))
        except ValueError:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ver = run(["ffprobe", "-version"]).splitlines()[0]
    rows = extract()
    cols = ["source", "prompt_id", "variant"] + CAT + NUM
    with open(OUT / "e018_features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print("files probed:", len(rows), flush=True)

    # constants (label-free, over the whole corpus)
    constant, num_cols, cat_cols = {}, [], []
    for c in NUM:
        u = np.unique(np.array([r[c] for r in rows], dtype=float)[~np.isnan([r[c] for r in rows])])
        if len(u) <= 1:
            constant[c] = float(u[0]) if len(u) else None
        else:
            num_cols.append(c)
    for c in CAT:
        u = sorted(set(str(r[c]) for r in rows))
        if len(u) <= 1:
            constant[c] = u[0]
        else:
            cat_cols.append(c)
    distinct = {c: len(set(str(r[c]) for r in rows)) for c in NUM + CAT}
    # per-source summary of every feature (for the write-up)
    per_source = {}
    for s in [REAL] + GENS:
        rs = [r for r in rows if r["source"] == s]
        d = {"n": len(rs)}
        for c in num_cols:
            a = np.array([r[c] for r in rs], dtype=float)
            d[c] = [float(np.nanmin(a)), float(np.nanmedian(a)), float(np.nanmax(a))] if not np.all(np.isnan(a)) else None
        for c in cat_cols:
            vals, cnt = np.unique([str(r[c]) for r in rs], return_counts=True)
            d[c] = {str(v): int(k) for v, k in zip(vals, cnt)}
        per_source[s] = d

    src = np.array([r["source"] for r in rows])
    gid = np.array([r["prompt_id"] for r in rows])
    y_real = (src != REAL).astype(int)  # 1 = generated
    folds = make_folds(gid)
    ug = np.unique(gid)
    rows_by_group = {g: np.where(gid == g)[0] for g in ug}
    rng = np.random.default_rng(SEED)
    R = np.array(rows, dtype=object)
    results = dict(env=dict(os=platform.platform(), python=sys.version.split()[0], sklearn=sklearn.__version__,
                            numpy=np.__version__, ffprobe=ver),
                   n_files=len(rows), n_real=int((y_real == 0).sum()), n_generated=int(y_real.sum()),
                   constant_features=constant, distinct_values=distinct, per_source=per_source,
                   features_used_numeric=num_cols, features_used_categorical=cat_cols)

    def auc_fn(y, s):
        return lambda idx: roc_auc_score(y[idx], s[idx])

    # ----- Condition A
    cond_a, oof_store, fold_models = {}, {}, {}
    for name in ["lr", "gbt"]:
        oof = np.full(len(rows), np.nan)
        chosen, fm = [], []
        for k in range(N_FOLDS):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            enc = Encoder(num_cols, cat_cols).fit(R[tr])
            Xtr, Xte = enc.transform(R[tr]), enc.transform(R[te])
            m, p = fit_model(name, Xtr, y_real[tr], gid[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
            chosen.append(p)
            fm.append((m, enc, te))
        oof_store[name], fold_models[name] = oof, fm
        auc = roc_auc_score(y_real, oof)
        ci = boot_ci(auc_fn(y_real, oof), gid, rows_by_group, ug, rng)
        per_gen = {}
        for g in GENS:
            sel = np.where((src == g) | (src == REAL))[0]
            yy, ss = y_real[sel], oof[sel]
            gg = gid[sel]
            rbg = {q: np.where(gg == q)[0] for q in ug}
            ci_g = boot_ci(lambda idx: roc_auc_score(yy[idx], ss[idx]), gg, rbg, ug, rng)
            per_gen[g] = dict(auc=float(roc_auc_score(yy, ss)), ci95=ci_g)
        cond_a[name] = dict(auc=float(auc), ci95=ci, chosen_hyperparameter_per_fold=chosen, per_generator=per_gen)
        print("A", name, auc, ci, flush=True)
    results["condition_A_real_vs_generated"] = cond_a
    lo = {n: cond_a[n]["ci95"][0] for n in cond_a}
    hi = {n: cond_a[n]["ci95"][1] for n in cond_a}
    if any(v > 0.70 for v in lo.values()):
        verdict = "INDICATED"
    elif all(v < 0.60 for v in hi.values()):
        verdict = "NOT INDICATED"
    else:
        verdict = "INCONCLUSIVE"
    results["verdict_rule_condition_A"] = verdict

    # ----- Condition B: leave-one-generator-out
    cond_b = {}
    for name in ["lr", "gbt"]:
        cond_b[name] = {}
        for g in GENS:
            oof = np.full(len(rows), np.nan)
            for k in range(N_FOLDS):
                te = np.where((folds == k) & ((src == g) | (src == REAL)))[0]
                tr = np.where((folds != k) & (src != g))[0]
                enc = Encoder(num_cols, cat_cols).fit(R[tr])
                m, _ = fit_model(name, enc.transform(R[tr]), y_real[tr], gid[tr])
                oof[te] = m.predict_proba(enc.transform(R[te]))[:, 1]
            sel = np.where(~np.isnan(oof))[0]
            yy, ss, gg = y_real[sel], oof[sel], gid[sel]
            rbg = {q: np.where(gg == q)[0] for q in ug}
            ci = boot_ci(lambda idx: roc_auc_score(yy[idx], ss[idx]), gg, rbg, ug, rng)
            cond_b[name][g] = dict(auc=float(roc_auc_score(yy, ss)), ci95=ci)
            print("B", name, g, cond_b[name][g], flush=True)
    results["condition_B_leave_one_generator_out"] = cond_b

    # ----- Condition C: 7-class identification
    classes = [REAL] + GENS
    y7 = np.array([classes.index(s) for s in src])
    cond_c = {}
    for name in ["lr", "gbt"]:
        pred = np.zeros(len(rows), dtype=int)
        for k in range(N_FOLDS):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            enc = Encoder(num_cols, cat_cols).fit(R[tr])
            m, _ = fit_model(name, enc.transform(R[tr]), y7[tr], gid[tr], multiclass=True)
            pred[te] = m.predict(enc.transform(R[te]))
        acc_fn = lambda idx: float(np.mean(pred[idx] == y7[idx]))
        bal_fn = lambda idx: float(balanced_accuracy_score(y7[idx], pred[idx]))
        conf = np.zeros((7, 7), dtype=int)
        for a, b in zip(y7, pred):
            conf[a, b] += 1
        cond_c[name] = dict(accuracy=acc_fn(np.arange(len(rows))), accuracy_ci95=boot_ci(acc_fn, gid, rows_by_group, ug, rng),
                            balanced_accuracy=bal_fn(np.arange(len(rows))),
                            balanced_accuracy_ci95=boot_ci(bal_fn, gid, rows_by_group, ug, rng),
                            confusion_rows_true_cols_pred=conf.tolist(), class_order=classes)
        print("C", name, cond_c[name]["accuracy"], cond_c[name]["balanced_accuracy"], flush=True)
    cond_c["chance_balanced_accuracy"] = 1 / 7
    cond_c["chance_accuracy_constant_majority_predictor"] = float(np.bincount(y7).max() / len(y7))
    results["condition_C_generator_identification"] = cond_c

    # ----- attribution: single-feature AUC (direction-free) and grouped permutation importance (gbt, cond A)
    single = []
    for c in num_cols:
        a = np.array([r[c] for r in rows], dtype=float)
        a = np.where(np.isnan(a), np.nanmedian(a), a)
        v = roc_auc_score(y_real, a)
        single.append((c, float(max(v, 1 - v))))
    for c in cat_cols:
        for val in sorted(set(str(r[c]) for r in rows)):
            ind = np.array([str(r[c]) == val for r in rows], dtype=float)
            v = roc_auc_score(y_real, ind)
            single.append((f"{c}={val}", float(max(v, 1 - v))))
    single.sort(key=lambda t: -t[1])
    results["single_feature_auc_real_vs_generated"] = [dict(feature=n, auc_direction_free=a) for n, a in single]

    prng = np.random.default_rng(SEED + 2)
    imp = {}
    for m, enc, te in fold_models["gbt"]:
        Xte = enc.transform(R[te])
        base = roc_auc_score(y_real[te], m.predict_proba(Xte)[:, 1])
        groups = np.array(enc.groups)
        for f in sorted(set(groups)):
            cols_f = np.where(groups == f)[0]
            drops = []
            for _ in range(10):
                Xp = Xte.copy()
                Xp[:, cols_f] = Xte[prng.permutation(len(te))][:, cols_f]
                drops.append(base - roc_auc_score(y_real[te], m.predict_proba(Xp)[:, 1]))
            imp.setdefault(f, []).append(float(np.mean(drops)))
    results["permutation_importance_gbt_auc_drop"] = sorted(
        [dict(feature=f, mean_auc_drop=float(np.mean(v))) for f, v in imp.items()],
        key=lambda d: -d["mean_auc_drop"])

    with open(OUT / "e018_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("VERDICT", verdict, "done", flush=True)


if __name__ == "__main__":
    main()
