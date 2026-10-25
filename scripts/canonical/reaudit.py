"""E-019 re-audit on canonical copies (see docs/EXPERIMENTS.md, E-019). Reuses E-018 functions by import
(the E-018 script is not modified). DeepAction: metadata and residual-pixel baselines with grouped CV.
DF26: descriptive statistics only, no classifier is fitted on DF26 labels.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/canonical/reaudit.py
"""
import csv
import importlib.util
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e018", ROOT / "scripts" / "audit" / "e018_metadata_audit.py")
e18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e18)

CAN = ROOT / "data" / "_canonical"
OUT = ROOT / "data" / "_results" / "canonical"
GENS, REAL = e18.GENS, e18.REAL
PIX = ["n_frames_decoded", "mad_mean", "mad_std", "lapvar_mean", "lapvar_std", "luma_mean", "sat_mean"]


def pixel_feats(path):
    cap = cv2.VideoCapture(str(path))
    gray, sat = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        sat.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    cap.release()
    n = len(gray)
    mad = [float(np.abs(gray[i + 1] - gray[i]).mean()) for i in range(n - 1)]
    lap = [float(cv2.Laplacian(g, cv2.CV_32F).var()) for g in gray]
    nan = float("nan")
    return dict(n_frames_decoded=float(n), mad_mean=float(np.mean(mad)) if mad else nan,
                mad_std=float(np.std(mad)) if mad else nan, lapvar_mean=float(np.mean(lap)) if lap else nan,
                lapvar_std=float(np.std(lap)) if lap else nan,
                luma_mean=float(np.mean([g.mean() for g in gray])) if gray else nan,
                sat_mean=float(np.mean(sat)) if sat else nan)


def feats(path):
    d = e18.probe(path)
    d.update(pixel_feats(path))
    return d


def write_csv(name, rows, cols):
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def split_cols(rows, num, cat):
    cn, cc, const = [], [], {}
    for c in num:
        a = np.array([r[c] for r in rows], dtype=float)
        u = np.unique(a[~np.isnan(a)])
        if len(u) <= 1:
            const[c] = float(u[0]) if len(u) else None
        else:
            cn.append(c)
    for c in cat:
        u = sorted(set(str(r[c]) for r in rows))
        if len(u) <= 1:
            const[c] = u[0]
        else:
            cc.append(c)
    return cn, cc, const


def audit(rows, num, cat, label, rng):
    cn, cc, const = split_cols(rows, num, cat)
    src = np.array([r["source"] for r in rows])
    gid = np.array([r["prompt_id"] for r in rows])
    y = (src != REAL).astype(int)
    folds = e18.make_folds(gid)
    ug = np.unique(gid)
    rbg = {g: np.where(gid == g)[0] for g in ug}
    R = np.array(rows, dtype=object)
    res = dict(constant_features=const, used_numeric=cn, used_categorical=cc, A={}, B={})
    for name in ["lr", "gbt"]:
        oof = np.full(len(rows), np.nan)
        for k in range(e18.N_FOLDS):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            enc = e18.Encoder(cn, cc).fit(R[tr])
            m, _ = e18.fit_model(name, enc.transform(R[tr]), y[tr], gid[tr])
            oof[te] = m.predict_proba(enc.transform(R[te]))[:, 1]
        ci = e18.boot_ci(lambda idx: roc_auc_score(y[idx], oof[idx]), gid, rbg, ug, rng)
        res["A"][name] = dict(auc=float(roc_auc_score(y, oof)), ci95=ci)
        print(label, "A", name, res["A"][name], flush=True)
        res["B"][name] = {}
        for g in GENS:
            o2 = np.full(len(rows), np.nan)
            for k in range(e18.N_FOLDS):
                te = np.where((folds == k) & ((src == g) | (src == REAL)))[0]
                tr = np.where((folds != k) & (src != g))[0]
                enc = e18.Encoder(cn, cc).fit(R[tr])
                m, _ = e18.fit_model(name, enc.transform(R[tr]), y[tr], gid[tr])
                o2[te] = m.predict_proba(enc.transform(R[te]))[:, 1]
            sel = np.where(~np.isnan(o2))[0]
            yy, ss, gg = y[sel], o2[sel], gid[sel]
            rb2 = {q: np.where(gg == q)[0] for q in ug}
            c2 = e18.boot_ci(lambda idx: roc_auc_score(yy[idx], ss[idx]), gg, rb2, ug, rng)
            res["B"][name][g] = dict(auc=float(roc_auc_score(yy, ss)), ci95=c2)
            print(label, "B", name, g, res["B"][name][g], flush=True)
    lo = [res["A"][n]["ci95"][0] for n in res["A"]]
    hi = [res["A"][n]["ci95"][1] for n in res["A"]]
    res["verdict_A"] = "REMAINS" if any(v > 0.70 for v in lo) else ("REMOVED" if all(v < 0.60 for v in hi) else "INCONCLUSIVE")
    single = []
    for c in cn:
        a = np.array([r[c] for r in rows], dtype=float)
        a = np.where(np.isnan(a), np.nanmedian(a), a)
        v = roc_auc_score(y, a)
        single.append(dict(feature=c, auc_direction_free=float(max(v, 1 - v)), generated_higher=bool(v > 0.5)))
    for c in cc:
        for val in sorted(set(str(r[c]) for r in rows)):
            v = roc_auc_score(y, np.array([str(r[c]) == val for r in rows], dtype=float))
            single.append(dict(feature=f"{c}={val}", auc_direction_free=float(max(v, 1 - v)), generated_higher=bool(v > 0.5)))
    res["single_feature"] = sorted(single, key=lambda d: -d["auc_direction_free"])
    return res


def per_source_summary(rows, cols):
    out = {}
    for s in sorted(set(r["source"] for r in rows)):
        rs = [r for r in rows if r["source"] == s]
        d = {"n": len(rs)}
        for c in cols:
            a = np.array([r[c] for r in rs], dtype=float)
            d[c] = [float(np.nanmin(a)), float(np.nanmedian(a)), float(np.nanmax(a))] if not np.all(np.isnan(a)) else None
        out[s] = d
    return out


def counts(vals):
    out = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = dict(env=dict(os=platform.platform(), python=sys.version.split()[0], numpy=np.__version__,
                            opencv=cv2.__version__, ffprobe=e18.run(["ffprobe", "-version"]).splitlines()[0]))
    # ---------- DeepAction
    files = sorted(p for p in (CAN / "deepaction").glob("*/*/*.mp4") if p.parent.parent.name in GENS + [REAL])
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = list(ex.map(feats, files))
    rows = [dict(source=f.parent.parent.name, prompt_id=int(f.parent.name), variant=f.stem, **d) for f, d in zip(files, fs)]
    write_csv("deepaction_canonical_features.csv", rows, ["source", "prompt_id", "variant"] + e18.CAT + e18.NUM + PIX)
    da = dict(n=len(rows), n_real=sum(r["source"] == REAL for r in rows),
              n_generated=sum(r["source"] != REAL for r in rows))
    da["frame_count_by_source"] = {s: counts([int(r["n_frames_decoded"]) for r in rows if r["source"] == s])
                                   for s in [REAL] + GENS}
    rng = np.random.default_rng(e18.SEED)
    da["metadata_only"] = audit(rows, e18.NUM, e18.CAT, "meta", rng)
    da["residual_pixel"] = audit(rows, PIX, [], "pixel", rng)
    da["per_source_pixel"] = per_source_summary(rows, PIX)
    da["per_source_meta"] = per_source_summary(rows, ["file_size", "bit_rate", "fps", "nb_frames"])
    results["deepaction"] = da
    # ---------- DF26 descriptive only
    vid = {}
    with open(ROOT / "data" / "df26" / "metadata" / "videos.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid[r["video_path"]] = r
    base = CAN / "df26"
    dfiles = sorted(base.rglob("*.mp4"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        dfs = list(ex.map(feats, dfiles))
    drows, missing = [], 0
    for f, d in zip(dfiles, dfs):
        rel = f.relative_to(base).as_posix()
        m = vid.get(rel)
        if m is None:
            missing += 1
            continue
        drows.append(dict(source=m["generator"], label=m["label"], path=rel, **d))
    write_csv("df26_canonical_features.csv", drows, ["source", "label", "path"] + e18.CAT + e18.NUM + PIX)
    cols = ["fps", "nb_frames", "file_size", "bit_rate", "gop_mean_i_dist", "width", "height", "duration"] + PIX
    labels = sorted(set(r["label"] for r in drows))
    dd = dict(n=len(drows), n_not_in_metadata=missing, by_label={}, by_generator=per_source_summary(drows, cols),
              categorical_by_label={})
    for lab in labels:
        dd["by_label"][lab] = per_source_summary([dict(r, source=lab) for r in drows if r["label"] == lab], cols)[lab]
    for c in e18.CAT:
        dd["categorical_by_label"][c] = {lab: counts([str(r[c]) for r in drows if r["label"] == lab]) for lab in labels}
    results["df26_descriptive_only"] = dd
    (OUT / "e019_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
