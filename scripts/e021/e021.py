"""E-021: follow-up on the DEVELOPMENT half of FakeAVCeleb (dev_x only). Protocol: docs/EXPERIMENTS.md E-021.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/e021/e021.py sample|probe|recode|score|pixels|analyse [--recoded]

Only rows whose split is dev_x are used; the script aborts otherwise. No checkpoint, threshold or temperature is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_results" / "e021"
REC = OUT / "recoded"
SPLIT = DATA / "_splits" / "fakeavceleb-79d3f7051972.csv"
MANIFEST = DATA / "_manifests" / "fakeavceleb.manifest.csv"
CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
SEED = 20261026
NB = 2000


def load_dev() -> dict:
    from dfd.data.fakeavceleb_split import note_field
    from dfd.data.manifest import read_manifest
    from dfd.data.splits import read_split
    asg = read_split(SPLIT)
    out = {}
    for r in read_manifest(MANIFEST):
        if asg.get(r.sample_id) == "dev_x":
            out[r.sample_id] = {"sid": r.sample_id, "path": r.path, "cat": note_field(r.notes, "category"), "method": r.method,
                                "group": r.group_id, "cell": note_field(r.notes, "cell")}
    return out


def read_sample() -> list[dict]:
    with open(OUT / "sample.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    dev = load_dev()
    for r in rows:
        assert r["sid"] in dev, "non dev_x clip in sample"
    return rows


def cmd_sample(a):
    dev = load_dev()
    rng = np.random.default_rng(SEED)
    sel = [v for v in dev.values() if v["cat"] in ("RVRA", "RVFA")]
    for cat in ("FVRA", "FVFA"):
        pool = sorted([v for v in dev.values() if v["cat"] == cat], key=lambda v: v["sid"])
        # proportional allocation over cell, then random inside the cell
        cells = sorted({v["cell"] for v in pool})
        want = 250
        shuffled = [pool[i] for i in rng.permutation(len(pool))]
        chosen = []
        quota = {c: want * sum(1 for v in pool if v["cell"] == c) / len(pool) for c in cells}
        base = {c: int(np.floor(q)) for c, q in quota.items()}
        rest = want - sum(base.values())
        for c in sorted(cells, key=lambda c: -(quota[c] - base[c]))[:rest]:
            base[c] += 1
        for c in cells:
            chosen += [v for v in shuffled if v["cell"] == c][: base[c]]
        sel += chosen
    OUT.mkdir(parents=True, exist_ok=True)
    sel.sort(key=lambda v: v["sid"])
    with open(OUT / "sample.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid", "path", "cat", "method", "group", "cell"])
        w.writeheader()
        w.writerows(sel)
    from collections import Counter
    print(len(sel), Counter(v["cat"] for v in sel), "identities", len({v["group"] for v in sel}))


def ffprobe(p: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)], capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    au = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), {})
    fm = j.get("format", {})

    def num(x, d=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return d
    fr = v.get("r_frame_rate", "0/1").split("/")
    fps = num(fr[0]) / max(num(fr[1], 1.0), 1e-9) if len(fr) == 2 else 0.0
    return {"v_codec": v.get("codec_name", "none"), "v_profile": v.get("profile", "none"), "v_bitrate": num(v.get("bit_rate")), "fps": fps,
            "nb_frames": num(v.get("nb_frames")), "v_duration": num(v.get("duration")), "width": num(v.get("width")), "height": num(v.get("height")),
            "pix_fmt": v.get("pix_fmt", "none"), "container": fm.get("format_name", "none"), "size": num(fm.get("size")), "f_bitrate": num(fm.get("bit_rate")),
            "encoder": (fm.get("tags") or {}).get("encoder", "none"), "a_codec": au.get("codec_name", "none"), "a_rate": num(au.get("sample_rate")),
            "a_bitrate": num(au.get("bit_rate")), "av_dur_diff": num(au.get("duration")) - num(v.get("duration")), "n_streams": len(j.get("streams", []))}


def cmd_probe(a):
    rows = read_sample()
    out = []
    with ThreadPoolExecutor(6) as ex:
        for r, pr in zip(rows, ex.map(lambda r: ffprobe(DATA / r["path"]), rows)):
            out.append({"sid": r["sid"], **pr})
    with open(OUT / "probe.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("probed", len(out))


def rec_path(sid: str) -> Path:
    return REC / (sid.replace(":", "_").replace("/", "__") + ".mp4")


def _recode(r):
    o = rec_path(r["sid"])
    if o.exists() and o.stat().st_size > 0:
        return 0
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(DATA / r["path"]), "-an", "-map_metadata", "-1", "-map_chapters", "-1",
           "-vf", "scale=224:224:flags=bicubic,fps=25", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-threads", "1", str(o)]
    return subprocess.run(cmd, capture_output=True).returncode


def cmd_recode(a):
    rows = read_sample()
    REC.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        rc = list(ex.map(_recode, rows))
    print("recoded", len(rows), "failures", sum(1 for x in rc if x))


COLS = ["sample_id", "visual_available", "visual_score", "visual_n_frames", "warnings", "error", "ms"]


def cmd_score(a):
    from dfd.infer.library import inspect
    from dfd.infer.real_adapter import RealAdapter
    rows = read_sample()
    pred = OUT / ("scores_recoded.csv" if a.recoded else "scores_orig.csv")
    done = set()
    if pred.exists():
        with open(pred, encoding="utf-8", newline="") as f:
            done = {r["sample_id"] for r in csv.DictReader(f)}
    todo = [r for r in rows if r["sid"] not in done]
    print("to score", len(todo), flush=True)
    ad = RealAdapter(CKPT)
    new = not pred.exists()
    t0 = time.time()
    with open(pred, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for k, r in enumerate(todo, 1):
            p = rec_path(r["sid"]) if a.recoded else DATA / r["path"]
            row = dict.fromkeys(COLS, "")
            row["sample_id"] = r["sid"]
            t = time.time()
            try:
                res = ad.analyse(inspect(p, p.name))
                v = res.per_modality.get("visual")
                if v is not None:
                    row["visual_available"], row["visual_score"], row["visual_n_frames"] = int(v.available), "" if v.score is None else v.score, v.n_units or 0
                row["warnings"] = "|".join(x["code"] for x in res.warnings)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            row["ms"] = round((time.time() - t) * 1000)
            w.writerow(row)
            if k % 50 == 0:
                f.flush()
                print(k, len(todo), round(time.time() - t0), "s", flush=True)


def pix_one(r):
    import cv2
    cap = cv2.VideoCapture(str(DATA / r["path"]))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        return None
    idx = np.unique(np.linspace(0, len(frames) - 1, 32).astype(int))
    g = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in idx]
    diffs = [float(np.mean(np.abs(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32))))
             for i in idx if i + 1 < len(frames)]
    lap = [float(cv2.Laplacian(x, cv2.CV_32F).var()) for x in g]
    blk = []
    for x in g:
        dh = np.abs(np.diff(x, axis=1))
        cols = np.arange(dh.shape[1])
        on = dh[:, cols % 8 == 7].mean()
        off = dh[:, cols % 8 != 7].mean()
        blk.append(float(on / max(off, 1e-6)))
    return {"sid": r["sid"], "frame_diff": float(np.mean(diffs)), "lap_var": float(np.mean(lap)), "blockiness": float(np.mean(blk)),
            "lum_mean": float(np.mean([x.mean() for x in g])), "lum_std": float(np.mean([x.std() for x in g]))}


def cmd_pixels(a):
    rows = read_sample()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(pix_one, rows))
    res = [x for x in res if x]
    with open(OUT / "pixels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    print("pixels", len(res), "of", len(rows))


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def auc_ci(y, s, g):
    from dfd.eval.bootstrap import cluster_bootstrap
    r = cluster_bootstrap(np.asarray(y), np.asarray(s), np.asarray(g), metric="auc", n_boot=NB)
    return {"auc": round(r.estimate, 4), "low": round(r.low, 4), "high": round(r.high, 4), "n": int(len(y))}


def paired_diff(y, s1, s2, g):
    """Cluster bootstrap of AUC(s2) - AUC(s1) with the same resampled identities."""
    from sklearn.metrics import roc_auc_score
    y, s1, s2, g = map(np.asarray, (y, s1, s2, g))
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NB):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if len(set(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], s2[ii]) - roc_auc_score(y[ii], s1[ii]))
    pt = roc_auc_score(y, s2) - roc_auc_score(y, s1)
    return {"diff": round(float(pt), 4), "low": round(float(np.percentile(d, 2.5)), 4), "high": round(float(np.percentile(d, 97.5)), 4)}


def features_meta(probe, with_audio):
    cat = ["v_codec", "v_profile", "pix_fmt", "container", "encoder"] + (["a_codec"] if with_audio else [])
    num = ["v_bitrate", "fps", "nb_frames", "v_duration", "width", "height", "size", "f_bitrate"] + (["a_rate", "a_bitrate", "av_dur_diff"] if with_audio else [])
    levels = {c: sorted({p[c] for p in probe}) for c in cat}
    X = [[float(p[n]) for n in num] + [1.0 if p[c] == lv else 0.0 for c in cat for lv in levels[c]] for p in probe]
    return np.array(X)


def oof(X, y, g, kind):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        if kind == "logreg":
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        else:
            m = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=SEED)
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def cmd_analyse(a):
    from collections import Counter
    rows = read_sample()
    by = {r["sid"]: r for r in rows}
    probe = {r["sid"]: r for r in read_csv(OUT / "probe.csv")}
    for r in probe.values():
        for k in ("v_bitrate", "fps", "nb_frames", "v_duration", "width", "height", "size", "f_bitrate", "a_rate", "a_bitrate", "av_dur_diff"):
            r[k] = float(r[k])
    so = {r["sample_id"]: r for r in read_csv(OUT / "scores_orig.csv")}
    sr = {r["sample_id"]: r for r in read_csv(OUT / "scores_recoded.csv")}
    px = {r["sid"]: r for r in read_csv(OUT / "pixels.csv")}
    res: dict = {"experiment": "E-021", "date": time.strftime("%Y-%m-%d"), "n_sample": len(rows), "seed": SEED, "n_boot": NB}
    res["sample_by_category"] = dict(Counter(r["cat"] for r in rows))
    res["identities"] = len({r["group"] for r in rows})

    def sc(d, sid):
        r = d.get(sid)
        if r and not r["error"] and r["visual_available"] == "1" and r["visual_score"] != "":
            return float(r["visual_score"])
        return None
    res["scoring_failures"] = {"orig": sum(1 for r in rows if sc(so, r["sid"]) is None), "recoded": sum(1 for r in rows if sc(sr, r["sid"]) is None)}
    res["probe_failures"] = sum(1 for r in rows if probe[r["sid"]]["v_codec"] == "none")

    # ---- Q1 ----
    stem = {}
    for r in rows:
        if r["cat"] in ("RVRA", "RVFA"):
            d, b = os.path.split(r["path"].replace("RealVideo-RealAudio", "RV").replace("RealVideo-FakeAudio", "RV"))
            k = (d, b[:-9] if b.endswith("_fake.mp4") else b[:-4])
            stem.setdefault(k, {})[r["cat"]] = r["sid"]
    pairs = [(v["RVRA"], v["RVFA"]) for v in stem.values() if "RVRA" in v and "RVFA" in v]
    q1: dict = {"n_pairs": len(pairs), "n_unpaired": sum(1 for v in stem.values() if not ("RVRA" in v and "RVFA" in v))}
    desc = {}
    for cat in ("RVRA", "RVFA"):
        ps = [probe[r["sid"]] for r in rows if r["cat"] == cat]
        desc[cat] = {k: dict(Counter(str(p[k]) for p in ps).most_common(6)) for k in ("v_codec", "v_profile", "pix_fmt", "encoder", "a_codec", "a_rate", "fps", "width", "height")}
        desc[cat]["nb_frames_median"] = float(np.median([p["nb_frames"] for p in ps]))
        desc[cat]["v_bitrate_median"] = float(np.median([p["v_bitrate"] for p in ps]))
        desc[cat]["size_median"] = float(np.median([p["size"] for p in ps]))
    q1["descriptive"] = desc
    same_stream = same_frames = same_enc = differ_any = 0
    keys = ["v_codec", "v_profile", "v_bitrate", "fps", "width", "height", "pix_fmt"]
    for x, y in pairs:
        px_, py_ = probe[x], probe[y]
        s_ = all(px_[k] == py_[k] for k in keys)
        f_ = px_["nb_frames"] == py_["nb_frames"]
        e_ = px_["encoder"] == py_["encoder"]
        same_stream += s_
        same_frames += f_
        same_enc += e_
        differ_any += (not s_) or (not f_) or (not e_)
    n = max(len(pairs), 1)
    q1["pair_equal_share"] = {"video_stream_params": round(same_stream / n, 4), "frame_count": round(same_frames / n, 4), "encoder_tag": round(same_enc / n, 4)}
    q1["share_pairs_differing_in_stream_or_frames_or_encoder"] = round(differ_any / n, 4)
    pp = [(x, y) for x, y in pairs if sc(so, x) is not None and sc(so, y) is not None]
    d = np.array([sc(so, y) - sc(so, x) for x, y in pp])
    q1["paired_diff_orig"] = {"n": len(pp), "median": round(float(np.median(d)), 5), "mean": round(float(d.mean()), 5), "share_positive": round(float(np.mean(d > 0)), 4)}
    ppr = [(x, y) for x, y in pairs if sc(sr, x) is not None and sc(sr, y) is not None]
    dr = np.array([sc(sr, y) - sc(sr, x) for x, y in ppr])
    q1["paired_diff_recoded"] = {"n": len(ppr), "median": round(float(np.median(dr)), 5), "mean": round(float(dr.mean()), 5), "share_positive": round(float(np.mean(dr > 0)), 4)}

    def sub(d, cats0, cats1, methods=None):
        xs = [r for r in rows if sc(d, r["sid"]) is not None and (r["cat"] in cats0 or (r["cat"] in cats1 and (methods is None or r["method"] in methods)))]
        return xs, [1 if r["cat"] in cats1 else 0 for r in xs], [sc(d, r["sid"]) for r in xs], [r["group"] for r in xs]
    q1["auc_RVFA_vs_RVRA_orig"] = auc_ci(*sub(so, {"RVRA"}, {"RVFA"})[1:])
    q1["auc_RVFA_vs_RVRA_recoded"] = auc_ci(*sub(sr, {"RVRA"}, {"RVFA"})[1:])
    q1["metadata_audio_only_note"] = "audio fields are in the descriptive block (a_codec, a_rate); the canonical re-encode removes audio"
    ao, ar = q1["auc_RVFA_vs_RVRA_orig"], q1["auc_RVFA_vs_RVRA_recoded"]
    # paired difference on the RVRA/RVFA subset
    xs, y, s1, g = sub(so, {"RVRA"}, {"RVFA"})
    ok = [i for i, r in enumerate(xs) if sc(sr, r["sid"]) is not None]
    q1["auc_diff_recoded_minus_orig"] = paired_diff([y[i] for i in ok], [s1[i] for i in ok], [sc(sr, xs[i]["sid"]) for i in ok], [g[i] for i in ok])
    gap_exists = ao["low"] > 0.5
    differ80 = q1["share_pairs_differing_in_stream_or_frames_or_encoder"] >= 0.8
    shrink = (ar["low"] <= 0.5 <= ar["high"]) or (abs(ar["auc"] - 0.5) <= 0.5 * abs(ao["auc"] - 0.5))
    refuted = ar["low"] > 0.5 and abs(ar["auc"] - 0.5) > 0.75 * abs(ao["auc"] - 0.5)
    q1["verdict"] = ("no gap on dev_x to explain" if not gap_exists else "SUPPORTED" if (differ80 and shrink) else "REFUTED" if refuted else "INCONCLUSIVE")
    q1["verdict_inputs"] = {"gap_exists": bool(gap_exists), "differ_ge_80pct": bool(differ80), "canonical_shrinks_or_contains_0.5": bool(shrink), "refuted_rule": bool(refuted)}
    res["Q1"] = q1

    # ---- Q2 ----
    ids = [r["sid"] for r in rows]
    yv = np.array([1 if by[i]["cat"] in ("FVRA", "FVFA") else 0 for i in ids])
    ya = np.array([1 if by[i]["cat"] in ("RVFA", "FVFA") else 0 for i in ids])
    g = np.array([by[i]["group"] for i in ids])
    PR = [probe[i] for i in ids]
    Xv, Xa = features_meta(PR, False), features_meta(PR, True)
    have_px = [i in px for i in ids]
    Xb = np.array([[float(px[i][k]) if i in px else 0.0 for k in ("frame_diff", "lap_var", "blockiness", "lum_mean", "lum_std")] for i in ids])
    sets = {"A-video": Xv, "A-all": Xa, "B-pixel": Xb, "A-video+B": np.hstack([Xv, Xb])}
    q2: dict = {"pixel_rows_missing": int(len(ids) - sum(have_px))}
    for tname, y in (("video_target", yv), ("audio_target", ya)):
        q2[tname] = {}
        for fs, X in sets.items():
            q2[tname][fs] = {}
            for kind in ("logreg", "gboost"):
                s = oof(X, y, g, kind)
                q2[tname][fs][kind] = auc_ci(y, s, g)
    best = max((q2["video_target"][fs][k]["low"], fs, k) for fs in ("A-video", "A-all", "A-video+B") for k in ("logreg", "gboost"))
    q2["video_target_best_metadata_set"] = {"lower_bound": best[0], "set": best[1], "model": best[2]}
    bp = max(q2["video_target"]["B-pixel"][k]["low"] for k in ("logreg", "gboost"))
    bph = max(q2["video_target"]["B-pixel"][k]["high"] for k in ("logreg", "gboost"))
    q2["verdict_metadata"] = "material" if best[0] > 0.80 else ("negligible" if max(q2["video_target"][fs][k]["high"] for fs in ("A-video", "A-all", "A-video+B") for k in ("logreg", "gboost")) < 0.60 else "moderate")
    q2["verdict_pixel_alone"] = "material" if bp > 0.80 else ("negligible" if bph < 0.60 else "moderate")
    # which single metadata fields separate (descriptive)
    q2["video_codec_by_category"] = {c: dict(Counter(probe[r["sid"]]["v_codec"] for r in rows if r["cat"] == c)) for c in ("RVRA", "RVFA", "FVRA", "FVFA")}
    q2["audio_codec_by_category"] = {c: dict(Counter(probe[r["sid"]]["a_codec"] for r in rows if r["cat"] == c)) for c in ("RVRA", "RVFA", "FVRA", "FVFA")}
    q2["video_codec_by_method"] = {m: dict(Counter(probe[r["sid"]]["v_codec"] for r in rows if r["method"] == m)) for m in sorted({r["method"] for r in rows if r["cat"] in ("FVRA", "FVFA")})}
    res["Q2"] = q2

    # ---- Q3 ----
    q3: dict = {}
    rr = [r for r in rows if sc(so, r["sid"]) is not None and sc(sr, r["sid"]) is not None]
    y = [1 if r["cat"] in ("FVRA", "FVFA") else 0 for r in rr]
    gg = [r["group"] for r in rr]
    s1 = [sc(so, r["sid"]) for r in rr]
    s2 = [sc(sr, r["sid"]) for r in rr]
    q3["n_scored_both"] = len(rr)
    q3["pooled_orig"], q3["pooled_recoded"] = auc_ci(y, s1, gg), auc_ci(y, s2, gg)
    q3["pooled_diff_recoded_minus_orig"] = paired_diff(y, s1, s2, gg)
    q3["by_method"] = {}
    for m in ("faceswap", "fsgan", "wav2lip", "faceswap-wav2lip", "fsgan-wav2lip"):
        sel = [i for i, r in enumerate(rr) if r["cat"] in ("RVRA", "RVFA") or r["method"] == m]
        yy, a1, a2, g2 = [y[i] for i in sel], [s1[i] for i in sel], [s2[i] for i in sel], [gg[i] for i in sel]
        if len(set(yy)) < 2 or sum(yy) < 5:
            continue
        q3["by_method"][m] = {"orig": auc_ci(yy, a1, g2), "recoded": auc_ci(yy, a2, g2), "diff": paired_diff(yy, a1, a2, g2), "n_fake": int(sum(yy))}
    ao_, ar_, dd = q3["pooled_orig"], q3["pooled_recoded"], q3["pooled_diff_recoded_minus_orig"]
    q3["verdict"] = "RETAINED" if (ar_["low"] > 0.80 and not dd["high"] < -0.05) else ("LARGELY LOST" if ar_["high"] < 0.70 else "PARTLY RETAINED")
    q3["score_median_by_category"] = {c: {"orig": round(float(np.median([sc(so, r["sid"]) for r in rr if r["cat"] == c])), 4),
                                         "recoded": round(float(np.median([sc(sr, r["sid"]) for r in rr if r["cat"] == c])), 4)} for c in ("RVRA", "RVFA", "FVRA", "FVFA")}
    res["Q3"] = q3
    (OUT / "e021_results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(res, indent=1, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "probe", "recode", "score", "pixels", "analyse"])
    ap.add_argument("--recoded", action="store_true")
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "probe": cmd_probe, "recode": cmd_recode, "score": cmd_score, "pixels": cmd_pixels, "analyse": cmd_analyse}[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
