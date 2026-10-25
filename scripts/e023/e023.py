"""E-023: codec-matched control, DEVELOPMENT half only. Protocol: docs/EXPERIMENTS.md E-023.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/e023/e023.py prep|score|analyse
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e021"))
import e021  # noqa: E402

O = e021.OUT.parent / "e023"
CAN = O / "canon"
SEED, NB = 20261026, 2000
BR = {"M": 317957, "L": 180000}


def sets() -> dict[str, list[dict]]:
    rows = e021.read_sample()                       # asserts every sid is dev_x
    dev = e021.load_dev()
    S = {"C1": [r for r in rows if r["cat"] == "RVRA"], "C1L": [r for r in rows if r["cat"] == "RVRA"],
         "C2": [r for r in rows if r["cat"] in ("FVRA", "FVFA")]}
    with open(O / "fvra_all_probe.csv", encoding="utf-8") as f:
        fv = list(csv.DictReader(f))
    h = [r for r in fv if r["v_codec"] == "h264"]
    mp = sorted([r for r in fv if r["v_codec"] == "mpeg4"], key=lambda r: r["sid"])
    rng = np.random.default_rng(SEED)
    k = [mp[i] for i in rng.permutation(len(mp))[: len(h)]]
    for name, lst in (("H", h), ("K", k)):
        S[name] = [{"sid": r["sid"], "path": dev[r["sid"]]["path"], "cat": "FVRA", "method": r["method"], "group": r["group"]} for r in lst]
    for v in S.values():
        for r in v:
            assert r["sid"] in dev
    return S


def can_path(name, sid):
    return CAN / name / (sid.replace(":", "_").replace("/", "__") + ".mp4")


def canon_cmd(src, out):
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-map_metadata", "-1", "-map_chapters", "-1", "-vf", "scale=224:224:flags=bicubic,fps=25",
            "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1", str(out)]


def do_one(args):
    name, r = args
    out = can_path(name, r["sid"])
    if out.exists() and out.stat().st_size > 0:
        return (name, r["sid"], 0, "")
    src = e021.DATA / r["path"]
    ach = ""
    mid = out.with_suffix(".mid.mp4")
    if name in ("C1", "C1L", "C2"):
        rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-map_metadata", "-1", "-map_chapters", "-1", "-vf", "fps=25", "-c:v", "mpeg4",
                             "-b:v", str(BR["L" if name == "C1L" else "M"]), "-g", "12", "-bf", "0", "-pix_fmt", "yuv420p", "-threads", "1", str(mid)],
                            capture_output=True).returncode
        if rc:
            return (name, r["sid"], rc, "")
        pr = e021.ffprobe(mid)
        ach = f'{pr["v_bitrate"]:.0f}|{pr["v_codec"]}'
        src = mid
    rc = subprocess.run(canon_cmd(src, out), capture_output=True).returncode
    if mid.exists():
        mid.unlink()
    return (name, r["sid"], rc, ach)


def cmd_prep():
    S = sets()
    jobs = []
    for n in ("C1", "C1L", "C2", "H", "K"):
        (CAN / n).mkdir(parents=True, exist_ok=True)
        jobs += [(n, r) for r in S[n]]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(do_one, jobs))
    with open(O / "prep.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "sid", "rc", "achieved_bitrate|codec"])
        w.writerows(res)
    print("jobs", len(res), "failures", sum(1 for x in res if x[2]))


def cmd_score():
    from dfd.infer.library import inspect
    from dfd.infer.real_adapter import RealAdapter
    S = sets()
    ad = RealAdapter(e021.CKPT)
    todo = [("C1", "C1", True), ("C1L", "C1L", True), ("C2", "C2", True), ("H_orig", "H", False), ("K_orig", "K", False), ("H_can", "H", True), ("K_can", "K", True)]
    pred = O / "scores.csv"
    done = set()
    if pred.exists():
        done = {(r["label"], r["sid"]) for r in e021.read_csv(pred)}
    new = not pred.exists()
    t0 = time.time()
    with open(pred, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "sid", "score", "error"])
        if new:
            w.writeheader()
        n = 0
        for label, sname, can in todo:
            for r in S[sname]:
                if (label, r["sid"]) in done:
                    continue
                p = can_path(sname, r["sid"]) if can else e021.DATA / r["path"]
                row = {"label": label, "sid": r["sid"], "score": "", "error": ""}
                try:
                    v = ad.analyse(inspect(p, p.name)).per_modality.get("visual")
                    if v is not None and v.available and v.score is not None:
                        row["score"] = v.score
                    else:
                        row["error"] = "visual unavailable"
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                w.writerow(row)
                n += 1
                if n % 100 == 0:
                    f.flush()
                    print(n, label, round(time.time() - t0), "s", flush=True)


def run_aucs(specs: dict, diffs: dict):
    """specs: name -> (real [(group, score)], fake [(group, score)]); diffs: name -> (a, b) meaning AUC(a) - AUC(b).
    One identity resampling (real and fake rows of an identity move together) is shared by every AUC and difference."""
    from sklearn.metrics import roc_auc_score
    names = list(specs)
    rows = {n: (np.array([g for g, _ in specs[n][0]]), np.array([s for _, s in specs[n][0]]), np.array([g for g, _ in specs[n][1]]), np.array([s for _, s in specs[n][1]])) for n in names}
    ug = np.unique(np.concatenate([np.concatenate([rows[n][0], rows[n][2]]) for n in names]))
    ix = {n: ({u: np.where(rows[n][0] == u)[0] for u in ug}, {u: np.where(rows[n][2] == u)[0] for u in ug}) for n in names}

    def one(gs):
        out = {}
        for n in names:
            ri = np.concatenate([ix[n][0][u] for u in gs])
            fi = np.concatenate([ix[n][1][u] for u in gs])
            if len(ri) == 0 or len(fi) == 0:
                out[n] = np.nan
                continue
            y = np.r_[np.zeros(len(ri)), np.ones(len(fi))]
            out[n] = roc_auc_score(y, np.r_[rows[n][1][ri], rows[n][3][fi]])
        for d, (a, b) in diffs.items():
            out[d] = out[a] - out[b]
        return out
    rng = np.random.default_rng(SEED)
    pt = one(ug)
    bs = [one(rng.choice(ug, len(ug))) for _ in range(NB)]
    res = {}
    for k in pt:
        v = np.array([b[k] for b in bs], float)
        res[k] = {"est": round(float(pt[k]), 4), "low": round(float(np.nanpercentile(v, 2.5)), 4), "high": round(float(np.nanpercentile(v, 97.5)), 4)}
    res["_n"] = {n: [len(specs[n][0]), len(specs[n][1])] for n in names}
    return res


def verdict(r):
    if r["high"] < 0.60:
        return "SHORTCUT EXPLAINS THE RESIDUAL (upper bound < 0.60)"
    if r["low"] > 0.70:
        return "MANIPULATION EVIDENCE REMAINS (lower bound > 0.70)"
    return "INCONCLUSIVE"


def cmd_analyse():
    S = sets()
    sc, fails = {}, {}
    for r in e021.read_csv(O / "scores.csv"):
        if r["error"] or r["score"] == "":
            fails[r["label"]] = fails.get(r["label"], 0) + 1
        else:
            sc[(r["label"], r["sid"])] = float(r["score"])
    rec = {r["sample_id"]: r for r in e021.read_csv(e021.OUT / "scores_recoded.csv")}
    ori = {r["sample_id"]: r for r in e021.read_csv(e021.OUT / "scores_orig.csv")}
    rows = e021.read_sample()

    def e21(d, sid):
        r = d.get(sid)
        return float(r["visual_score"]) if r and not r["error"] and r["visual_score"] != "" else None
    rv = [r for r in rows if r["cat"] == "RVRA"]
    fk = [r for r in rows if r["cat"] in ("FVRA", "FVFA")]

    def L(items, getter):
        return [(r["group"], getter(r)) for r in items if getter(r) is not None]
    real_can = L(rv, lambda r: e21(rec, r["sid"]))
    real_ori = L(rv, lambda r: e21(ori, r["sid"]))
    real_c1 = L(rv, lambda r: sc.get(("C1", r["sid"])))
    real_c1l = L(rv, lambda r: sc.get(("C1L", r["sid"])))
    fk_can = L(fk, lambda r: e21(rec, r["sid"]))
    fk_ori = L(fk, lambda r: e21(ori, r["sid"]))
    fk_c2 = L(fk, lambda r: sc.get(("C2", r["sid"])))
    res = {"experiment": "E-023", "date": time.strftime("%Y-%m-%d"), "seed": SEED, "n_boot": NB, "score_failures": fails}
    specs = {"orig_vs_orig": (real_ori, fk_ori), "B_canon_vs_canon": (real_can, fk_can), "P1_matchedM_vs_canon": (real_c1, fk_can), "S1_matchedL_vs_canon": (real_c1l, fk_can),
             "R1_canon_vs_fakes_x2": (real_can, fk_c2), "both_mpeg4_M": (real_c1, fk_c2)}
    diffs = {"P1_minus_B": ("P1_matchedM_vs_canon", "B_canon_vs_canon"), "S1_minus_B": ("S1_matchedL_vs_canon", "B_canon_vs_canon"),
             "R1_minus_B": ("R1_canon_vs_fakes_x2", "B_canon_vs_canon"), "both_minus_B": ("both_mpeg4_M", "B_canon_vs_canon")}
    for m in sorted({r["method"] for r in fk}):
        sel = [r for r in fk if r["method"] == m]
        specs[f"B_method_{m}"] = (real_can, L(sel, lambda r: e21(rec, r["sid"])))
        specs[f"P1_method_{m}"] = (real_c1, L(sel, lambda r: e21(rec, r["sid"])))
        diffs[f"P1_minus_B_method_{m}"] = (f"P1_method_{m}", f"B_method_{m}")

    def H(label, n):
        return L(S[n], lambda r: sc.get((label, r["sid"])))
    for nm in ("orig", "can"):
        base = real_ori if nm == "orig" else real_can
        specs[f"AH_{nm}_vs_RVRA_{nm}"] = (base, H(f"H_{nm}", "H"))
        specs[f"AK_{nm}_vs_RVRA_{nm}"] = (base, H(f"K_{nm}", "K"))
        diffs[f"AH_minus_AK_{nm}"] = (f"AH_{nm}_vs_RVRA_{nm}", f"AK_{nm}_vs_RVRA_{nm}")
    specs["AH_can_vs_C1"] = (real_c1, H("H_can", "H"))
    specs["AK_can_vs_C1"] = (real_c1, H("K_can", "K"))
    diffs["AH_minus_AK_C1"] = ("AH_can_vs_C1", "AK_can_vs_C1")
    r = run_aucs(specs, diffs)
    res["results"] = r
    res["verdict_P1"] = verdict(r["P1_matchedM_vs_canon"])
    res["verdict_S1"] = verdict(r["S1_matchedL_vs_canon"])
    res["verdict_overall"] = res["verdict_P1"] if res["verdict_P1"] == res["verdict_S1"] else "INCONCLUSIVE (P1 and S1 in different bands)"

    def med(lst):
        return round(float(np.median([s for _, s in lst])), 4) if lst else None
    res["median_scores"] = {"RVRA_canon": med(real_can), "RVRA_M": med(real_c1), "RVRA_L": med(real_c1l), "fakes_canon": med(fk_can), "fakes_M_x2": med(fk_c2),
                            "H_orig": med(H("H_orig", "H")), "K_orig": med(H("K_orig", "K")), "H_can": med(H("H_can", "H")), "K_can": med(H("K_can", "K"))}
    with open(O / "prep.csv", encoding="utf-8") as f:
        pr = list(csv.DictReader(f))
    res["prep_failures"] = {n: sum(1 for x in pr if x["set"] == n and x["rc"] != "0") for n in ("C1", "C1L", "C2", "H", "K")}
    ach = {}
    for n in ("C1", "C1L", "C2"):
        v = [x["achieved_bitrate|codec"].split("|") for x in pr if x["set"] == n and x["achieved_bitrate|codec"]]
        ach[n] = {"n": len(v), "median_bitrate": float(np.median([float(a) for a, _ in v])) if v else None, "codec": sorted({c for _, c in v})}
    res["achieved"] = ach
    res["set_sizes"] = {k: len(v) for k, v in S.items()}
    res["group_counts"] = {"H": len({r["group"] for r in S["H"]}), "K": len({r["group"] for r in S["K"]})}
    (O / "e023_results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    {"prep": cmd_prep, "score": cmd_score, "analyse": cmd_analyse}[sys.argv[1]]()
