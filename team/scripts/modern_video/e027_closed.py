"""E-027 ONE counted look at DF26's closed-generator subset with the frozen candidate (docs/EXPERIMENTS.md E-027).

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_closed.py --final

Refuses without --final, if closed_look.json exists (the marker is written BEFORE any closed clip is read, so a crash cannot allow a
second look), or if any frozen checkpoint differs from freeze.json. Nothing is fitted, calibrated, thresholded or selected here: closed
clips are scored by the cross-fitted fold model of their paired real's fold, real clips by their own fold model; t_BA, t_10 and the Platt
calibrator come from freeze.json (open out-of-fold scores only). Ledger/** is protected, so no ledger entry is written.
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

import e027_lib as L

PRIMARY_SCEN = ("Direct_to_Camera_Casual", "Studio_Interview")


def main(argv) -> int:
    if "--final" not in argv:
        print("refused: the closed subset needs --final")
        return 2
    marker = L.OUT / "closed_look.json"
    if marker.exists():
        print("refused: the counted closed look has already been used (closed_look.json)")
        return 2
    fz = json.loads((L.CKPT / "freeze.json").read_text(encoding="utf-8"))
    for name, h in fz["checkpoint_sha256"].items():
        assert L.sha256(L.CKPT / name) == h, f"{name} changed since freeze"
    for c, h in fz["feature_table_sha256"].items():
        assert L.sha256(L.OUT / f"features_{c}_dev.npz") == h
    prereg = json.loads((L.OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["e027_block_sha256"] == fz["prereg_sha256"]
    L.jdump({"experiment": "E-027", "look": "closed subset, counted look 2 (E-020 was look 1)", "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
             "candidate": fz["chosen"], "checkpoints": fz["checkpoint_sha256"]}, marker)

    ch, cid = fz["chosen"], fz["encoder"]
    view = ch.split("-", 1)[0]
    t_ba, t_10 = fz["thresholds"]["t_BA"], fz["thresholds"]["t_10"]
    cal = fz["calibrator"]
    fmap = fz["fold_seed0_map"]
    from dfd.platform.genvideo.model import load_head
    fold_heads = {f: load_head(L.CKPT / f"fold{f}_{ch}.npz") for f in range(L.FOLDS)}
    fold_c3 = {f: load_head(L.CKPT / f"fold{f}_C3.npz") for f in range(L.FOLDS)}
    assert L.sha256(L.E020_HEAD) == L.E020_HEAD_SHA
    e020 = load_head(L.E020_HEAD)

    reals = [r for r in L.dev_rows() if r["y"] == 0]
    closed = L.closed_rows_FINAL_ONLY()
    rows = reals + closed
    z = np.load(L.E020_FEAT, allow_pickle=False)
    pos = {k: i for i, k in enumerate(z["id"])}
    take = np.array([pos[r["video_path"]] for r in rows])
    stats, clsm = z["stats"][take].astype(np.float64), z["cls_mean"][take].astype(np.float64)
    if cid == "IMG-07" and view == "P8":
        X = clsm
        extract_info = {"source": "stored E-020 fp32 DINOv2-small features (features_df26.npz)"}
    else:
        from e027_features import extract
        fd = np.load(L.OUT / f"features_{cid}_dev.npz", allow_pickle=False)
        p2 = {k: i for i, k in enumerate(fd["id"])}
        Er = fd["emb"][[p2[r["video_path"]] for r in reals]]
        Ec, keep, bad, info = extract(closed, cid)
        # fix 2026-10-27 (before the closed look): preregistered exclusion rule instead of an assertion, so a failing clip is listed and
        # excluded (a generator with more than 2 percent failures is reported as failed) and cannot crash the counted look midway
        np.savez_compressed(L.OUT / f"features_{cid}_closed.npz", emb=Ec, id=np.array([r["video_path"] for r in keep]))
        keep_ids = {r["video_path"] for r in keep}
        closed = [r for r in closed if r["video_path"] in keep_ids]
        rows = reals + closed
        take = np.array([pos[r["video_path"]] for r in rows])
        stats, clsm = z["stats"][take].astype(np.float64), z["cls_mean"][take].astype(np.float64)
        X = L.view(np.concatenate([Er, Ec]).astype(np.float64), view)
        n_bad = {g: sum(1 for b in bad if f"/{g}/" in b["id"]) for g in L.CLOSED_GENS}
        extract_info = {"source": f"extracted now with {cid}", **info, "failures": bad, "failures_per_generator": n_bad}
    y = np.array([r["y"] for r in rows])
    gen = np.array([r["generator"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    scen = np.array([r["scenario"] for r in rows])
    fold = np.array([fmap[g] for g in grp])
    s = np.empty(len(y)); s3 = np.empty(len(y))
    for f in range(L.FOLDS):
        m = fold == f
        s[m] = fold_heads[f].score(X[m]); s3[m] = fold_c3[f].score(stats[m])
    se = e020.score(clsm)
    pc = np.clip(s, 1e-6, 1 - 1e-6)
    scal = 1 / (1 + np.exp(-(cal["a"] * np.log(pc / (1 - pc)) + cal["b"])))

    from dfd.eval.bootstrap import cluster_bootstrap, paired_bootstrap, ClusterResampler
    from dfd.eval.metrics import eer, ece
    from sklearn.metrics import average_precision_score

    def auc_block(mask):
        b = cluster_bootstrap(y[mask], s[mask], grp[mask], metric="auc", seed=L.BOOT_SEED)
        return {"auc": round(b.estimate, 4), "ci95": [round(b.low, 4), round(b.high, 4)], "n": int(mask.sum()), "n_fake": int(y[mask].sum())}

    prim = (y == 1) | np.isin(scen, PRIMARY_SCEN)
    res = {"experiment": "E-027", "stage": "closed look", "candidate": ch, "encoder": cid, "features": extract_info,
           "n_closed": int(len(closed)), "n_real_primary": int(((y == 0) & prim).sum()), "n_real_all": int((y == 0).sum())}
    res["primary_auc_vs_200_scenario_reals"] = auc_block(prim)
    lo, hi = res["primary_auc_vs_200_scenario_reals"]["ci95"]
    res["H_X3c"] = "SUPPORTED" if lo > 0.70 else ("NOT SUPPORTED" if hi < 0.60 else "INCONCLUSIVE")
    res["auc_vs_all_271_reals"] = auc_block(np.ones(len(y), bool))
    res["per_generator"] = {}
    for g in L.CLOSED_GENS:
        m = prim & ((gen == g) | (y == 0))
        d = auc_block(m)
        d["eer"] = round(float(eer(y[m], s[m])), 4)
        nb = extract_info.get("failures_per_generator", {}).get(g, 0)
        d["feature_failures"] = nb
        d["reported_as_failed"] = bool(nb > 0.02 * (nb + int((gen == g).sum())))
        res["per_generator"][g] = d
    rs = ClusterResampler(y[prim], grp[prim], L.BOOT_SEED)
    ap_pt = float(average_precision_score(y[prim], s[prim]))
    aps = [float(average_precision_score(y[prim], s[prim], sample_weight=w)) for w in rs.weights(L.N_BOOT)]
    res["pr_auc_average_precision"] = {"estimate": round(ap_pt, 4), "ci95": L.ci(aps), "prevalence": round(float(y[prim].mean()), 4)}
    res["at_threshold"] = {}
    for tn, t in (("t_BA", t_ba), ("t_10", t_10)):
        yp, sp, gp = y[prim], s[prim], gen[prim]
        tp = int(((sp >= t) & (yp == 1)).sum()); fn = int(((sp < t) & (yp == 1)).sum())
        fp = int(((sp >= t) & (yp == 0)).sum()); tn_ = int(((sp < t) & (yp == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn)
        d = {"threshold": t, "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn_},
             "fpr_reals": {"rate": round(fp / (fp + tn_), 4), "wilson95": L.wilson(fp, fp + tn_)},
             "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
             "fnr_per_generator": {}}
        for g in L.CLOSED_GENS:
            mg = gp == g
            k = int((sp[mg] < t).sum())
            d["fnr_per_generator"][g] = {"rate": round(k / mg.sum(), 4), "wilson95": L.wilson(k, int(mg.sum())), "n": int(mg.sum())}
        allr = y == 0
        k = int((s[allr] >= t).sum())
        d["fpr_all_271_reals"] = {"rate": round(k / allr.sum(), 4), "wilson95": L.wilson(k, int(allr.sum()))}
        res["at_threshold"][tn] = d
    bc = cluster_bootstrap(y[prim], scal[prim], grp[prim], metric="ece", seed=L.BOOT_SEED)
    br = cluster_bootstrap(y[prim], s[prim], grp[prim], metric="ece", seed=L.BOOT_SEED)
    res["ece"] = {"calibrated": [round(bc.estimate, 4), round(bc.low, 4), round(bc.high, 4)], "raw": [round(br.estimate, 4), round(br.low, 4), round(br.high, 4)],
                  "note": "prevalence 794 generated to 200 real"}
    res["paired"] = {}
    for nm, other in (("minus_C3_crossfit", s3), ("minus_E020_head", se)):
        d, a, b = paired_bootstrap(y[prim], s[prim], other[prim], grp[prim], metric="auc", seed=L.BOOT_SEED)
        res["paired"][nm] = {"estimate": round(d.estimate, 4), "ci95": [round(d.low, 4), round(d.high, 4)], "other_auc": round(b.estimate, 4),
                             "other_ci95": [round(b.low, 4), round(b.high, 4)]}
    for nm, other in (("C3_crossfit", s3), ("E020_head", se)):
        res["per_generator_" + nm] = {}
        for g in L.CLOSED_GENS:
            m = prim & ((gen == g) | (y == 0))
            b = cluster_bootstrap(y[m], other[m], grp[m], metric="auc", seed=L.BOOT_SEED)
            res["per_generator_" + nm][g] = {"auc": round(b.estimate, 4), "ci95": [round(b.low, 4), round(b.high, 4)]}
    # error analysis (closed), at t_BA
    ea = {}
    for col in ("scenario", "speaking_style", "camera_type"):
        v = np.array([r[col] for r in rows])
        tab = {}
        for g in L.CLOSED_GENS + ["real"]:
            for val in sorted(set(v)):
                m = (gen == g) & (v == val) & prim
                if m.sum() == 0:
                    continue
                k = int(((s[m] < t_ba) if g != "real" else (s[m] >= t_ba)).sum())
                tab[f"{g} | {val}"] = {"n": int(m.sum()), "errors": k, "rate": round(k / m.sum(), 4), "wilson95": L.wilson(k, int(m.sum())),
                                       "kind": "false-positive rate" if g == "real" else "false-negative rate"}
        ea[col] = tab
    res["error_analysis_t_BA"] = ea
    with open(L.OUT / "closed_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "generator", "label", "group", "fold", "scenario", "speaking_style", "camera_type", "score", "score_platt", "C3", "E020"])
        for i, r in enumerate(rows):
            w.writerow([r["video_path"], r["generator"], r["y"], r["group"], fold[i], r["scenario"], r["speaking_style"], r["camera_type"],
                        round(float(s[i]), 6), round(float(scal[i]), 6), round(float(s3[i]), 6), round(float(se[i]), 6)])
    res["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    L.jdump(res, L.OUT / "closed_results.json")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
