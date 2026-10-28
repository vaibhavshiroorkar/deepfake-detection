"""E-027 development protocol on DF26's open subset (docs/EXPERIMENTS.md E-027, preregistered; prereg.json must exist).

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_dev.py

Reads only the development pool (271 real clips, six open generators); the closed rows of the stored E-020 table are dropped at load
time. Leave-one-family-out (primary) and leave-one-generator-out (secondary), fold partitions over source groups with seeds 0, 1, 2;
macro AUC with a shared cluster bootstrap; preregistered selection, thresholds, calibration (X6); freezes the chosen candidate's fold
models, C3 fold models and the deployable model into checkpoints/modern-E027/ and writes freeze.json. Refuses to overwrite results.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import e027_lib as L

SELECTABLE = [(e, v) for e in L.COST for v in ("P8", "PT14")]
DIAG_VIEWS = ("P14", "PT14s", "D14")


def load_dev():
    rows = L.dev_rows()
    ids = np.array([r["video_path"] for r in rows])
    y = np.array([r["y"] for r in rows])
    gen = np.array([r["generator"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    z = np.load(L.E020_FEAT, allow_pickle=False)
    zid = list(z["id"])
    pos = {k: i for i, k in enumerate(zid)}
    take = np.array([pos[i] for i in ids])
    assert not np.any(z["access"][take] == "closed"), "closed row in development pool"
    stored = {k: z[k][take].astype(np.float64) for k in ("stats", "restrav21", "cls_mean")}
    assert np.array_equal(z["label"][take].astype(int), y)
    emb = {}
    for cid in L.COST:
        f = np.load(L.OUT / f"features_{cid}_dev.npz", allow_pickle=False)
        p2 = {k: i for i, k in enumerate(f["id"])}
        missing = [i for i in ids if i not in p2]
        assert not missing, f"{cid}: {len(missing)} clips without features"
        emb[cid] = f["emb"][[p2[i] for i in ids]].astype(np.float64)
    return rows, ids, y, gen, grp, stored, emb


def job(name, X, y, gen, grp, protocol, seed, kind):
    return name, protocol, seed, L.oof_scores(X, y, gen, grp, protocol, seed, kind)


def summarise(mb: L.MacroBoot, S_by_seed, protocol):
    ev = mb.per_gen_evals(S_by_seed, protocol)
    m = mb.macro(ev)
    return m, {"macro_auc": round(m["point"], 4), "ci95": L.ci(m["reps"]), "seed_values": [round(v, 4) for v in m["per_seed"]],
               "seed_sd": None if m["seed_sd"] is None else round(m["seed_sd"], 4),
               "per_generator": {g: {"auc": round(m["per_gen"][g], 4), "ci95": L.ci(m["reps_gen"][g])} for g in L.OPEN_GENS}}


def diff(a, b):
    return {"estimate": round(a["point"] - b["point"], 4), "ci95": L.ci(a["reps"] - b["reps"])}


def main() -> int:
    if not (L.OUT / "prereg.json").exists():
        print("prereg.json missing: refused")
        return 2
    if (L.OUT / "dev_results.json").exists():
        print("dev_results.json exists: refusing to overwrite")
        return 2
    t0 = time.time()
    rows, ids, y, gen, grp, stored, emb = load_dev()
    # sanity: the live-path 14-frame IMG-07 embeddings reproduce the stored E-020 cls_mean on the 8 PICK frames
    p8_gap = float(np.abs(L.view(emb["IMG-07"], "P8") - stored["cls_mean"]).max())
    folds = {s: L.fold_of_groups(grp, s) for s in L.SEEDS}
    with open(L.OUT / "groups_folds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "generator", "label", "group"] + [f"fold_seed{s}" for s in L.SEEDS])
        for r in rows:
            w.writerow([r["video_path"], r["generator"], r["y"], r["group"]] + [folds[s][r["group"]] for s in L.SEEDS])
    # leakage check: no group on both sides of any fold (by construction; verified)
    for s in L.SEEDS:
        fg = {}
        for g in grp:
            fg.setdefault(g, set()).add(folds[s][g])
        assert all(len(v) == 1 for v in fg.values())

    feats = {}
    for cid in L.COST:
        for v in ("P8", "PT14", "P14", "PT14s", "D14"):
            X = stored["cls_mean"] if (cid == "IMG-07" and v == "P8") else L.view(emb[cid], v)
            feats[f"{v}-{cid}"] = (X, "LR")
    feats["TR21"] = (stored["restrav21"], "MLP")
    feats["C3"] = (stored["stats"], "LR")
    tasks = [(n, X, k, p, s) for n, (X, k) in feats.items() for p in ("LOFO", "LOGO") for s in L.SEEDS]
    print(f"{len(tasks)} fit jobs, groups {len(set(grp))}, P8 gap {p8_gap:.2e}", flush=True)
    res = Parallel(n_jobs=10, verbose=5)(delayed(job)(n, X, y, gen, grp, p, s, k) for n, X, k, p, s in tasks)
    S = {}
    for n, p, s, sc in res:
        S.setdefault(n, {}).setdefault(p, {})[s] = sc
    print(f"fits done {time.time() - t0:.0f} s", flush=True)

    from dfd.platform.genvideo.model import load_head
    assert L.sha256(L.E020_HEAD) == L.E020_HEAD_SHA
    e020 = load_head(L.E020_HEAD).score(stored["cls_mean"])

    mb = L.MacroBoot(y, gen, grp)
    M, out = {}, {"experiment": "E-027", "stage": "development", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                  "n_clips": int(len(y)), "n_real": int((y == 0).sum()), "n_groups": int(len(set(grp))),
                  "n_real_groups": int(len(set(grp[y == 0]))), "p8_img07_live_vs_stored_max_abs_diff": p8_gap, "candidates": {}}
    for n in feats:
        out["candidates"][n] = {}
        for p in ("LOFO", "LOGO"):
            m, d = summarise(mb, [S[n][p][s] for s in L.SEEDS], p)
            M[(n, p)] = m
            out["candidates"][n][p] = d
    ev = mb.per_gen_evals([e020], None)
    M[("E020", "none")] = me = mb.macro(ev)
    out["candidates"]["E020 (installed head, not refitted)"] = {"none": {"macro_auc": round(me["point"], 4), "ci95": L.ci(me["reps"]),
        "per_generator": {g: {"auc": round(me["per_gen"][g], 4), "ci95": L.ci(me["reps_gen"][g])} for g in L.OPEN_GENS}}}

    # ---- selection (preregistered), LOFO
    elig = []
    sel = {}
    for cid, v in SELECTABLE:
        n = f"{v}-{cid}"
        d3 = diff(M[(n, "LOFO")], M[("C3", "LOFO")])
        lo = out["candidates"][n]["LOFO"]["ci95"][0]
        ok = lo > 0.60 and d3["ci95"][0] > 0
        sel[n] = {"lofo_lower": lo, "minus_C3": d3, "eligible": ok}
        if ok:
            elig.append(n)
    order = [f"{v}-{cid}" for cid, v in SELECTABLE]                      # cost order
    if elig:
        best = max(elig, key=lambda n: M[(n, "LOFO")]["point"])
        chosen = best
        for n in order:
            if n in elig and order.index(n) < order.index(best):
                d = diff(M[(best, "LOFO")], M[(n, "LOFO")])
                if d["ci95"][0] <= 0 <= d["ci95"][1]:
                    chosen = n
                    break
        verdict = "SUPPORTED"
    else:
        best = chosen = max(order, key=lambda n: M[(n, "LOFO")]["point"])
        verdict = "NOT SUPPORTED"
    out["selection"] = {"eligibility": sel, "eligible": elig, "best_point": best, "chosen": chosen, "H_X3": verdict,
                        "chosen_label": "eligible" if elig else "not eligible (highest point estimate, frozen for the counted closed look)"}
    ch = chosen
    chosen_enc = ch.split("-", 1)[1]
    out["comparisons_LOFO"] = {
        f"{ch} - C3": diff(M[(ch, "LOFO")], M[("C3", "LOFO")]),
        f"{ch} - E020": diff(M[(ch, "LOFO")], M[("E020", "none")]),
        f"{ch} - TR21": diff(M[(ch, "LOFO")], M[("TR21", "LOFO")]),
        f"{best} - {ch}": diff(M[(best, "LOFO")], M[(ch, "LOFO")]),
        "C3 - E020": diff(M[("C3", "LOFO")], M[("E020", "none")]),
    }
    for cid in L.COST:
        out["comparisons_LOFO"][f"P8-{cid} - P8-IMG-07"] = diff(M[(f"P8-{cid}", "LOFO")], M[("P8-IMG-07", "LOFO")])
    out["temporal_LOFO"] = {}
    for cid in L.COST:
        a = {k: M[(f"{k}-{cid}", "LOFO")] for k in ("P8", "P14", "PT14", "PT14s", "D14")}
        t = {"PT14 - P14": diff(a["PT14"], a["P14"]), "PT14 - PT14s": diff(a["PT14"], a["PT14s"]), "P14 - P8": diff(a["P14"], a["P8"]),
             "D14 - C3": diff(a["D14"], M[("C3", "LOFO")]), "D14 macro": out["candidates"][f"D14-{cid}"]["LOFO"]["macro_auc"],
             "D14 ci95": out["candidates"][f"D14-{cid}"]["LOFO"]["ci95"]}
        lo1 = t["PT14 - P14"]["ci95"][0]
        lo2 = t["PT14 - PT14s"]["ci95"][0]
        t["H_T"] = ("uses temporal change and frame order" if (lo1 > 0 and lo2 > 0) else "uses temporal change (not shown to need frame order)"
                    if lo1 > 0 else "no evidence that temporal features add to pooled appearance")
        out["temporal_LOFO"][cid] = t
    out["LOGO_minus_LOFO"] = {n: diff(M[(n, "LOGO")], M[(n, "LOFO")]) for n in [ch, "C3"]}

    # ---- thresholds (fold seed 0, LOFO, open out-of-fold only)
    gu = L.gen_unit("LOFO")
    S0 = S[ch]["LOFO"][0]
    sc, yy = [], []
    for g in L.OPEN_GENS:
        m = (gen == g) | (y == 0)
        sc.append(S0[gu[g]][m]); yy.append(y[m])
    sc, yy = np.concatenate(sc), np.concatenate(yy)
    t_ba, ba = L.ba_threshold(sc, yy)
    real_pool = np.concatenate([S0[u][y == 0] for u in L.units("LOFO")])
    t_10 = float(np.percentile(real_pool, 90))
    out["thresholds"] = {"t_BA": t_ba, "balanced_accuracy_oof": round(ba, 4), "t_10": t_10,
                         "oof_real_fpr_at_t_BA": round(float((real_pool >= t_ba).mean()), 4),
                         "oof_fake_tpr_at_t_BA": round(float((sc[yy == 1] >= t_ba).mean()), 4),
                         "oof_real_fpr_at_t_10": round(float((real_pool >= t_10).mean()), 4),
                         "oof_fake_tpr_at_t_10": round(float((sc[yy == 1] >= t_10).mean()), 4)}

    # ---- X6 calibration: Platt on logit(score), leave-one-family-out over the OOF sets
    from sklearn.linear_model import LogisticRegression
    from dfd.eval.bootstrap import cluster_bootstrap

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    fam_sets = {}
    for u, gl in L.units("LOFO").items():
        m = np.isin(gen, gl) | (y == 0)
        fam_sets[u] = (S0[u][m], y[m], grp[m])
    raw_p, cal_p, yy2, gg2 = [], [], [], []
    per_fam = {}
    for u in fam_sets:
        trs = [fam_sets[v] for v in fam_sets if v != u]
        xs = np.concatenate([a[0] for a in trs]); ys = np.concatenate([a[1] for a in trs])
        c = LogisticRegression(C=1e6, max_iter=1000).fit(logit(xs)[:, None], ys)
        s_u, y_u, g_u = fam_sets[u]
        pc = c.predict_proba(logit(s_u)[:, None])[:, 1]
        from dfd.eval.metrics import ece
        per_fam[u] = {"ece_raw": round(float(ece(y_u, s_u)), 4), "ece_platt_lofo": round(float(ece(y_u, pc)), 4)}
        raw_p.append(s_u); cal_p.append(pc); yy2.append(y_u); gg2.append(g_u)
    raw_p, cal_p, yy2, gg2 = map(np.concatenate, (raw_p, cal_p, yy2, gg2))
    b_raw = cluster_bootstrap(yy2, raw_p, gg2, metric="ece", seed=L.BOOT_SEED)
    b_cal = cluster_bootstrap(yy2, cal_p, gg2, metric="ece", seed=L.BOOT_SEED)
    # fix 2026-10-27 (before the first run): the final calibrator is fitted on ALL out-of-fold scores, i.e. the union of the three
    # family sets used in the assessment (each generated clip once, each real once per family run), not on the per-generator pool `sc`
    # used for t_BA, which counts every real twice per family and changes the class ratio the assessed calibrator was fitted at.
    cfin = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw_p)[:, None], yy2)
    out["calibration_X6"] = {"method": "Platt scaling on logit(score); leave-one-family-out assessment; 15-bin ECE; pooled sets keep each family's "
                                       "2:1 generated-to-real ratio",
                             "per_family": per_fam, "pooled_ece_raw": [round(b_raw.estimate, 4), round(b_raw.low, 4), round(b_raw.high, 4)],
                             "pooled_ece_platt": [round(b_cal.estimate, 4), round(b_cal.low, 4), round(b_cal.high, 4)],
                             "final_calibrator": {"a": float(cfin.coef_[0, 0]), "b": float(cfin.intercept_[0]),
                                                  "fitted_on": "all open LOFO out-of-fold scores, fold seed 0: union of the three family sets (each generated clip once, each real clip once per family run)"}}

    # ---- per-clip OOF table (LOFO seed 0 and seed-mean, chosen, C3, E020)
    with open(L.OUT / "oof_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        hdr = ["video_path", "generator", "label", "group", "scenario", "speaking_style", "camera_type", "fold_seed0"]
        hdr += [f"{ch}_LOFO_seed0_unit_{u}" for u in L.units("LOFO")] + [f"{ch}_LOFO_seedmean_own_unit", "C3_LOFO_seed0_own_unit", "E020_head"]
        w.writerow(hdr)
        for i, r in enumerate(rows):
            own = [S0[u][i] for u in L.units("LOFO")]
            if r["y"] == 1:
                u = gu[r["generator"]]
                sm = np.mean([S[ch]["LOFO"][s][u][i] for s in L.SEEDS]); c3 = S["C3"]["LOFO"][0][u][i]
            else:
                sm = np.mean([S[ch]["LOFO"][s][u][i] for s in L.SEEDS for u in L.units("LOFO")])
                c3 = np.mean([S["C3"]["LOFO"][0][u][i] for u in L.units("LOFO")])
            w.writerow([r["video_path"], r["generator"], r["y"], r["group"], r["scenario"], r["speaking_style"], r["camera_type"],
                        folds[0][r["group"]]] + [round(float(v), 6) for v in own] + [round(float(sm), 6), round(float(c3), 6), round(float(e020[i]), 6)])

    # ---- freeze (before any closed clip is read)
    from dfd.platform.genvideo.model import save_head
    L.CKPT.mkdir(parents=True, exist_ok=True)
    fmap0 = folds[0]
    fold0 = np.array([fmap0[g] for g in grp])
    Xc, kc = feats[ch]
    X3, _ = feats["C3"]
    common = {"experiment": "E-027", "protocol": "canonical-v3", "frames": "8 PICK frames" if ch.startswith("P8") else "14 frames",
              "feature": ch.split("-")[0], "encoder": chosen_enc, "encoder_name": L.ENCODERS[chosen_enc],
              "trained_on": "DF26 open-generator subset (six generators, 1,626 clips) and DF26 real clips (271); closed subset never used",
              "head": "standardise + logistic regression C=1 balanced", "t_BA": t_ba, "t_10": t_10,
              "platt": out["calibration_X6"]["final_calibrator"], "selection": out["selection"]["chosen_label"],
              "status": "EXPERIMENTAL, not validated; not installed", "fold_seed": 0}
    files = {}
    for f in range(L.FOLDS):
        tr = fold0 != f
        for nm, X in ((ch, Xc), ("C3", X3)):
            m = L.fit("LR", X[tr], y[tr], 0)
            p = L.CKPT / f"fold{f}_{nm}.npz"
            save_head(p, m[0], m[1], m[2], m[3], t_ba if nm == ch else 0.5,
                      {**common, "head_id": f"E027-{nm}-fold{f}", "role": f"cross-fit fold model {f} (trained on clips outside fold {f})",
                       "feature": nm.split("-")[0] if nm != "C3" else "stats7", "encoder": chosen_enc if nm != "C3" else "none"})
            files[p.name] = L.sha256(p)
    m = L.fit("LR", Xc, y, 0)
    p = L.CKPT / f"deploy_{ch}.npz"
    save_head(p, m[0], m[1], m[2], m[3], t_ba, {**common, "head_id": f"E027-{ch}-deploy", "role": "deployable model, all development clips"})
    files[p.name] = L.sha256(p)
    feat_hash = {c: L.sha256(L.OUT / f"features_{c}_dev.npz") for c in L.COST}
    L.jdump({"experiment": "E-027", "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "chosen": ch, "encoder": chosen_enc,
             "thresholds": out["thresholds"], "calibrator": out["calibration_X6"]["final_calibrator"], "checkpoint_sha256": files,
             "feature_table_sha256": feat_hash, "e020_features_sha256": L.sha256(L.E020_FEAT),
             "prereg_sha256": json.loads((L.OUT / "prereg.json").read_text())["e027_block_sha256"],
             "fold_seed0_map": {g: int(v) for g, v in fmap0.items()}}, L.CKPT / "freeze.json")
    (L.OUT / "freeze.json").write_text((L.CKPT / "freeze.json").read_text(encoding="utf-8"), encoding="utf-8")
    out["freeze"] = files
    out["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "seconds": round(time.time() - t0, 1)}
    L.jdump(out, L.OUT / "dev_results.json")
    print(json.dumps({k: out[k] for k in ("selection", "thresholds", "comparisons_LOFO", "temporal_LOFO", "calibration_X6")}, indent=1, default=str))
    for n in out["candidates"]:
        for p, d in out["candidates"][n].items():
            print(f"{n:22s} {p}: {d['macro_auc']} {d['ci95']} sd={d.get('seed_sd')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
