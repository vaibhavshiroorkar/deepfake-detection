"""E-027 source-confound re-check: the frozen deployable E-027 model on the E-026 real clips (FF++ val originals through the split guard,
LAV-DF official-train real clips; selection reused from scripts/genvideo/e026.py). Development roles only; DF26 is not read here.

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_confound.py

Writes data/_results/modern/e027/confound_scores.csv and confound_results.json (refuses to overwrite). Changes no model or threshold.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import e027_lib as L


def load_e026():
    spec = importlib.util.spec_from_file_location("e026", L.ROOT / "scripts" / "genvideo" / "e026.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if (L.OUT / "confound_results.json").exists():
        print("confound_results.json exists: refusing to overwrite")
        return 2
    fz = json.loads((L.CKPT / "freeze.json").read_text(encoding="utf-8"))
    for name, h in fz["checkpoint_sha256"].items():
        assert L.sha256(L.CKPT / name) == h, f"{name} changed since freeze"
    dev = json.loads((L.OUT / "dev_results.json").read_text(encoding="utf-8"))
    ch, cid = fz["chosen"], fz["encoder"]
    view = ch.split("-", 1)[0]
    from dfd.platform.genvideo import canonical
    from dfd.platform.genvideo.model import load_head
    from e027_features import Enc
    head = load_head(L.CKPT / f"deploy_{ch}.npz")
    t_ba, t_10 = fz["thresholds"]["t_BA"], fz["thresholds"]["t_10"]
    enc = Enc(cid)
    E = load_e026()
    ff = E.clips_ffpp()
    lav, n_elig = E.clips_lavdf()
    rows = []
    t0 = time.time()
    for source, clips in (("R-FF", ff), ("R-LAV", lav)):
        for sid, path in clips:
            row = {"source": source, "sample_id": sid, "path": str(path.relative_to(L.ROOT)).replace("\\", "/"), "score": "", "error": ""}
            try:
                with tempfile.TemporaryDirectory() as td:
                    dst = Path(td) / "c.mp4"
                    canonical.canonicalise(path, dst)
                    fr = canonical.read_frames(dst)
                e = enc(fr)[None]
                row["score"] = round(float(head.score(L.view(e, view))[0]), 6)
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
            rows.append(row)
    with open(L.OUT / "confound_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # DF26 real reference: the chosen candidate's out-of-fold real scores (fold seed 0, mean over the three family runs)
    ref = []
    with open(L.OUT / "oof_scores.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["label"] == "0":
                ref.append(np.mean([float(r[k]) for k in r if k.startswith(f"{ch}_LOFO_seed0_unit_")]))
    ref = np.array(ref)
    rng = np.random.default_rng(L.BOOT_SEED)
    from dfd.eval.metrics import roc_auc

    def auc_vs(o, r):
        a = lambda o, r: float(roc_auc(np.r_[np.ones(len(o)), np.zeros(len(r))], np.r_[o, r]))  # noqa: E731
        reps = [a(rng.choice(o, len(o)), rng.choice(r, len(r))) for _ in range(L.N_BOOT)]
        return {"auc": round(a(o, r), 4), "ci95": L.ci(reps)}

    def desc(s, thr):
        k, n = int((s >= thr).sum()), len(s)
        return {"n": n, "flagged": k, "fpr": round(k / n, 4), "wilson95": L.wilson(k, n)}
    res = {"experiment": "E-027", "stage": "source-confound re-check", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "model": f"deploy_{ch}.npz",
           "model_sha256": fz["checkpoint_sha256"][f"deploy_{ch}.npz"], "t_BA": t_ba, "t_10": t_10, "n_lavdf_eligible": n_elig,
           "seconds": round(time.time() - t0, 1), "sources": {}}
    ref_fpr = dev["thresholds"]["oof_real_fpr_at_t_BA"]
    res["df26_real_oof_fpr_at_t_BA"] = ref_fpr
    res["df26_real_oof_mean_score_fpr_at_t_BA"] = desc(ref, t_ba)
    pooled = []
    for src in ("R-FF", "R-LAV"):
        ok = np.array([r["score"] for r in rows if r["source"] == src and r["error"] == ""], dtype=float)
        nbad = sum(1 for r in rows if r["source"] == src and r["error"])
        d = {"failures": nbad, "at_t_BA": desc(ok, t_ba), "at_t_10": desc(ok, t_10),
             "quantiles": {q: round(float(np.percentile(ok, q)), 4) for q in (5, 25, 50, 75, 95)}, "auc_vs_df26_real_oof": auc_vs(ok, ref)}
        res["sources"][src] = d
        pooled.append(ok)
    P = np.concatenate(pooled)
    pd_ = {"at_t_BA": desc(P, t_ba), "at_t_10": desc(P, t_10), "auc_vs_df26_real_oof": auc_vs(P, ref),
           "quantiles": {q: round(float(np.percentile(P, q)), 4) for q in (5, 25, 50, 75, 95)}}
    res["sources"]["pooled"] = pd_
    lo, hi = pd_["at_t_BA"]["wilson95"]
    res["verdict"] = ("CONFOUND PERSISTS" if (lo > ref_fpr and lo > 0.5) else "NO EXCESS" if hi <= ref_fpr else "PARTIAL")
    res["rule"] = "CONFOUND PERSISTS if pooled Wilson lower bound > DF26 real OOF FPR at t_BA and > 0.5; NO EXCESS if pooled upper bound <= that rate; else PARTIAL"
    L.jdump(res, L.OUT / "confound_results.json")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
