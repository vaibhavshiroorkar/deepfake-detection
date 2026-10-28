"""E-027 error analysis on the open-subset out-of-fold scores of the chosen candidate (LOFO, fold seed 0, threshold t_BA).

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_errors.py

Generated clips: false-negative rate per generator by scenario, speaking_style, camera_type (score from the run that held out the
generator's family). Real clips: false-positive rate by the same columns and by source kind (TalkingCelebs `tc:` or YouTube `yt:` group),
using each real clip's mean score over the three family runs. Counts only; no frame or clip is written anywhere.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

import e027_lib as L


def main() -> int:
    dev = json.loads((L.OUT / "dev_results.json").read_text(encoding="utf-8"))
    ch, t = dev["selection"]["chosen"], dev["thresholds"]["t_BA"]
    gu = L.gen_unit("LOFO")
    rows = list(csv.DictReader(open(L.OUT / "oof_scores.csv", encoding="utf-8")))
    for r in rows:
        if r["label"] == "1":
            r["s"] = float(r[f"{ch}_LOFO_seed0_unit_{gu[r['generator']]}"])
        else:
            r["s"] = float(np.mean([float(r[k]) for k in r if k.startswith(f"{ch}_LOFO_seed0_unit_")]))
        r["source_kind"] = "TalkingCelebs" if r["group"].startswith("tc:") else "YouTube (OpenVid / MAVOS-DD)"
    out = {"candidate": ch, "threshold_t_BA": t, "tables": {}}
    for col in ("scenario", "speaking_style", "camera_type", "source_kind"):
        tab = {}
        for g in L.OPEN_GENS + ["real"]:
            if col == "source_kind" and g != "real":
                continue
            vals = sorted({r[col] for r in rows})
            for v in vals:
                sel = [r for r in rows if r["generator"] == g and r[col] == v]
                if not sel:
                    continue
                k = sum((r["s"] < t) if g != "real" else (r["s"] >= t) for r in sel)
                tab[f"{g} | {v}"] = {"n": len(sel), "errors": int(k), "rate": round(k / len(sel), 4), "wilson95": L.wilson(int(k), len(sel)),
                                     "kind": "false-positive rate" if g == "real" else "false-negative rate"}
        out["tables"][col] = tab
    per_gen = {}
    for g in L.OPEN_GENS:
        sel = [r for r in rows if r["generator"] == g]
        k = sum(r["s"] < t for r in sel)
        per_gen[g] = {"n": len(sel), "fn": int(k), "fnr": round(k / len(sel), 4), "wilson95": L.wilson(int(k), len(sel))}
    sel = [r for r in rows if r["generator"] == "real"]
    k = sum(r["s"] >= t for r in sel)
    per_gen["real (false positives)"] = {"n": len(sel), "fp": int(k), "fpr": round(k / len(sel), 4), "wilson95": L.wilson(int(k), len(sel))}
    out["per_generator"] = per_gen
    L.jdump(out, L.OUT / "error_analysis.json")
    print(json.dumps(out["per_generator"], indent=1))
    for col, tab in out["tables"].items():
        print("==", col)
        for k, v in tab.items():
            print(f"  {k:55s} n={v['n']:4d} err={v['errors']:4d} rate={v['rate']:.3f} {v['wilson95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
