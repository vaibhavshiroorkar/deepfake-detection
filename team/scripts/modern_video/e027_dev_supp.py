"""E-027 supplementary development record (written 2026-10-27 after e027_dev.py, before the closed look; reads the open subset only).

    PYTHONPATH="src;scripts/modern_video" .venv/Scripts/python.exe scripts/modern_video/e027_dev_supp.py

e027_dev.py applied the preregistered simplicity rule but did not store the paired differences it tested. This script recomputes the
deterministic out-of-fold scores of the selectable candidates (same code, same folds, same bootstrap resamples), checks that every macro
AUC equals dev_results.json, and writes data/_results/modern/e027/dev_supp.json with: best minus every cheaper eligible candidate (the
simplicity check), LOGO minus LOFO for every selectable candidate, and the per-generator LOFO AUC of the chosen candidate. Nothing is
selected, fitted for use, or frozen here.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import e027_lib as L
from e027_dev import SELECTABLE, diff, load_dev, summarise


def main() -> int:
    if (L.OUT / "dev_supp.json").exists():
        print("dev_supp.json exists: refusing to overwrite")
        return 2
    dev = json.loads((L.OUT / "dev_results.json").read_text(encoding="utf-8"))
    rows, ids, y, gen, grp, stored, emb = load_dev()
    mb = L.MacroBoot(y, gen, grp)
    M, check = {}, {}
    for cid, v in SELECTABLE:
        n = f"{v}-{cid}"
        X = stored["cls_mean"] if (cid == "IMG-07" and v == "P8") else L.view(emb[cid], v)
        for p in ("LOFO", "LOGO"):
            m, d = summarise(mb, [L.oof_scores(X, y, gen, grp, p, s) for s in L.SEEDS], p)
            M[(n, p)] = m
            check[f"{n} {p}"] = [d["macro_auc"], dev["candidates"][n][p]["macro_auc"], d["ci95"] == dev["candidates"][n][p]["ci95"]]
    same = all(a == b and c for a, b, c in check.values())
    sel = dev["selection"]
    best, order = sel["best_point"], [f"{v}-{cid}" for cid, v in SELECTABLE]
    simp = {f"{best} - {n}": diff(M[(best, "LOFO")], M[(n, "LOFO")]) for n in order
            if n in sel["eligible"] and order.index(n) < order.index(best)}
    out = {"experiment": "E-027", "stage": "development supplement", "reproduces_dev_results": same, "check": check,
           "simplicity_check_LOFO": simp,
           "LOGO_minus_LOFO": {n: diff(M[(n, "LOGO")], M[(n, "LOFO")]) for n in order}}
    L.jdump(out, L.OUT / "dev_supp.json")
    print(json.dumps(out, indent=1))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
