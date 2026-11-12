"""E-013b: Xception baseline at learning rate 3e-4 (grid extension, D-062). Recipe of E-013 unchanged.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline/run_baseline_b.py

Stage 1: seed 0 at 3e-4. Selected (validation AUC strictly above every E-013 rate) -> seeds 1 and 2 at 3e-4.
Writes data/_results/baseline/selection_b.json (best_lr over four rates). selection.json of E-013 is not touched. Resumable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dfd.train import finetune as FT
from run_baseline import OUT, log, safe

NEW = 3e-4


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    val = dict(sel["val_auc"])
    r = safe(NEW, 0)
    v = r.get("val_auc") if r.get("status") == "ok" else None
    val[f"{NEW:g}"] = v
    good = {k: x for k, x in val.items() if x is not None and x == x}
    best = float(max(good, key=lambda k: good[k]))
    log(f"seed 0 validation AUC by rate: {val}; selected {best:g}")
    (OUT / "selection_b.json").write_text(json.dumps(dict(best_lr=best, val_auc=val, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    if best == NEW:
        for s in (1, 2):
            safe(NEW, s)
    (OUT / "DONE_b").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("finished")


if __name__ == "__main__":
    main()
