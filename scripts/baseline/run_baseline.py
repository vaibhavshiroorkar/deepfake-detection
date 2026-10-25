"""E-013: re-run published baseline (D-025): Xception (IMG-04) fine-tuned on the same FF++ frames with the E-005 recipe and budget.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline/run_baseline.py

Stage 1: three learning rates (1e-5, 3e-5, 1e-4), seed 0. Selection by validation AUC only. Stage 2: seeds 1 and 2 at the selected rate.
Resumable (finished runs are skipped). Output data/_results/baseline/; weights checkpoints/I-1/IMG-04_* (not in git).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from dfd.train import finetune as FT

OUT = Path("data/_results/baseline")
LRS = [1e-5, 3e-5, 1e-4]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def safe(lr, seed):
    try:
        return FT.run("IMG-04", lr, seed, OUT, workers=10)
    except Exception as e:   # recorded, not hidden
        log(f"FAILED IMG-04 {lr} {seed}: {type(e).__name__}: {e}")
        d = OUT / f"IMG-04_lr{lr:g}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        r = dict(name=d.name, cand="IMG-04", lr=lr, seed=seed, status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        (d / "result.json").write_text(json.dumps(r, indent=1))
        return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {lr: safe(lr, 0) for lr in LRS}
    ok = {lr: r for lr, r in res.items() if r.get("status") == "ok" and r.get("val_auc") == r.get("val_auc")}
    best = max(ok, key=lambda k: ok[k]["val_auc"])
    log(f"selected lr {best:g} (validation AUC {ok[best]['val_auc']:.4f})")
    (OUT / "selection.json").write_text(json.dumps(dict(best_lr=best, val_auc={f"{k:g}": v.get("val_auc") for k, v in res.items()}, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    for s in (1, 2):
        safe(best, s)
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("finished")


if __name__ == "__main__":
    main()
