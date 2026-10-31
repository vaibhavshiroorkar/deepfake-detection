"""E-033: score the frozen product checkpoint on the eligible LAV-DF test pool (protocol: docs/EXPERIMENTS.md E-033).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/score_e033.py score          # every pool clip, reads no label, resumable
    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/score_e033.py reproduce      # rescore 20 seeded clips, compare (D-044 tolerance 0.02)

The scoring loop is the one of scripts/e015/e015.py: the product adapter, one clip at a time. Rows already in predictions.csv are never rewritten.
Refuses to run without data/_results/posteval/E-033/prereg.json (the preregistration must exist first).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "_results" / "posteval" / "E-033"
PRED = OUT / "predictions.csv"
MANIFEST = DATA / "_manifests" / "posteval" / "e033_lavdf.manifest.csv"
CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
COLS = ["sample_id", "visual_available", "visual_score", "visual_n_frames", "audio_available", "audio_score", "audio_n_units", "warnings", "error", "ms"]


def pool():
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return [(r["sample_id"], r["path"]) for r in csv.DictReader(f)]   # path only; labels are not read here


def score_one(ad, path):
    from dfd.infer.library import inspect
    row = dict.fromkeys(COLS, "")
    t = time.time()
    try:
        res = ad.analyse(inspect(path, path.name))
        v, a = res.per_modality.get("visual"), res.per_modality.get("audio")
        if v is not None:
            row["visual_available"], row["visual_score"], row["visual_n_frames"] = int(v.available), "" if v.score is None else v.score, v.n_units or 0
        if a is not None:
            row["audio_available"], row["audio_score"], row["audio_n_units"] = int(a.available), "" if a.score is None else a.score, a.n_units or 0
        row["warnings"] = "|".join(x["code"] for x in res.warnings)
    except Exception as e:  # noqa: BLE001 - recorded, never retried silently
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    row["ms"] = round((time.time() - t) * 1000)
    return row


def cmd_score() -> int:
    from dfd.infer.real_adapter import RealAdapter
    ids = pool()
    done = set()
    if PRED.exists():
        with open(PRED, encoding="utf-8", newline="") as f:
            done = {r["sample_id"] for r in csv.DictReader(f)}
    todo = [(s, p) for s, p in ids if s not in done]
    print(f"pool {len(ids)}, already scored {len(done)}, to score {len(todo)}", flush=True)
    if not todo:
        return 0
    ad = RealAdapter(CKPT)
    new = not PRED.exists()
    t_all = time.time()
    with open(PRED, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for k, (sid, rel) in enumerate(todo, 1):
            row = score_one(ad, DATA / rel)
            row["sample_id"] = sid
            w.writerow(row)
            if k % 100 == 0:
                f.flush()
                print(f"{k}/{len(todo)} {round(time.time() - t_all)} s", flush=True)
    return 0


def cmd_reproduce() -> int:
    from dfd.infer.real_adapter import RealAdapter
    out = OUT / "reproducibility.json"
    if out.exists():
        sys.exit(f"{out} exists; not overwritten")
    with open(PRED, encoding="utf-8", newline="") as f:
        stored = {r["sample_id"]: r for r in csv.DictReader(f)}
    ids = pool()
    if len(stored) != len(ids):
        sys.exit("refused: predictions do not cover the pool")
    pick = sorted(np.random.default_rng(20261031).choice(len(ids), 20, replace=False).tolist())
    ad = RealAdapter(CKPT)
    rows, worst = [], 0.0
    for i in pick:
        sid, rel = ids[i]
        new = score_one(ad, DATA / rel)
        rec = {"sample_id": sid}
        for m in ("visual", "audio"):
            a, b = stored[sid][f"{m}_score"], new[f"{m}_score"]
            if a != "" and b != "":
                d = abs(float(a) - float(b))
                worst = max(worst, d)
                rec[m] = {"stored": float(a), "rescored": float(b), "abs_diff": d}
            else:
                rec[m] = {"stored": a, "rescored": b}
        rows.append(rec)
    res = {"n": len(rows), "max_abs_diff": worst, "tolerance": 0.02, "pass": worst <= 0.02, "seed": 20261031, "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print("max abs diff", worst, "pass", res["pass"])
    return 0


if __name__ == "__main__":
    if not (OUT / "prereg.json").exists():
        sys.exit("refused: preregistration record data/_results/posteval/E-033/prereg.json is missing")
    sys.exit({"score": cmd_score, "reproduce": cmd_reproduce}[sys.argv[1]]())
