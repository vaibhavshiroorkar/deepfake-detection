"""E-034: score the frozen T-I head on the downloaded unread Community Forensics-Eval shards (protocol: docs/EXPERIMENTS.md E-034).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/score_e034.py score       # every downloaded row, resumable per shard
    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/score_e034.py reproduce   # rescore 50 seeded rows, compare (tolerance 0.02)

Scoring path: the product's whole-image path, RealAdapter._score_whole_image("T-I", bgr) on cv2.imdecode(bytes, IMREAD_COLOR), one image at a
time (as E-028). The image name, label and generator of each parquet row are checked against data/cf_eval/_meta/index.csv; the label is not
used for scoring. The SHA-256 of the image bytes is stored for the duplicate check. Rows already in predictions.csv are never rewritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts" / "modern_image"))
import _env  # noqa: E402,F401  (adds the pyarrow folder, data/cf_eval/_pylib)

OUT = DATA / "_results" / "posteval" / "E-034"
PRED = OUT / "predictions.csv"
SEL = DATA / "_manifests" / "posteval" / "e034_cfeval.shards.json"
SHARDS = DATA / "cf_eval_e034" / "shards"
CKPT = ROOT / "checkpoints" / "I-5-freeze-candidate"
COLS = ["sample_id", "shard", "row", "sha256", "ok", "score", "h", "w", "error", "ms"]


def index_rows():
    with open(DATA / "cf_eval" / "_meta" / "index.csv", encoding="utf-8") as f:
        return {(r["shard"], int(r["row"])): r for r in csv.DictReader(f)}


def iter_shard(path: Path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    row = 0
    for b in pf.iter_batches(batch_size=32, columns=["image_data", "image_name", "label", "model_name"]):
        d = b.to_pydict()
        for k in range(b.num_rows):
            yield row, {c: d[c][k] for c in d}
            row += 1


def score_bytes(ad, data: bytes) -> dict:
    out = {"sha256": hashlib.sha256(data).hexdigest(), "ok": 0, "score": "", "h": "", "w": "", "error": ""}
    t = time.time()
    try:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            out["error"] = "cv2.imdecode returned None"
        else:
            out["h"], out["w"] = bgr.shape[:2]
            out["score"] = ad._score_whole_image("T-I", bgr)
            out["ok"] = 1
    except Exception as e:  # noqa: BLE001 - recorded, never retried silently
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    out["ms"] = round((time.time() - t) * 1000)
    return out


def cmd_score() -> int:
    from dfd.infer.real_adapter import RealAdapter
    sel = json.load(open(SEL))
    idx = index_rows()
    done = set()
    if PRED.exists():
        with open(PRED, encoding="utf-8", newline="") as f:
            done = {r["shard"] for r in csv.DictReader(f)}
    todo = [s["file"] for s in sel["shards"] if s["file"] not in done]
    print(f"{len(sel['shards'])} shards, {len(done)} already scored, {len(todo)} to score", flush=True)
    if not todo:
        return 0
    ad = RealAdapter(CKPT)
    new = not PRED.exists()
    mism = 0
    t0 = time.time()
    with open(PRED, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for k, shard in enumerate(todo, 1):
            rows = []
            for row, d in iter_shard(SHARDS / shard):
                r = idx.get((shard, row))
                if r is None or r["image_name"] != d["image_name"] or str(d["label"]) != r["label"] or d["model_name"] != r["model_name"]:
                    mism += 1
                    continue
                rec = score_bytes(ad, d["image_data"])
                rec.update({"sample_id": f"cfeval:{Path(shard).stem}:{row}", "shard": shard, "row": row})
                rows.append(rec)
            w.writerows(rows)          # a shard is written whole, so a resume restarts at shard boundaries
            f.flush()
            print(f"{k}/{len(todo)} {shard} rows {len(rows)} mismatches {mism} {round(time.time() - t0)} s", flush=True)
    return 0


def cmd_reproduce() -> int:
    from dfd.infer.real_adapter import RealAdapter
    out = OUT / "reproducibility.json"
    if out.exists():
        sys.exit(f"{out} exists; not overwritten")
    with open(PRED, encoding="utf-8", newline="") as f:
        stored = [r for r in csv.DictReader(f) if r["ok"] == "1"]
    pick = sorted(np.random.default_rng(20261031).choice(len(stored), 50, replace=False).tolist())
    want = {(stored[i]["shard"], int(stored[i]["row"])): stored[i] for i in pick}
    ad = RealAdapter(CKPT)
    rows, worst = [], 0.0
    for shard in sorted({s for s, _ in want}):
        for row, d in iter_shard(SHARDS / shard):
            if (shard, row) in want:
                new = score_bytes(ad, d["image_data"])
                old = want[(shard, row)]
                diff = abs(float(old["score"]) - float(new["score"]))
                worst = max(worst, diff)
                rows.append({"sample_id": old["sample_id"], "stored": float(old["score"]), "rescored": new["score"], "abs_diff": diff,
                             "same_bytes": new["sha256"] == old["sha256"]})
    res = {"n": len(rows), "max_abs_diff": worst, "tolerance": 0.02, "pass": worst <= 0.02 and len(rows) == 50, "seed": 20261031, "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print("max abs diff", worst, "pass", res["pass"])
    return 0


if __name__ == "__main__":
    if not (OUT / "prereg.json").exists():
        sys.exit("refused: preregistration record data/_results/posteval/E-034/prereg.json is missing")
    sys.exit({"score": cmd_score, "reproduce": cmd_reproduce}[sys.argv[1]]())
