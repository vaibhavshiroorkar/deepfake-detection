"""E-034 POST HOC (not preregistered): per-generator AP and accuracy with each generator's fakes against ITS OWN paired reals.

Why: the preregistered secondary analysis scored each generator's fakes against all G4 reals, so its prevalence (4 to 22 percent fake) differs
from the paper's per-generator pairing and its AP is not on the scale of the Community Forensics Table 2 values. Here a real image belongs to a
generator's cell when the index row's model_name names that generator (the card's pairing). Reads the stored E-034 Lab score file only; writes
data/_results/posteval/E-034/posthoc_paired.json once. Nothing is chosen or changed on it.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/posteval/posthoc_e034_paired.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from dfd.eval.metrics import roc_auc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "_results" / "posteval" / "E-034" / "posthoc_paired.json"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwritten")
thr = json.load(open(ROOT / "checkpoints" / "I-5-freeze-candidate" / "metadata.json"))["tasks"]["T-I"]["threshold"]["value"]
idx = {(r["shard"], int(r["row"])): r for r in csv.DictReader(open(ROOT / "data" / "cf_eval" / "_meta" / "index.csv", encoding="utf-8"))}
man = {r["sample_id"]: r for r in csv.DictReader(open(ROOT / "data" / "_manifests" / "posteval" / "e034_cfeval.manifest.csv", encoding="utf-8"))}
rows = list(csv.DictReader(open(ROOT / "data" / "_evallab" / "imports" / "posteval" / "E-034_TI.csv", encoding="utf-8")))
cells = defaultdict(lambda: {"y": [], "s": []})
for r in rows:
    m = man[r["sample_id"]]
    gen = idx[(m["shard"], int(m["row"]))]["model_name"]
    cells[gen]["y"].append(int(r["label"]))
    cells[gen]["s"].append(float(r["score"]))
per, skipped = {}, {}
for gen, c in sorted(cells.items()):
    y, s = np.array(c["y"]), np.array(c["s"])
    nf, nr = int(y.sum()), int((1 - y).sum())
    if nf == 0 or nr == 0:
        skipped[gen] = {"n_fake": nf, "n_real": nr, "reason": "one class only in the downloaded shards"}
        continue
    pred = s >= thr
    per[gen] = {"n_fake": nf, "n_real": nr, "auc": roc_auc(y, s), "ap": float(average_precision_score(y, s)),
                "accuracy": float(np.mean(pred == (y == 1))),
                "balanced_accuracy": float(0.5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0])))}
res = {"label": "post hoc, not preregistered; per-generator cells with the card's own pairing; frozen threshold %.6f" % thr,
       "per_generator": per, "skipped_cells": skipped, "n_cells": len(per),
       "mean_ap": float(np.mean([v["ap"] for v in per.values()])) if per else None,
       "mean_accuracy": float(np.mean([v["accuracy"] for v in per.values()])) if per else None,
       "mean_auc": float(np.mean([v["auc"] for v in per.values()])) if per else None}
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
