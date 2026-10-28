"""E-030: EER (and the AUC gate) of the speech task from the stored sealed score files of the E-014 freeze. Reads only; runs no model.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/eval/e030_eer.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dfd.eval.bootstrap import cluster_bootstrap
from dfd.eval.metrics import eer, roc_auc

ROOT = Path(__file__).resolve().parents[2]
SETS = (("asvspoof2019_la", "test_in"), ("in_the_wild", "test_x"))
TAGS = ("I-0", "I-2", "I-3", "I-4")
SEED = 20261027


def load(tag):
    rows = defaultdict(dict)                  # (dataset, split) -> seed -> {sample_id: (label, group, score)}
    with open(ROOT / "data" / "_final" / "scores" / f"{tag}.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["split"])
            if r["modality"] == "audio" and key in SETS:
                rows[key].setdefault(int(r["seed"]), {})[r["sample_id"]] = (int(r["label"]), r["group_id"], float(r["score"]))
    return rows


def main():
    out = {"experiment": "E-030", "metric": "eer (src/dfd/eval/metrics.eer)", "bootstrap": {"n": 2000, "seed": SEED, "groups": "group_id"}, "rows": []}
    for tag in TAGS:
        data = load(tag)
        for key in SETS:
            seeds = sorted(data[key])
            ids = sorted(data[key][seeds[0]])
            assert all(sorted(data[key][s]) == ids for s in seeds), "seeds score different samples"
            y = np.array([data[key][seeds[0]][i][0] for i in ids])
            g = np.array([data[key][seeds[0]][i][1] for i in ids])
            S = np.array([[data[key][s][i][2] for i in ids] for s in seeds])
            e_seed = [float(eer(y, row)) for row in S]
            a_seed = [float(roc_auc(y, row)) for row in S]
            be = cluster_bootstrap(y, S, g, metric="eer", seed=SEED)
            ba = cluster_bootstrap(y, S, g, metric="auc", seed=SEED)
            row = {"tag": tag, "dataset": key[0], "split": key[1], "n": int(len(y)), "n_fake": int(y.sum()), "n_groups": int(len(set(g))),
                   "seeds": seeds, "eer_seed_mean": round(float(np.mean(e_seed)), 4), "eer_seed_sd": round(float(np.std(e_seed, ddof=1)), 4),
                   "eer_estimate": round(be.estimate, 4), "eer_ci95": [round(be.low, 4), round(be.high, 4)],
                   "auc_seed_mean": round(float(np.mean(a_seed)), 4), "auc_estimate": round(ba.estimate, 4), "auc_ci95": [round(ba.low, 4), round(ba.high, 4)]}
            out["rows"].append(row)
            print(row)
    dst = ROOT / "data" / "_results" / "e030" / "e030_results.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
