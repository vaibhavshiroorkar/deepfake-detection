"""Helpers for the Evaluation Lab tests: small synthetic CSVs in tmp_path, never the real data."""

import csv
import json

import numpy as np
from dfd.platform.evallab import EvalLab

COLS = ["run_id", "iteration", "seed", "modality", "dataset", "split", "sample_id", "group_id", "label", "score"]


def write_rows(path, rows, extra=()):
    cols = COLS + list(extra)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    return path


def block(split, dataset, fake, real, *, run="R-s0", it="I-0", seed=0, gprefix=None, sprefix=None, modality="visual", **extra):
    """Rows of one block: one group and one sample per score."""
    gp = gprefix or f"{dataset}_{split}"
    sp = sprefix or gp
    rows = []
    for k, (y, s) in enumerate([(1, v) for v in fake] + [(0, v) for v in real]):
        rows.append({"run_id": run, "iteration": it, "seed": seed, "modality": modality, "dataset": dataset,
                     "split": split, "sample_id": f"{sp}_s{k}", "group_id": f"{gp}_g{k}", "label": y, "score": s, **extra})
    return rows


def synth(rng, n_fake, n_real, strength=2.0):
    f = 1 / (1 + np.exp(-(strength + rng.normal(0, 1.2, n_fake))))
    r = 1 / (1 + np.exp(-(-strength + rng.normal(0, 1.2, n_real))))
    return [round(float(v), 6) for v in f], [round(float(v), 6) for v in r]


def make_lab(tmp_path, dirs=None, ledger_path=None, sync=True):
    return EvalLab(root=tmp_path, dirs=dirs or ["scores"], data_dir=tmp_path / "data" / "_evallab",
                   ledger_path=ledger_path or (tmp_path / "ledger.jsonl"), synchronous=sync)


def log_entry(ledger_path, *, iteration, dataset, split, sha, run_ids=("R-s0",)):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"date": "2026-10-25", "iteration": iteration, "dataset": dataset, "split": split,
                            "run_ids": list(run_ids), "modalities": ["visual"], "score_file_sha256": sha}) + "\n")


def cfg_for(lab, path, **kw):
    """A run config for the catalogue entry of `path` (repo-relative to the lab root)."""
    entry = next(sf for sf in lab.catalogue.scan()["score_files"] if sf["path"].endswith(path.name))
    base = {"score_file_id": entry["id"], "iteration": "I-0", "modality": "visual", "dataset": "dsA", "split": "dev_x",
            "threshold_source": "val_fit", "n_boot": 200, "purpose": "development", "label": "t"}
    base.update(kw)
    return base
