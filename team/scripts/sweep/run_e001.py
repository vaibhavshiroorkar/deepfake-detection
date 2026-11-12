"""E-001 queue (D-036): crops for the Celeb-DF dev_x slice, then frozen-feature extraction per candidate.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sweep/run_e001.py
Each step is resumable. Weights must already be downloaded and verified (scripts/sweep/fetch_encoders.py).
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", r"D:\cache\hf\hub")
os.environ.setdefault("TORCH_HOME", r"D:\cache\torch")
PY = sys.executable
SUB = Path("data/_manifests/_subsets/celebdf_v2_dev_x.csv")
CROPS = "data/_crops/celebdf_v2_yunet_n16"
CANDS = ["IMG-07", "IMG-08", "IMG-01", "IMG-04", "IMG-17", "IMG-13", "IMG-15"]


def subset_manifest():
    split = {r["sample_id"]: r["split"] for r in csv.DictReader(open("data/_splits/celebdf_v2-82a3a70594e9.csv", encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open("data/_manifests/celebdf_v2.manifest.csv", encoding="utf-8")) if split.get(r["sample_id"]) == "dev_x"]
    SUB.parent.mkdir(parents=True, exist_ok=True)
    with open(SUB, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "path", "modality"])
        w.writerows([[r["sample_id"], r["path"], r["modality"]] for r in rows])
    print("dev_x videos", len(rows), flush=True)


def main():
    subset_manifest()
    subprocess.run([PY, "-m", "dfd.prep.pipeline", "--manifest", str(SUB), "--out", CROPS, "--workers", "8"], check=True)
    for c in CANDS:
        print("== extract", c, flush=True)
        r = subprocess.run([PY, "-m", "dfd.sweep.extract_video", "--cand", c])
        if r.returncode:
            print("FAILED", c, r.returncode, flush=True)


if __name__ == "__main__":
    main()
