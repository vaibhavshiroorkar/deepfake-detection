"""Face crops for the sealed test videos (data preparation, not evaluation): no score is computed and no label is read.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/prepare_crops.py [--workers 6] [--only celebdf_v2 dfd celebdf_pp lavdf_subset]

For every video dataset the rows of the sealed test split (test_x or test_in) that have no finished crops are written to a subset manifest under
data/_manifests/_subsets/ and processed by dfd.prep.pipeline with its default settings (YuNet, 16 frames, 1.3 enlargement, 224 px, JPEG 95: the
same preprocessing id as the training crops). The split file is read as plain CSV to know which videos to prepare; the loaders' final-only guard
is not involved because nothing is evaluated. Resumable: finished videos are skipped by the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from dfd.prep.pipeline import folder_name

JOBS = {   # key -> manifest, split csv, sealed split name, crops dir
    "celebdf_v2": ("data/_manifests/celebdf_v2.manifest.csv", "data/_splits/celebdf_v2-82a3a70594e9.csv", "test_x", "data/_crops/celebdf_v2_yunet_n16"),
    "dfd": ("data/_manifests/dfd.manifest.csv", "data/_splits/dfd-f5ff5576814a.csv", "test_x", "data/_crops/dfd_yunet_n16"),
    "celebdf_pp": ("data/_manifests/celebdf_pp.manifest.csv", "data/_splits/celebdf_pp-3f9df077b9eb.csv", "test_x", "data/_crops/celebdf_pp_yunet_n16"),
    "lavdf_subset": ("data/_manifests/lavdf_subset.manifest.csv", "data/_splits/lavdf_subset-e9d12decaf1f.csv", "test_in", "data/_crops/lavdf_subset_yunet_n16"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    a = ap.parse_args()
    for key in a.only:
        manifest, split_csv, split, crops = JOBS[key]
        keep = {r["sample_id"] for r in csv.DictReader(open(split_csv, encoding="utf-8", newline="")) if r["split"] == split}
        rows = list(csv.reader(open(manifest, encoding="utf-8", newline="")))
        head, body = rows[0], rows[1:]
        todo = [r for r in body if r[0] in keep and not (Path(crops) / folder_name(r[0]) / "_done.json").is_file()]
        print(f"{key}: {len(keep)} sealed {split} videos, {len(todo)} without finished crops", flush=True)
        if not todo:
            continue
        sub = Path("data/_manifests/_subsets") / f"{key}_{split}_crops.manifest.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        with open(sub, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(head)
            w.writerows(todo)
        r = subprocess.run([sys.executable, "-m", "dfd.prep.pipeline", "--manifest", str(sub), "--out", crops, "--workers", str(a.workers)])
        if r.returncode != 0:
            raise SystemExit(f"{key}: pipeline exited with {r.returncode}")


if __name__ == "__main__":
    main()
