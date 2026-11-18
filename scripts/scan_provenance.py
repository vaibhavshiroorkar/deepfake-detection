"""Run the website's file check on every HiFi-AVDF file and record how many carry an AI label.

    .venv/Scripts/python.exe scripts/scan_provenance.py

Writes results/web-provenance.json: per generator (and for the real clips), the files read and the
files with a label. HiFi-AVDF is the only corpus here made with commercial generators, so it is the
one where a generator's own label could survive.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from deepfake_detection.cli.common import _git_commit
    from deepfake_detection.webapp.provenance import read

    with open(ROOT / "data" / "manifests" / "hifiavdf.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(4) as pool:
        found = list(pool.map(lambda r: bool(read(ROOT / "data" / r["video_path"])), rows))
    files, labelled = Counter(), Counter()
    for row, hit in zip(rows, found, strict=True):
        files[row["method"]] += 1
        labelled[row["method"]] += hit
    out = ROOT / "results" / "web-provenance.json"
    out.write_text(json.dumps({"date": date.today().isoformat(), "code_commit": _git_commit(), "corpus": "HiFi-AVDF",
                               "groups": {g: {"files": files[g], "labelled": labelled[g]} for g in sorted(files)}},
                              indent=1) + "\n", encoding="utf-8")
    for g in sorted(files):
        print(f"{g:12} {labelled[g]:5} / {files[g]:5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
