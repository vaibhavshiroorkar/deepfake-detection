"""Completeness of the face crops of the sealed video sets (data preparation check, no score and no label is read).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/freeze/check_crops.py

For every job of prepare_crops.py: videos of the sealed split, videos with a finished crop folder, videos with at least one detected face frame
among the first 16 frames (the frames the scorers use), and the distribution of detected frames per video.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from dfd.prep.pipeline import folder_name

from prepare_crops import JOBS


def main():
    ok_all = True
    for key, (manifest, split_csv, split, crops) in JOBS.items():
        keep = [r["sample_id"] for r in csv.DictReader(open(split_csv, encoding="utf-8", newline="")) if r["split"] == split]
        done = no_face = 0
        hist = Counter()
        for sid in keep:
            d = Path(crops) / folder_name(sid)
            if not (d / "_done.json").is_file():
                continue
            done += 1
            n = 0
            with open(d / "index.csv", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["detected"] == "1" and row["crop_path"] and int(row["frame_index"]) < 16:
                        n += 1
            hist[n] += 1
            no_face += n == 0
        ok = done == len(keep)
        ok_all &= ok
        print(f"{key} {split}: {len(keep)} videos, {done} finished, {no_face} with no detected face frame; detected frames of 16 (count of videos): "
              + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())) + ("  COMPLETE" if ok else "  INCOMPLETE"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
