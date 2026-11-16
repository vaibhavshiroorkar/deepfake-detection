"""Label neutral LAV-DF clips by what their kept window contains (P-001 amendment A9).

    .venv/Scripts/python.exe scripts/lavdf_window_labels.py \
        --manifest data/manifests/lavdf-test-neutral.csv --metadata data/LAV-DF/LAV-DF/metadata.min.json \
        --output data/manifests/lavdf-test-window.csv

A clip is fake if its kept window overlaps manipulated time by at least one video
frame. The file label stays in `file_clip_fake` so both can be reported.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from deepfake_detection.data.ingestion_checks import FRAME_SEC as MIN_OVERLAP_SEC
from deepfake_detection.data.ingestion_checks import overlap


def window_label(row: dict, meta: dict, cut: dict) -> dict:
    start = float(cut["offset_sec"])
    end = start + float(cut["kept_sec"])
    fake = overlap(start, end, meta["fake_periods"]) >= MIN_OVERLAP_SEC
    out = dict(row)
    out["file_clip_fake"] = row["clip_fake"]
    out["clip_fake"] = str(int(fake))
    out["video_fake"] = str(int(fake and meta["modify_video"]))
    out["audio_fake"] = str(int(fake and meta["modify_audio"]))
    if not fake:
        out["manipulation_type"] = "RealVideo-RealAudio"
        out["method"] = "lavdf-RealVideo-RealAudio"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    meta = {m["file"]: m for m in json.loads(args.metadata.read_text(encoding="utf-8"))}
    cuts_path = args.manifest.with_suffix(".cuts.csv")
    with cuts_path.open(newline="", encoding="utf-8") as h:
        cuts = {r["clip_id"]: r for r in csv.DictReader(h)}
    with args.manifest.open(newline="", encoding="utf-8") as h:
        reader = csv.DictReader(h)
        fields = list(reader.fieldnames or ()) + ["file_clip_fake"]
        rows = list(reader)
    out, moves = [], Counter()
    for row in rows:
        name = row["video_path"].replace("\\", "/").split("/")[-1]
        split = Path(row["video_path"].replace("\\", "/")).parent.name
        labelled = window_label(row, meta[f"{split}/{name}"], cuts[row["clip_id"]])
        moves[(row["clip_fake"], labelled["clip_fake"])] += 1
        out.append(labelled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    summary = {f"file {a} -> window {b}": n for (a, b), n in sorted(moves.items())}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
