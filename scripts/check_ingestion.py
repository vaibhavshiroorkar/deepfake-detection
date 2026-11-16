"""Run the ingestion checks on one neutral corpus and fail if any check fails.

    .venv/Scripts/python.exe scripts/check_ingestion.py --manifest data/manifests/dfdc-neutral.csv \
        --features runs/avhubert-dfdc-neutral-a10 --audit results/shortcut-audit/dfdc-neutral/summary.json \
        --probe --output results/ingestion/dfdc.json

Each input is optional; a check runs only when its input is given. LAV-DF's label check needs
`--segments data/LAV-DF/LAV-DF/metadata.min.json`. The cut log next to the manifest gives each clip's
window. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from deepfake_detection.data.ingestion_checks import (
    face_coverage,
    full_span,
    label_in_window,
    shortcut_ceiling,
)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _durations(path: Path) -> tuple[float, float]:
    """(video seconds, audio seconds) of one file, from the stream headers."""
    out = subprocess.run(  # noqa: S603 - fixed ffprobe arguments, the path is a local file
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],  # noqa: S607
        capture_output=True, text=True, check=True).stdout
    streams = {s["codec_type"]: float(s.get("duration", 0.0)) for s in json.loads(out)["streams"]}
    return streams.get("video", 0.0), streams.get("audio", 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, help="feature folder with index.csv (face coverage)")
    parser.add_argument("--audit", type=Path, help="shortcut-audit summary.json of the neutral copies")
    parser.add_argument("--segments", type=Path, help="metadata with fake_periods (LAV-DF)")
    parser.add_argument("--probe", action="store_true", help="read each file's stream lengths (full span)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rows = _rows(args.manifest)
    results = []
    if args.features:
        index = _rows(args.features / "index.csv")
        results.append(face_coverage([float(r["face"]) if r["status"] == "ok" else 0.0 for r in index]))
    if args.probe:
        paths = [args.data_dir / r["video_path"].replace("\\", "/") for r in rows]
        with ThreadPoolExecutor(args.workers) as pool:
            # Clips with no audio track are counted by the audit, not here.
            results.append(full_span([d for d in pool.map(_durations, paths) if d[1] > 0]))
    if args.segments:
        meta = {m["file"]: m for m in json.loads(args.segments.read_text(encoding="utf-8"))}
        cuts = {r["clip_id"]: r for r in _rows(args.manifest.with_suffix(".cuts.csv"))}
        windows, periods = [], []
        for r in rows:
            if r["clip_fake"] != "1":
                continue
            path = r["video_path"].replace("\\", "/")
            m = meta[f"{Path(path).parent.name}/{Path(path).name}"]
            start = float(cuts[r["clip_id"]]["offset_sec"])
            windows.append((start, start + float(cuts[r["clip_id"]]["kept_sec"])))
            periods.append(m["fake_periods"])
        results.append(label_in_window(windows, periods))
    if args.audit:
        summary = json.loads(args.audit.read_text(encoding="utf-8"))
        results.append(shortcut_ceiling(summary["clip_fake"]["shortcut_ceiling_auc"]))
    if not results:
        parser.error("give at least one of --features, --probe, --segments, --audit")

    for r in results:
        print(f"{'pass' if r.passed else 'FAIL'}  {r.name:20} {r.value:.3f} (bar {r.bar})  {r.detail}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {"manifest": str(args.manifest), "checks": [asdict(r) for r in results],
                  "passed": all(r.passed for r in results)}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
