"""Count the study's cleaned clips with less than 3.00 s of picture or sound, per corpus and side.

The cleaner judged "enough sound" from the container's length, so a source whose audio ends early could
leave a cleaned clip with under 3.00 s of sound. This measures every clip the study's manifests list.

    .venv/Scripts/python.exe scripts/audit_clip_lengths.py [--probes runs/clip-lengths/probes.csv]

Without --probes it runs ffprobe on every clip (about 30 minutes for 31,000 clips) and saves the probes.
Writes results/clip-lengths/summary.json, which scripts/build_numbers.py reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFESTS = {
    "FakeAVCeleb": "fakeavceleb-neutral",
    "DFDC": "dfdc-neutral",
    "LAV-DF": "lavdf-test-neutral",
    "HiFi-AVDF": "hifiavdf-neutral",
}
LENGTH = 3.0


def probe(path: Path) -> dict:
    """Stream lengths in seconds; None for a missing stream. Fails loudly on a file ffprobe cannot read."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    found = {"audio": None, "video": None}
    for stream in json.loads(result.stdout).get("streams", []):
        if stream.get("codec_type") in found and stream.get("duration"):
            found[stream["codec_type"]] = float(stream["duration"])
    return found


def is_real(row: dict) -> bool:
    return row["method"] == "real" or row["manipulation_type"] == "RealVideo-RealAudio"


def load_probes(path: Path) -> dict[str, dict]:
    """Probes saved by an earlier run, keyed by the manifest's video_path (relative to data/)."""
    probes = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = "neutral/" + row["file"].replace("\\", "/")
            probes[key] = {
                k: float(row[k]) if row[k] else None for k in ("audio", "video")
            }
    return probes


def summarize(manifests: dict[str, list[dict]], probes: dict[str, dict]) -> dict:
    summary = {}
    for corpus, rows in manifests.items():
        sides = {
            s: {"clips": 0, "short_video": 0, "short_audio": 0, "no_audio": 0}
            for s in ("real", "fake")
        }
        for row in rows:
            found = probes[row["video_path"]]
            counts = sides["real" if is_real(row) else "fake"]
            counts["clips"] += 1
            counts["short_video"] += (
                found["video"] is not None and found["video"] < LENGTH
            )
            counts["no_audio"] += found["audio"] is None
            counts["short_audio"] += (
                found["audio"] is not None and found["audio"] < LENGTH
            )
        summary[corpus] = sides
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--probes",
        type=Path,
        help="reuse a probes CSV (file, audio, video) instead of probing",
    )
    args = parser.parse_args()
    manifests = {}
    for corpus, name in MANIFESTS.items():
        with (DATA / "manifests" / f"{name}.csv").open(encoding="utf-8") as handle:
            manifests[corpus] = list(csv.DictReader(handle))
    if args.probes:
        probes = load_probes(args.probes)
    else:
        paths = [row["video_path"] for rows in manifests.values() for row in rows]
        with ThreadPoolExecutor(4) as pool:
            probes = dict(
                zip(paths, pool.map(lambda p: probe(DATA / p), paths), strict=True)
            )
        saved = ROOT / "runs" / "clip-lengths" / "probes.csv"
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "audio", "video"])
            for path, found in probes.items():
                writer.writerow(
                    [
                        path.removeprefix("neutral/"),
                        found["audio"] or "",
                        found["video"] or "",
                    ]
                )
    out = ROOT / "results" / "clip-lengths" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summarize(manifests, probes), indent=2) + "\n", encoding="utf-8"
    )
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
