"""Measure the website's flag and "can't tell" lines for videos checked in several parts.

    .venv/Scripts/python.exe scripts/build_parts_lines.py [--per-corpus 60] [--device cuda]

The site checks a video in 3-second parts and its most unusual part decides. The Bonferroni bound it used
assumes the parts are unrelated, so it set the line far too high and long fakes could not reach it. This
scores real videos of the calibration corpora whole, part by part, as the site does, and puts the lines
where 5 in 100 (flag) and 20 in 100 (can't tell) real videos of each length cross them, each corpus
weighted equally. Lines never fall below the single-part lines, and never fall as videos get longer.

Scores are saved per video as they come (runs/parts-lines/scores.jsonl), so a crash loses one video.
Writes parts_lines into the calibration file the site reads (web-models.json may name it); the rest is kept.
Scores come from the website's own models, so the lines match what the site runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Original files of real videos. DFDC's are not here, and its cleaned copies are one part each.
MANIFESTS = {
    "FakeAVCeleb": "fakeavceleb",
    "LAV-DF": "lavdf-test-sample",
    "HiFi-AVDF": "hifiavdf",
}
SCORES = ROOT / "runs" / "parts-lines" / "scores.jsonl"
MIN_VIDEOS = 25  # a part count with fewer real videos is pooled with the next one up
MAX_SECONDS = 60  # the website's upload limit


def _duration(path: Path) -> float:
    run = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        return float(run.stdout.strip())
    except ValueError:
        return 0.0


def draw(per_corpus: int, seed: int = 41) -> list[dict]:
    """Real videos long enough for at least two parts, per corpus, with a fixed seed."""
    chosen = []
    for corpus, name in MANIFESTS.items():
        with open(
            ROOT / "data" / "manifests" / f"{name}.csv", encoding="utf-8", newline=""
        ) as f:
            rows = [r for r in csv.DictReader(f) if r.get("clip_fake", "0") == "0"]
        rng = random.Random(f"{seed}-{corpus}")  # noqa: S311 - a seeded sample, not security
        rng.shuffle(rows)
        picked = 0
        for r in rows:
            path = ROOT / "data" / r["video_path"]
            if path.exists() and _duration(path) >= 6.5:
                chosen.append({"corpus": corpus, "clip_id": r["clip_id"], "path": path})
                picked += 1
                if picked == per_corpus:
                    break
    return chosen


def website(device: str):
    """The website's scorer factory and the calibration file it reads (results/web-models.json may swap both)."""
    import os

    os.environ["DDF_SCORER_DEVICE"] = device
    from deepfake_detection.webapp import site

    return site.make_scorer, site._calibration_file("video")


def most_unusual(percentiles: list[float]) -> float:
    return max(percentiles)


def lines(
    videos: list[dict],
    calibration,
    reading=most_unusual,
    floor: tuple[float, float] | None = None,
) -> dict[int, tuple[float, float]]:
    """{parts: (flag, lower)} from the real videos' most unusual part, under `reading` (a part's checks'
    percentiles to one score: the most unusual check, or the evidence sum)."""
    from deepfake_detection.webapp.calibration import corpus_weights, weighted_quantile

    by_count = defaultdict(list)
    for v in videos:
        readings = []
        for s in v["parts"]:
            seen = [p for p in calibration.percentiles(s).values() if p is not None]
            readings.append(reading(seen) if seen else None)
        readings = [r for r in readings if r is not None]
        if len(readings) >= 2:
            by_count[len(readings)].append((v["corpus"], max(readings)))
    out, pool = {}, []
    floor = floor or (calibration.threshold, calibration.lower)
    for n in sorted(by_count):
        pool += by_count[n]
        if len(pool) < MIN_VIDEOS and n != max(by_count):
            continue
        values = [m for _, m in pool]
        weights = corpus_weights([c for c, _ in pool])
        flag = max(weighted_quantile(values, weights, 0.95), floor[0])
        lower = max(weighted_quantile(values, weights, 0.80), floor[1])
        floor = (flag, lower)
        out[n] = (round(flag, 4), round(lower, 4))
        pool = []
    return out


def main() -> int:
    from deepfake_detection.cli.common import _git_commit
    from deepfake_detection.webapp import site
    from deepfake_detection.webapp.calibration import Calibration
    from deepfake_detection.webapp.scorer import ClipRefused

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-corpus", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if SCORES.exists():
        for line in SCORES.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            done[entry["clip_id"]] = entry
    make_scorer, path = website(args.device)
    scorer = None
    videos = []
    for k, video in enumerate(draw(args.per_corpus)):
        if video["clip_id"] not in done:
            scorer = scorer or make_scorer()
            with tempfile.TemporaryDirectory() as work:
                work = Path(work)
                source = work / "upload.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(video["path"]),
                        "-t",
                        str(MAX_SECONDS),
                        "-c",
                        "copy",
                        str(source),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                try:
                    parts = [p.scores() for p in scorer.score(source, work / "job")]
                except ClipRefused:
                    parts = []
            entry = {
                "clip_id": video["clip_id"],
                "corpus": video["corpus"],
                "parts": parts,
            }
            with open(SCORES, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            done[video["clip_id"]] = entry
            print(f"{k + 1} {video['corpus']} {len(parts)} parts", flush=True)
        videos.append(done[video["clip_id"]] | {"corpus": video["corpus"]})
    calibration = Calibration.from_json(path.read_text(encoding="utf-8"))
    measured = lines(videos, calibration)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["parts_lines"] = {str(n): list(v) for n, v in measured.items()}
    if calibration.fusion is not None:
        from deepfake_detection.webapp.calibration import evidence_sum

        fused = lines(
            videos,
            calibration,
            reading=evidence_sum,
            floor=(calibration.fusion.threshold, calibration.fusion.lower),
        )
        data["fusion"]["parts_lines"] = {str(n): list(v) for n, v in fused.items()}
        print("evidence sum:", json.dumps(fused))
    data.setdefault("meta", {})["parts_lines"] = {
        "built": date.today().isoformat(),
        "code_commit": _git_commit(),
        "device": args.device,
        "real_videos": len(videos),
        "per_corpus": args.per_corpus,
        "rule": "flag and can't-tell lines at the 95th and 80th weighted percentile of real videos' most "
        "unusual part, per part count; never below the single-part lines or a shorter count's",
    }
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    site.calibration.cache_clear()
    print(json.dumps(measured))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
