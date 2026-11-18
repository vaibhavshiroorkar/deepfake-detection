"""Audit the website on real-world versions of known clips: does a real clip stay real, does a fake stay caught?

    .venv/Scripts/python.exe scripts/audit_web_inputs.py [--per-side 20] [--device cuda]

Draws real and fake clips from the four corpora's original files (a fixed seed), keeps the first 4 s so each is
one part, and makes versions a visitor's upload could arrive as:

  as_is       the file as it is
  phone       re-encoded small, as a messaging app or phone share does (H.264 CRF 30 at 480p, AAC 64k)
  browser     a browser recording's format (VP8 video, Opus audio at 32k)
  denoised    the audio through noise suppression, as a phone or headset applies it
  low_res     the video downscaled to 240p

Each version goes through the website's own scorer and lines (video mode). Writes
results/web-input-audit.json: per version, how many real clips were flagged and how many fakes were caught.
Clip names are not written, so the file holds nothing from the datasets.
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

# DFDC has no manifest of original files here, so its cleaned copies stand in.
MANIFESTS = {
    "FakeAVCeleb": "fakeavceleb",
    "DFDC": "dfdc-neutral",
    "LAV-DF": "lavdf-test-sample",
    "HiFi-AVDF": "hifiavdf",
}
VERSIONS = {
    "as_is": [],
    "phone": [
        "-vf",
        "scale=-2:480",
        "-c:v",
        "libx264",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
    ],
    "browser": ["-c:v", "libvpx", "-b:v", "1M", "-c:a", "libopus", "-b:a", "32k"],
    "denoised": ["-c:v", "copy", "-af", "afftdn=nr=25:nf=-40", "-c:a", "aac"],
    "low_res": ["-vf", "scale=-2:240", "-c:v", "libx264", "-c:a", "copy"],
}
SUFFIX = {"browser": ".webm"}


def _rows(name: str) -> list[dict]:
    path = ROOT / "data" / "manifests" / f"{name}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def draw(per_side: int, seed: int = 29) -> list[dict]:
    rng = random.Random(seed)  # noqa: S311 - a seeded sample, not security
    reals, fakes = [], []
    for corpus, name in MANIFESTS.items():
        for r in _rows(name):
            path = ROOT / "data" / r["video_path"]
            if path.exists():
                fake = (
                    r.get("clip_fake") == "1"
                    if "clip_fake" in r
                    else r["manipulation_type"] != "RealVideo-RealAudio"
                )
                (fakes if fake else reals).append({"corpus": corpus, "path": path})
    per_corpus = max(1, per_side // len(MANIFESTS))
    chosen = []
    for side, label in ((reals, 0), (fakes, 1)):
        for corpus in MANIFESTS:
            pool = [c for c in side if c["corpus"] == corpus]
            for c in rng.sample(pool, min(per_corpus, len(pool))):
                chosen.append(c | {"label": label})
    return chosen


def make_version(source: Path, version: str, target: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        "4",
        *VERSIONS[version],
        str(target),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return target


def main() -> int:
    from deepfake_detection.cli.common import _git_commit
    from deepfake_detection.webapp import site
    from deepfake_detection.webapp.scorer import ClipRefused, Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-side", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    scorer, calibration = Scorer(device=args.device), site.calibration()
    tally: dict = defaultdict(
        lambda: {"real": defaultdict(int), "fake": defaultdict(int)}
    )
    by_model: dict = defaultdict(lambda: defaultdict(list))
    for k, clip in enumerate(draw(args.per_side)):
        side = "fake" if clip["label"] else "real"
        for version in VERSIONS:
            with tempfile.TemporaryDirectory() as work:
                work = Path(work)
                try:
                    made = make_version(
                        clip["path"], version, work / f"v{SUFFIX.get(version, '.mp4')}"
                    )
                    parts = scorer.score(made, work / "job")
                    decision = calibration.decide_parts([p.scores() for p in parts])
                    kind = decision.kind
                    for model, pct in decision.percentiles.items():
                        if pct is not None:
                            by_model[(version, side)][model].append(round(pct, 3))
                except ClipRefused:
                    kind = "cant_check"
                except subprocess.CalledProcessError:
                    kind = "not_made"
            tally[version][side][kind] += 1
        print(
            f"{k + 1} {clip['corpus']} {side}: "
            + ", ".join(
                f"{v}={max(tally[v][side], key=tally[v][side].get)}" for v in VERSIONS
            ),
            flush=True,
        )
    summary = {}
    for version, sides in tally.items():
        summary[version] = {}
        for side, counts in sides.items():
            total = sum(counts.values())
            summary[version][side] = {
                "clips": total,
                "counts": dict(counts),
                ("flagged" if side == "real" else "caught"): round(
                    counts.get("signs", 0) / total, 3
                )
                if total
                else None,
                "median_percentile": {
                    m: sorted(v)[len(v) // 2]
                    for m, v in by_model[(version, side)].items()
                },
            }
    out = ROOT / "results" / "web-input-audit.json"
    out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "code_commit": _git_commit(),
                "device": args.device,
                "per_side": args.per_side,
                "versions": summary,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    for version, sides in summary.items():
        r, f = sides.get("real", {}), sides.get("fake", {})
        print(
            f"{version:9} real flagged {r.get('flagged')}  fakes caught {f.get('caught')}  "
            f"real medians {r.get('median_percentile')}"
        )
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
