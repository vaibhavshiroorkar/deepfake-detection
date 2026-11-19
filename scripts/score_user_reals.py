"""Score real recordings from everyday devices for the website's lines: webcams, headsets, phones, re-sent files.

    .venv/Scripts/python.exe scripts/score_user_reals.py [--device cuda]

Every file in data/user-real (git-ignored: these are people's own recordings) must be a real recording, made
with consent. Each goes through the website's own path (site.to_clip for recordings, then every 3-second part)
and each part's four check scores are written to results/realworld/user-reals.csv as the corpus "Everyday
devices". scripts/build_web_calibration.py adds them as real clips when results/web-models.json names the file
under "extra_reals", so a visitor's real webcam or phone recording is compared with recordings like it.
Found 11-19: a real webcam recording with a headset read as fake to the voice check (0.91).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FOLDER = ROOT / "data" / "user-real"
OUT = ROOT / "results" / "realworld" / "user-reals.csv"
CORPUS = "Everyday devices"
CHECKS = ("face", "voice", "sync", "gen")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["DDF_SCORER_DEVICE"] = args.device
    from deepfake_detection.webapp import site

    scorer = site.make_scorer()
    rows = []
    for path in sorted(
        p for p in FOLDER.iterdir() if p.suffix.lower() in site.EXTENSIONS
    ):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            kind = site.classify(path.suffix.lower(), "", site.probe(path))
            clip = (
                path
                if kind == "video" and path.suffix.lower() != ".webm"
                else site.to_clip(path, kind, work / "clip.mp4")
            )
            for i, part in enumerate(scorer.score(clip, work / "job", mode=kind)):
                scores = part.scores()
                rows.append(
                    {"corpus": CORPUS, "clip_id": f"{path.name}#{i}"}
                    | {
                        c: "" if scores.get(c) is None else f"{scores[c]:.6f}"
                        for c in CHECKS
                    }
                )
        print(path.name, "scored", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["corpus", "clip_id", *CHECKS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} parts -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
