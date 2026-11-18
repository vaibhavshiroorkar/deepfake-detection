"""Build the website's picture calibration: the face check's scores on still frames of real clips.

    .venv/Scripts/python.exe scripts/build_picture_calibration.py [--per-side 100] [--device cuda]

A picture is a single frame held still. The face check learned from moving video, so a still can look
unusual to it even when the face is real (a real FakeAVCeleb frame scored above 99 in 100 real video
clips, 11-18). A picture is therefore placed among real pictures, not among real videos.

From each of the four calibration corpora (the clips scripts/build_web_calibration.py uses), up to
--per-side real and --per-side fake cleaned clips are drawn with a fixed seed. One frame at 1.0 s is
saved as a JPEG, as a camera picture would be, and goes through the website's own picture path
(site.to_clip, then the scorer in picture mode). Real frames set the lines; fake frames are counted
to say how many the lines catch. Writes results/web-calibration-picture.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# corpus: (sync predictions that list the calibration clips, cleaned-clip manifest)
CORPORA = {
    "FakeAVCeleb": (
        "p001-sync/sync-test-identity-strict-seed17",
        "fakeavceleb-neutral",
    ),
    "DFDC": ("p001-sync-targets/sync-dfdc-seed17", "dfdc-neutral"),
    "LAV-DF": ("p001-sync-targets/sync-lavdf-window-seed17", "lavdf-test-neutral"),
    "HiFi-AVDF": ("p001-sync-targets/sync-hifiavdf-seed17", "hifiavdf-neutral"),
}
FRAME_AT = 1.0


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def draw(per_side: int, seed: int = 17) -> list[dict]:
    """Up to per_side real and per_side fake calibration clips per corpus, with their cleaned files."""
    chosen = []
    for corpus, (predictions, manifest) in CORPORA.items():
        paths = {
            r["clip_id"]: r["video_path"]
            for r in _csv(ROOT / "data" / "manifests" / f"{manifest}.csv")
        }
        rows = [
            r
            for r in _csv(ROOT / "results" / f"{predictions}-predictions.csv")
            if r["clip_id"] in paths
        ]
        for label in ("0", "1"):
            side = sorted(
                (r for r in rows if r["label"] == label), key=lambda r: r["clip_id"]
            )
            draw_rng = random.Random(f"{seed}-{corpus}-{label}")  # noqa: S311 - a seeded sample, not security
            for r in draw_rng.sample(side, min(per_side, len(side))):
                chosen.append(
                    {
                        "corpus": corpus,
                        "label": int(label),
                        "clip_id": r["clip_id"],
                        "path": ROOT / "data" / paths[r["clip_id"]],
                    }
                )
    return chosen


def main() -> int:
    from deepfake_detection.cli.common import _git_commit
    from deepfake_detection.webapp import site
    from deepfake_detection.webapp.calibration import build
    from deepfake_detection.webapp.scorer import ClipRefused, Scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-side", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    scorer = Scorer(device=arguments.device)
    rows, skipped = [], 0
    clips = draw(arguments.per_side)
    for i, clip in enumerate(clips):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            jpg = work / "picture.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(FRAME_AT),
                    "-i",
                    str(clip["path"]),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(jpg),
                ],
                check=True,
            )
            try:
                parts = scorer.score(
                    site.to_clip(jpg, "picture", work / "picture.mp4"),
                    work / "job",
                    mode="picture",
                )
            except ClipRefused:
                skipped += 1
                continue
        face = parts[0].scores()["face"]
        if face is None:
            skipped += 1
            continue
        rows.append(
            {
                "corpus": clip["corpus"],
                "label": clip["label"],
                "scores": {"face": face, "voice": None, "sync": None},
            }
        )
        if i % 50 == 0:
            print(f"{i + 1}/{len(clips)}", flush=True)
    calibration, summary = build(rows, checks=("face",))
    data = json.loads(calibration.to_json())
    data["meta"] = {
        "built": date.today().isoformat(),
        "code_commit": _git_commit(),
        "device": arguments.device,
        "input": "picture: one frame at 1.0 s of a cleaned calibration clip, saved as JPEG",
        "per_side": arguments.per_side,
        "no_face": skipped,
        "outcomes": summary,
        "note": "Calibration frames come from P-001 test data; the site uses them only to place scores.",
    }
    out = ROOT / "results" / "web-calibration-picture.json"
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"no_face": skipped, "outcomes": summary}, indent=1))
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
