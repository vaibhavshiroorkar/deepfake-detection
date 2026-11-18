"""Build the exhibition demo clip set and record what the website answers for each.

    .venv/Scripts/python.exe scripts/build_demo_clips.py [--per-type 3] [--device cuda]
    .venv/Scripts/python.exe scripts/build_demo_clips.py --rescore [--device cpu]

--rescore scores the clips already listed in docs/demo-clips.csv again, without drawing or copying,
for when the website's scoring changes. Like the site, a clip is checked in 3-second parts and its
most unusual part decides.

Candidates are seeded draws from clips the models never trained on: the FakeAVCeleb test
partition and HiFi-AVDF. The original file is copied to data/demo/upload/ (what gets uploaded)
and an H.264 copy to data/demo/watch/ (to play to the audience; FakeAVCeleb's MPEG-4 files do not
play in browsers). Git ignores data/: these corpora may not be redistributed, so the clips are for
our own machine only. Each upload file is scored exactly as the site scores an upload (scorer.py,
then calibration.py), and docs/demo-clips.csv records the outcome.

Upload the originals, not the H.264 copies: one extra compression lowers the face check's score
(observed 11-17; project log).

The demo shows a chosen subset of these, picked to show each outcome. That choice says nothing
about accuracy; the paper's numbers do. docs/DEMO.md says so beside the list.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from deepfake_detection.webapp.calibration import Calibration
from deepfake_detection.webapp.scorer import ClipRefused, Scorer

ROOT = Path(__file__).resolve().parents[1]
UPLOAD, WATCH = ROOT / "data" / "demo" / "upload", ROOT / "data" / "demo" / "watch"
LIST = ROOT / "docs" / "demo-clips.csv"
FAKEAVCELEB_TYPES = {   # label: (manipulation_type, method or None)
    "real": ("RealVideo-RealAudio", None),
    "cloned voice": ("RealVideo-FakeAudio", "rtvc"),
    "face swap": ("FakeVideo-RealAudio", "faceswap"),
    "FSGAN face swap": ("FakeVideo-RealAudio", "fsgan"),
    "lip sync (Wav2Lip)": ("FakeVideo-RealAudio", "wav2lip"),
    "face swap + Wav2Lip, cloned voice": ("FakeVideo-FakeAudio", "faceswap-wav2lip"),
}
HIFI_TYPES = ("real", "pixverseV5", "sora2", "veo3.1", "kling2.5")


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def candidates(per_type: int, seed: int = 17) -> list[dict]:
    rng = random.Random(seed)  # noqa: S311 - a fixed sample of clips, not a secret
    test = {r["clip_id"] for r in _rows(ROOT / "runs/p001/split/test.csv")}
    fav = [r for r in _rows(ROOT / "data/manifests/fakeavceleb.csv") if r["clip_id"] in test]
    picked = []
    for label, (kind, method) in FAKEAVCELEB_TYPES.items():
        pool = [r for r in fav if r["manipulation_type"] == kind and (method is None or r["method"] == method)]
        picked += [{"corpus": "FakeAVCeleb", "type": label, **r} for r in rng.sample(pool, min(per_type, len(pool)))]
    hifi = _rows(ROOT / "data/manifests/hifiavdf.csv")
    for method in HIFI_TYPES:
        pool = [r for r in hifi if r["method"] == method]
        label = "real" if method == "real" else f"generated ({method})"
        picked += [{"corpus": "HiFi-AVDF", "type": label, **r} for r in rng.sample(pool, min(per_type, len(pool)))]
    return picked


def reencode(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p",  # noqa: S603, S607
                    "-c:a", "aac", "-movflags", "+faststart", str(target)], check=True, timeout=120)


def decide_clip(calibration: Calibration, scorer, path: Path, work: Path) -> dict:
    """What the website answers for one file: every 3-second part is scored, then one decision."""
    try:
        parts = scorer.score(path, work)
    except ClipRefused as error:
        return {"outcome": "cant_check", "raised": str(error), "face": "", "voice": "", "sync": "",
                "parts": 0, "part": ""}
    outcome = calibration.decide_parts([part.scores() for part in parts])
    deciding = parts[outcome.part] if outcome.part is not None else None
    pct = {k: "" if v is None else f"{v:.3f}" for k, v in outcome.percentiles.items()}
    return {"outcome": outcome.kind, "raised": "+".join(outcome.raised), **pct, "parts": len(parts),
            "part": "" if deciding is None else f"{deciding.start:.1f}-{deciding.end:.1f}"}


def rescore(calibration: Calibration, scorer) -> list[dict]:
    with open(LIST, encoding="utf-8", newline="") as f:
        listed = list(csv.DictReader(f))
    rows = []
    for row in listed:
        work = Path(tempfile.mkdtemp(prefix="demo-"))
        try:
            result = decide_clip(calibration, scorer, UPLOAD / row["file"], work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rows.append({k: row[k] for k in ("file", "corpus", "type", "truth")} | result)
        print(f"{row['corpus']:11} {row['type']:34} {row['truth']:5} -> {result['outcome']:10} {result['raised']}"
              f"  ({result['parts']} parts)", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rescore", action="store_true", help="score the listed clips again, in parts")
    arguments = parser.parse_args()
    calibration = Calibration.from_json((ROOT / "results" / "web-calibration.json").read_text(encoding="utf-8"))
    scorer = Scorer(device=arguments.device)
    rows = rescore(calibration, scorer) if arguments.rescore else []
    for clip in [] if arguments.rescore else candidates(arguments.per_type):
        name = f"{clip['corpus']}_{clip['type']}_{clip['clip_id']}".replace(" ", "-").replace("(", "").replace(")", "")
        name = "".join(c for c in name if c.isalnum() or c in "-_.+")[:120] + ".mp4"
        source = ROOT / "data" / clip["video_path"]
        target = UPLOAD / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        reencode(source, WATCH / name)
        work = Path(tempfile.mkdtemp(prefix="demo-"))
        try:
            result = decide_clip(calibration, scorer, target, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        truth = "fake" if clip["clip_fake"] == "1" else "real"
        rows.append({"file": name, "corpus": clip["corpus"], "type": clip["type"], "truth": truth} | result)
        print(f"{clip['corpus']:11} {clip['type']:34} {truth:5} -> {result['outcome']:10} {result['raised']}", flush=True)
    with open(LIST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips -> {UPLOAD.relative_to(ROOT)} (upload), {WATCH.relative_to(ROOT)} (watch); "
          f"outcomes -> {LIST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
