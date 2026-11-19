"""The website's voice check on real speech from the web, and on Veo 3 voices before and after a re-send.

    .venv/Scripts/python.exe scripts/voice_realworld.py [--per-set 150] [--device cuda]

Sets (fixed seed): FLEURS English test (CC BY 4.0) and LibriSpeech test-clean (CC BY 4.0), both real
speech downloaded to data/downloads/real-speech; our own Veo 3 clips and their real clips
(data/generated/raw). Each item is scored as the website scores a voice recording (site.to_clip("voice"),
first part), once as it is and once after an AAC 48 kbps mono re-encode, the kind of compression a
messaging app applies. Writes runs/voice-realworld/scores.jsonl (resumes) and
results/voice-realworld/report.json: per set and version, the share at or above the website's flag
line and its "can't tell" line for one part.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "runs" / "voice-realworld"
REPORT = ROOT / "results" / "voice-realworld" / "report.json"
SPEECH = ROOT / "data" / "downloads" / "real-speech"


def sets(per_set: int) -> dict[str, list[tuple[Path, int]]]:
    rng = random.Random(41)
    pools = {
        "fleurs-real": (sorted((SPEECH / "fleurs-en" / "x").rglob("*.wav")), 0),
        "librispeech-real": (sorted((SPEECH / "librispeech" / "x").rglob("*.flac")), 0),
        "own-real": (sorted((ROOT / "data" / "generated" / "raw" / "real").glob("*.mp4")), 0),
        "veo3-own": (sorted((ROOT / "data" / "generated" / "raw" / "fake").rglob("*.mp4")), 1),
    }
    out = {}
    for name, (files, label) in pools.items():
        files = list(files)
        rng.shuffle(files)
        out[name] = [(f, label) for f in files[:per_set]]
    return out


def resend(source: Path, target: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "48k", str(target)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target


def score(scorer, site, path: Path, work: Path) -> float | None:
    from deepfake_detection.preprocessing.neutralize import NeutralSettings, plan_windows

    clip = site.to_clip(path, "voice", work / "clip.mp4")
    parts = plan_windows(clip, NeutralSettings())
    if not parts:
        return None
    neutral = scorer._neutralize(clip, work / "neutral.mp4", parts[0][0])
    checks, _ = scorer._face_and_voice(neutral, work / "cache", offset=parts[0][0], top=1)
    return checks["voice"].score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-set", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["DDF_SCORER_DEVICE"] = args.device
    from deepfake_detection.webapp import site

    scorer = site.make_scorer()
    calibration = site.calibration()
    RUNS.mkdir(parents=True, exist_ok=True)
    scores_file = RUNS / "scores.jsonl"
    done = set()
    if scores_file.exists():
        done = {(r["set"], r["path"], r["version"]) for r in map(json.loads, scores_file.read_text(encoding="utf-8").splitlines())}
    for name, items in sets(args.per_set).items():
        for path, label in items:
            for version in ("original", "resent"):
                key = (name, str(path), version)
                if key in done:
                    continue
                with tempfile.TemporaryDirectory() as work:
                    work = Path(work)
                    try:
                        source = path if version == "original" else resend(path, work / "resent.m4a")
                        value = score(scorer, site, source, work)
                        reason = None
                    except (site.ClipRejected, RuntimeError, subprocess.SubprocessError) as error:
                        value, reason = None, f"{type(error).__name__}: {error}"[:160]
                with open(scores_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"set": name, "path": str(path), "label": label, "version": version, "score": value, "reason": reason}) + "\n")
            print(name, path.name, flush=True)
    rows = [json.loads(line) for line in scores_file.read_text(encoding="utf-8").splitlines()]
    report = {"flag_line": calibration.threshold, "lower_line": calibration.lower, "sets": {}}
    for name in sets(args.per_set):
        for version in ("original", "resent"):
            values = [r["score"] for r in rows if r["set"] == name and r["version"] == version and r["score"] is not None]
            pct = [calibration.percentiles({"voice": v})["voice"] for v in values]
            report["sets"].setdefault(name, {})[version] = {
                "scored": len(values),
                "flagged": sum(p >= calibration.threshold for p in pct) / max(len(pct), 1),
                "cant_tell_or_flagged": sum(p >= calibration.lower for p in pct) / max(len(pct), 1),
            }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
