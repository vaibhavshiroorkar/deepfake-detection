"""Time the website's scorer on real clips, part by part, and record it in results/web-timing.json.

    .venv/Scripts/python.exe scripts/time_web_parts.py data/demo/upload/<clip>.mp4 [...]

Runs the same Scorer the website runs, on CPU with two threads (the demo setting), after the models
are loaded. Each clip is checked in 3-second parts; the file keeps the parts and the seconds per clip.
Clip names are not recorded, only their length, so the file holds nothing from the datasets.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

THREADS = 2


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
        check=True,
    )
    return round(float(run.stdout.strip()), 2)


def main(paths: list[str]) -> int:
    import torch

    from deepfake_detection.cli.common import _git_commit
    from deepfake_detection.webapp.scorer import Scorer

    if not paths:
        raise SystemExit("give at least one clip")
    torch.set_num_threads(THREADS)
    start = time.time()
    scorer = Scorer(device="cpu")
    load = time.time() - start
    clips = []
    for name in paths:
        path = Path(name)
        with tempfile.TemporaryDirectory() as work:
            start = time.time()
            parts = scorer.score(path, Path(work))
            seconds = time.time() - start
        clips.append(
            {
                "video_seconds": _duration(path),
                "parts": len(parts),
                "seconds": round(seconds, 1),
            }
        )
        print(clips[-1], flush=True)
    out = ROOT / "results" / "web-timing.json"
    out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "code_commit": _git_commit(),
                "device": "cpu",
                "threads": THREADS,
                "load_seconds": round(load, 1),
                "clips": clips,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
