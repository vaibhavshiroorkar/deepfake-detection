"""Real-world real clips for the website's lines: everyday speech and talking-head videos, real only.

    .venv/Scripts/python.exe scripts/build_realworld_reals.py

The research corpora hold studio, broadcast and actor footage, so a visitor's webcam or phone recording
read as fake against them alone (found 11-19). This gathers real clips scored through the website's own
path into results/realworld/reals.csv, which scripts/build_web_calibration.py adds as their own corpora:

- Web speech: FLEURS English test and LibriSpeech test-clean (both CC BY 4.0), as downloaded
  (scripts/voice_realworld.py, "original").
- Web speech, re-sent: the same clips after an AAC 48 kbps mono re-encode, as a messaging app sends them.
- Web video: our real talking-head videos (data/generated/raw/real); the generated-video score of their
  first part (scripts/embed_videos_clip.py, the live head in models/website/gen-probe.json) and their
  voice score.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "runs" / "voice-realworld" / "scores.jsonl"
VIDEO = ROOT / "runs" / "gen-probe" / "videos-own.jsonl"
HEAD = ROOT / "models" / "website" / "gen-probe.json"
OUT = ROOT / "results" / "realworld" / "reals.csv"
SPEECH_SETS = ("fleurs-real", "librispeech-real")


def gen_logit(embedding, head: dict) -> float:
    z = (np.asarray(embedding) - np.asarray(head["mean"])) / np.asarray(head["scale"])
    return float(z @ np.asarray(head["coef"]) + head["intercept"])


def main() -> int:
    voice = [
        json.loads(line) for line in VOICE.read_text(encoding="utf-8").splitlines()
    ]
    head = json.loads(HEAD.read_text(encoding="utf-8"))
    rows = []
    for r in voice:
        if r["label"] == 0 and r["set"] in SPEECH_SETS and r["score"] is not None:
            corpus = (
                "Web speech" if r["version"] == "original" else "Web speech, re-sent"
            )
            rows.append(
                {
                    "corpus": corpus,
                    "clip_id": Path(r["path"]).name,
                    "voice": f"{r['score']:.6f}",
                    "gen": "",
                }
            )
    own_voice = {
        Path(r["path"]).name: r["score"]
        for r in voice
        if r["set"] == "own-real"
        and r["version"] == "original"
        and r["score"] is not None
    }
    for line in VIDEO.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["group"] != "real-own" or r["part"] != 0 or r["embedding"] is None:
            continue
        name = Path(r["path"]).name
        v = own_voice.get(name)
        rows.append(
            {
                "corpus": "Web video",
                "clip_id": name,
                "voice": "" if v is None else f"{v:.6f}",
                "gen": f"{gen_logit(r['embedding'], head):.6f}",
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["corpus", "clip_id", "voice", "gen"])
        writer.writeheader()
        writer.writerows(rows)
    counts: dict = {}
    for r in rows:
        counts[r["corpus"]] = counts.get(r["corpus"], 0) + 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
