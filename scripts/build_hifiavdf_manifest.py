"""Manifest for HiFi-AVDF (Peng et al., CVPR 2026 Findings), a locked target (P-001 A5).

Each generator folder pairs a real video `<n>.mp4` with a generated copy
`<n>_generated.mp4`. The generated copy has both streams generated, so it is
FakeVideo-FakeAudio. The group (`source`) is the pair, so a real clip and its own
generated copy always fall in the same fold of the shortcut audit.

    uv run python scripts/build_hifiavdf_manifest.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

FIELDS = ("clip_id", "dataset", "video_path", "manipulation_type", "method", "source", "target1", "target2",
          "clip_fake", "video_fake", "audio_fake", "race", "gender", "leading_silence_sec", "sync_start_sec")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/HiFi-AVDF"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/hifiavdf.csv"))
    arguments = parser.parse_args(argv)

    rows, missing = [], 0
    for listing in sorted(arguments.root.glob("*.csv")):
        generator = listing.stem
        with listing.open(newline="", encoding="utf-8-sig") as handle:
            for entry in csv.DictReader(handle):
                relative = entry["video_path"].strip()
                if not (arguments.root / relative).is_file():
                    missing += 1
                    continue
                video, audio = entry["visual_label"] == "1", entry["audio_label"] == "1"
                kind = f"{'Fake' if video else 'Real'}Video-{'Fake' if audio else 'Real'}Audio"
                pair = Path(relative).stem.removesuffix("_generated")
                rows.append({
                    "clip_id": f"hifiavdf__{generator}__{Path(relative).stem}",
                    "dataset": "HiFi-AVDF",
                    "video_path": f"HiFi-AVDF/{relative}",
                    "manipulation_type": kind,
                    "method": "real" if kind == "RealVideo-RealAudio" else generator,
                    "source": f"hifiavdf-{generator}-{pair}",
                    "target1": "-", "target2": "-",
                    "clip_fake": int(video or audio), "video_fake": int(video), "audio_fake": int(audio),
                    "race": "unknown", "gender": "unknown", "leading_silence_sec": "0.0", "sync_start_sec": "",
                })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "rows,", missing, "listed but missing;", dict(Counter((r["method"], r["clip_fake"]) for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
