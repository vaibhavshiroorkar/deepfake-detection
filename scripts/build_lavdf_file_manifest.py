"""One row per LAV-DF file, for audits that read whole files.

The training manifests cut LAV-DF into windows, and one file can give both a
forged window and a genuine one. File-level cues (codec, bitrate, silence)
cannot tell two windows of one file apart, so a file-level audit on a window
manifest understates the shortcut. This manifest labels each file by what was
done to it: `modify_video` and `modify_audio` from LAV-DF's metadata.

The group is the original recording, so a real file and its edits share a fold.

    uv run python scripts/build_lavdf_file_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/LAV-DF/LAV-DF/metadata.min.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/lavdf-files.csv")
    )
    arguments = parser.parse_args(argv)

    entries = json.loads(arguments.metadata.read_text(encoding="utf-8"))
    rows = []
    for entry in entries:
        video = "Fake" if entry["modify_video"] else "Real"
        audio = "Fake" if entry["modify_audio"] else "Real"
        kind = f"{video}Video-{audio}Audio"
        group = (entry["original"] or entry["file"]).replace("/", "_")
        rows.append(
            {
                "clip_id": "lavdf__" + entry["file"].replace("/", "_"),
                "dataset": "LAV-DF",
                "video_path": "LAV-DF/LAV-DF/" + entry["file"],
                "manipulation_type": kind,
                "method": "lavdf-" + kind,
                "source": "lavdf-" + group,
                "split": entry["split"],
            }
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), dict(Counter(r["manipulation_type"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
