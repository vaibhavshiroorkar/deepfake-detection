"""Sample LAV-DF by original recording, keeping each original with all its edits.

Every LAV-DF fake is an edit of one real recording. Sampling whole groups keeps
each real clip next to its own fakes, so real and fake differ only in the edit,
not in speaker, room or camera. The official partitions are identity-disjoint,
so a group never crosses train, dev and test.

    uv run python scripts/sample_lavdf.py --split test --groups 1000 --output data/manifests/lavdf-test-sample.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

CUE_FIELDS = ("clip_fake", "video_fake", "audio_fake")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/lavdf-files.csv")
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    with arguments.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == arguments.split]
    groups = sorted({r["source"] for r in rows})
    chosen = set(
        np.random.default_rng(arguments.seed).choice(
            groups, min(arguments.groups, len(groups)), replace=False
        )
    )
    kept = [r for r in rows if r["source"] in chosen]
    for r in kept:
        kind = r["manipulation_type"]
        r["video_fake"] = str(int(kind.startswith("FakeVideo-")))
        r["audio_fake"] = str(int(kind.endswith("-FakeAudio")))
        r["clip_fake"] = str(int(kind != "RealVideo-RealAudio"))
        r.update(
            target1="-",
            target2="-",
            race="unknown",
            gender="unknown",
            leading_silence_sec="0.0",
            sync_start_sec="",
        )
    fields = (
        "clip_id",
        "dataset",
        "video_path",
        "manipulation_type",
        "method",
        "source",
        "target1",
        "target2",
        "clip_fake",
        "video_fake",
        "audio_fake",
        "race",
        "gender",
        "leading_silence_sec",
        "sync_start_sec",
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    print(
        arguments.split,
        len(chosen),
        "groups",
        len(kept),
        "files",
        dict(Counter(r["manipulation_type"] for r in kept)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
