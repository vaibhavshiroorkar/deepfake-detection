"""Do two streams fail on the same clips?

Fusion can only help when streams make different mistakes. Two models that are
wrong about the same videos have nothing to tell each other, however good each
looks alone, and combining them adds parameters without adding evidence.

This reports that directly, per partition:

  - Pearson correlation between two streams' logits, which says whether they
    rank clips the same way.
  - Jaccard overlap of the clips each gets wrong at a fixed threshold, which
    says whether they fail together. This is the one that matters: two streams
    can correlate weakly overall and still share every error.

The threshold is the median logit per stream rather than 0. A stream's logits
are uncalibrated and their scale differs, so a shared zero would label one
stream wrong on almost everything and make its overlap meaningless.

    uv run python scripts/stream_correlation.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PARTITIONS = ("holdout", "in-domain", "dfdc")


def _load(path: Path) -> dict[str, dict[str, tuple[float, int]]]:
    """{stream: {clip_id: (logit, label)}} for the available rows only."""
    from deepfake_detection.fusion.store import FeatureStore

    found: dict[str, dict[str, tuple[float, int]]] = {}
    for row in FeatureStore(path).read():
        if row.available:
            found.setdefault(row.branch, {})[row.clip_id] = (row.logit, row.label)
    return found


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    report: dict[str, dict] = {}
    for partition in PARTITIONS:
        path = arguments.run_dir / "features" / f"{partition}.parquet"
        if not path.is_file():
            continue
        streams = _load(path)
        names = sorted(streams)
        if len(names) < 2:
            continue

        print(f"\n=== {partition} ===")
        wrong: dict[str, set[str]] = {}
        for name in names:
            rows = streams[name]
            logits = np.array([value for value, _ in rows.values()])
            labels = np.array([label for _, label in rows.values()])
            threshold = float(np.median(logits))
            predicted = (logits > threshold).astype(int)
            clips = list(rows)
            wrong[name] = {
                clips[i] for i in range(len(clips)) if predicted[i] != labels[i]
            }
            print(f"  {name:24} wrong on {len(wrong[name]):5,} of {len(clips):,}")

        rows_out = []
        print()
        print(f"  {'pair':50} {'corr':>7} {'shared errors':>14}")
        for left, right in itertools.combinations(names, 2):
            shared = sorted(set(streams[left]) & set(streams[right]))
            if len(shared) < 2:
                continue
            a = np.array([streams[left][c][0] for c in shared])
            b = np.array([streams[right][c][0] for c in shared])
            correlation = (
                float(np.corrcoef(a, b)[0, 1])
                if a.std() > 0 and b.std() > 0
                else float("nan")
            )
            union = wrong[left] | wrong[right]
            jaccard = len(wrong[left] & wrong[right]) / len(union) if union else 0.0
            print(f"  {left + ' / ' + right:50} {correlation:7.3f} {jaccard:13.1%}")
            rows_out.append(
                {
                    "left": left,
                    "right": right,
                    "logit_correlation": correlation,
                    "error_overlap": jaccard,
                    "clips": len(shared),
                }
            )
        report[partition] = {"pairs": rows_out}

    if not report:
        print("No feature stores found. Run scripts/score_streams.py first.")
        return 1

    out = arguments.output or arguments.run_dir / "evaluation" / "stream-correlation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
