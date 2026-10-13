"""One table for leave-one-family-out on FaceForensics++.

`score_by_family.py` writes the seen-family baseline, and
`train_family_holdout.sh` writes one file per held-out family. This joins them
so each family has its seen and unseen ROC-AUC side by side, and writes the
single artifact the registry row cites.

The comparison is unpaired. Both models score the same test clips, but
`score_by_family.py` keeps no per-clip scores, so the two intervals are
independent bootstraps and a drop is separated only where they do not overlap.

    uv run python scripts/summarize_family_holdout.py --run-dir runs/ffpp-20260913
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ffpp-20260913"))
    arguments = parser.parse_args(argv)
    evaluation = arguments.run_dir / "evaluation"

    seen = json.loads((evaluation / "by-family-seen.json").read_text(encoding="utf-8"))
    rows = {}
    missing = []
    for family, baseline in sorted(seen["families"].items()):
        path = evaluation / f"by-family-unseen-{family}.json"
        if not path.exists():
            missing.append(family)
            continue
        unseen = json.loads(path.read_text(encoding="utf-8"))["families"][family]
        rows[family] = {
            "seen": baseline["roc_auc"],
            "seen_ci": baseline["ci"],
            "unseen": unseen["roc_auc"],
            "unseen_ci": unseen["ci"],
            "drop": baseline["roc_auc"] - unseen["roc_auc"],
            # Unpaired, so separated means the intervals do not overlap.
            "separated": unseen["ci"][1] < baseline["ci"][0],
            "fake_clips": unseen["fake_clips"],
            "real_clips": unseen["real_clips"],
        }

    print(f"{'held-out family':28} {'seen':>7} {'unseen':>7} {'drop':>7}")
    for family, row in rows.items():
        print(
            f"{family:28} {row['seen']:7.4f} {row['unseen']:7.4f} "
            f"{row['drop']:+7.4f}{'' if row['separated'] else '  overlapping'}"
        )
    summary = {
        "protocol": "leave-one-family-out, FaceForensics++ c23, visual stream",
        "families": rows,
        "missing": missing,
    }
    if rows:
        count = len(rows)
        summary["macro_seen"] = sum(r["seen"] for r in rows.values()) / count
        summary["macro_unseen"] = sum(r["unseen"] for r in rows.values()) / count
        summary["macro_drop"] = summary["macro_seen"] - summary["macro_unseen"]
        print(
            f"{'macro':28} {summary['macro_seen']:7.4f} "
            f"{summary['macro_unseen']:7.4f} {summary['macro_drop']:+7.4f}"
        )
    if missing:
        print(f"no held-out arm yet for: {', '.join(missing)}")

    output = evaluation / "family-holdout.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
