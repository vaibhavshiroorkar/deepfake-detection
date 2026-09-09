"""Fusion ablation: does combining streams beat every stream on its own?

This is objective 3's success metric stated directly, so it is measured rather
than asserted. Every subset of the available branches gets its own fusion model,
fitted on the same out-of-fold features and scored on the same partitions, so
the only thing varying is which branches are in the input.

Two partitions, because they disagree and the disagreement is the finding:
the held-out in-domain test, and DFDC, which is the only cross-corpus set here
carrying audio and not built on VoxCeleb2.

Single-branch rows are the calibrated branch score, not a fusion of one, so
"fusion beats every individual stream" is a comparison against the branch as it
actually performs rather than against a handicapped version of it.

    uv run python scripts/run_ablation.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path


def _assemble(store_path: Path) -> tuple[list[str], dict, dict]:
    """Clip ids present for all three branches, their logits, and their labels."""
    from deepfake_detection.fusion.store import FeatureStore

    logits: dict[str, dict[str, float]] = collections.defaultdict(dict)
    quality: dict[str, tuple[float, bool, float]] = {}
    labels: dict[str, int] = {}
    for row in FeatureStore(store_path).read():
        if not row.available:
            continue
        logits[row.clip_id][row.branch] = row.logit
        labels[row.clip_id] = row.label
        quality[row.clip_id] = (
            row.face_coverage,
            row.audio_clipped,
            row.av_duration_delta_sec,
        )
    complete = sorted(k for k, v in logits.items() if len(v) == 3)
    return complete, {"logits": logits, "quality": quality}, labels


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.fusion.late import FusionSample, LateFusion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    run = arguments.run_dir

    train_ids, train_data, train_labels = _assemble(run / "features" / "oof.parquet")
    partitions = {}
    for name, store in (("in-domain", "test.parquet"), ("dfdc", "dfdc.parquet")):
        path = run / "features" / store
        if path.is_file():
            partitions[name] = _assemble(path)

    print(f"out-of-fold training clips: {len(train_ids):,}")
    for name, (ids, _, labels) in partitions.items():
        y = [labels[i] for i in ids]
        print(f"{name}: {len(ids):,} clips (fake {sum(y)}, real {len(y) - sum(y)})")
    print()

    def samples(ids, data, branches):
        return [
            FusionSample(
                branch_logits={b: data["logits"][i][b] for b in branches},
                face_coverage=data["quality"][i][0],
                audio_clipped=data["quality"][i][1],
                av_duration_delta_sec=data["quality"][i][2],
            )
            for i in ids
        ]

    branches = ("visual", "audio", "sync")
    combinations = [
        tuple(c)
        for size in (1, 2, 3)
        for c in itertools.combinations(branches, size)
    ]

    results = []
    for combo in combinations:
        model = LateFusion(branch_names=combo).fit(
            samples(train_ids, train_data, combo),
            [train_labels[i] for i in train_ids],
        )
        row = {"branches": list(combo), "size": len(combo)}
        for name, (ids, data, labels) in partitions.items():
            probabilities = model.predict_proba(samples(ids, data, combo))
            y = np.array([labels[i] for i in ids])
            row[name] = (
                float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None
            )
        results.append(row)

    # A single-branch row above is that branch calibrated; report the raw branch
    # too, so the comparison is against the stream as it actually behaves.
    raw = {}
    for name, (ids, data, labels) in partitions.items():
        y = np.array([labels[i] for i in ids])
        raw[name] = {
            b: float(roc_auc_score(y, [data["logits"][i][b] for i in ids]))
            for b in branches
        }

    width = max(len(" + ".join(r["branches"])) for r in results)
    header = f"{'branches':{width}}  " + "  ".join(f"{n:>10}" for n in partitions)
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: (r["size"], r["branches"])):
        label = " + ".join(row["branches"])
        cells = "  ".join(
            f"{row[n]:10.4f}" if row[n] is not None else f"{'n/a':>10}"
            for n in partitions
        )
        print(f"{label:{width}}  {cells}")
    print()
    print("raw branch scores, uncalibrated:")
    for b in branches:
        cells = "  ".join(f"{raw[n][b]:10.4f}" for n in partitions)
        print(f"  {b:{width - 2}}  {cells}")

    print()
    for name in partitions:
        best = max(
            (r for r in results if r[name] is not None), key=lambda r: r[name]
        )
        all_three = next(r for r in results if r["size"] == 3)
        singles = [r for r in results if r["size"] == 1 and r[name] is not None]
        beats_all = all(all_three[name] > s[name] for s in singles)
        print(
            f"{name}: best is {' + '.join(best['branches'])} at {best[name]:.4f}; "
            f"all-three {'beats' if beats_all else 'does NOT beat'} every single stream"
        )

    payload = {
        "out_of_fold_clips": len(train_ids),
        "partitions": {n: len(p[0]) for n, p in partitions.items()},
        "combinations": results,
        "raw_branch_auc": raw,
    }
    out = arguments.output or run / "evaluation" / "fusion-ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
