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

**Every row here uses the clip-level label, and that is why the visual number
below is not the one in the branch evaluation report.** The branch reports score
each branch against its own cue: for the visual branch a RealVideo-FakeAudio
clip is genuine, because the video really is unmanipulated, and the branch reads
0.9988 in-domain on that task. Fusion answers the clip-level question instead,
where that same clip is fake, and the identical visual logits read 0.9742.

Quoting 0.9988 beside a fusion number computed on the clip-level label makes
fusion look worth +0.0002, which is a comparison across two different tasks. On
one task, this one, the gain is +0.0159 [+0.0129, +0.0190]. The whole of it is
the 75 RealVideo-FakeAudio clips a visual branch cannot detect even in
principle, and the cue-label ablation the research design asks for is really
this distinction, measured.

    uv run python scripts/run_ablation.py --run-dir runs/program-20260906

`--branch-seed 29` or `43` repeats the whole ablation on that seed's stores: the
out-of-fold store `run_crossfit_seeds.ps1` exported from its fold models, and
the test and DFDC stores scored by its final branches. It writes
`fusion-ablation-seed<N>.json` beside the seed-17 report.
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
    identity: dict[str, str] = {}
    labels: dict[str, int] = {}
    for row in FeatureStore(store_path).read():
        if not row.available:
            continue
        logits[row.clip_id][row.branch] = row.logit
        labels[row.clip_id] = row.label
        identity[row.clip_id] = row.source_identity
        quality[row.clip_id] = (
            row.face_coverage,
            row.audio_clipped,
            row.av_duration_delta_sec,
        )
    complete = sorted(k for k, v in logits.items() if len(v) == 3)
    return (
        complete,
        {"logits": logits, "quality": quality, "identity": identity},
        labels,
    )


def _scored(ids, data, labels, probabilities):
    """Bootstrap items: FakeAVCeleb puts many clips on one speaker, so the
    resampling unit is the source identity rather than the clip."""
    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class Item:
        label: int
        probability: float
        source_identity: str

    return [
        Item(labels[i], float(p), data["identity"][i])
        for i, p in zip(ids, probabilities, strict=True)
    ]


def _interval(items, *, seed: int, samples: int):
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    def statistic(rows):
        y = [r.label for r in rows]
        if len(set(y)) < 2:
            raise ValueError("Bootstrap sample does not contain both classes")
        return float(roc_auc_score(y, [r.probability for r in rows]))

    return cluster_bootstrap_interval(items, statistic, samples=samples, seed=seed)


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.fusion.late import FusionSample, LateFusion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--branch-seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    run = arguments.run_dir
    # Seed 17 wrote its stores without a suffix, and its out-of-fold store is
    # `oof.parquet`. The later seeds' stores carry the seed in the name.
    suffix = "" if arguments.branch_seed == 17 else f"-seed{arguments.branch_seed}"
    oof = "oof.parquet" if not suffix else f"store{suffix}.parquet"

    train_ids, train_data, train_labels = _assemble(run / "features" / oof)
    partitions = {}
    for name, stem in (("in-domain", "test"), ("dfdc", "dfdc")):
        path = run / "features" / f"{stem}{suffix}.parquet"
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
        tuple(c) for size in (1, 2, 3) for c in itertools.combinations(branches, size)
    ]

    results = []
    scores: dict[tuple[str, ...], dict[str, object]] = {}
    for combo in combinations:
        model = LateFusion(branch_names=combo).fit(
            samples(train_ids, train_data, combo),
            [train_labels[i] for i in train_ids],
        )
        row = {"branches": list(combo), "size": len(combo), "intervals": {}}
        scores[combo] = {}
        for name, (ids, data, labels) in partitions.items():
            probabilities = model.predict_proba(samples(ids, data, combo))
            scores[combo][name] = probabilities
            y = np.array([labels[i] for i in ids])
            if len(set(y)) != 2:
                row[name] = None
                continue
            row[name] = float(roc_auc_score(y, probabilities))
            # A bare AUC cannot say whether one subset beats another. The
            # headline claim is a comparison, so every row carries an interval
            # resampled over speakers.
            interval = _interval(
                _scored(ids, data, labels, probabilities),
                seed=arguments.seed,
                samples=arguments.bootstrap_samples,
            )
            row["intervals"][name] = {
                "lower": interval.lower,
                "upper": interval.upper,
            }
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

    # "Fusion beats the strongest single stream" is a paired comparison on the
    # same clips, so it is measured as one. Ranking the two AUCs answers a
    # different and easier question.
    from deepfake_detection.evaluation.bootstrap import (
        PairedPrediction,
        paired_auc_difference,
    )

    print()
    paired_report: dict[str, dict] = {}
    for name, (ids, data, labels) in partitions.items():
        scored_rows = [r for r in results if r[name] is not None]
        if not scored_rows:
            continue
        best = max(scored_rows, key=lambda r: r[name])
        all_three = next(r for r in results if r["size"] == 3)
        singles = [r for r in scored_rows if r["size"] == 1]
        beats_all = all(all_three[name] > s[name] for s in singles)
        print(
            f"{name}: best is {' + '.join(best['branches'])} at {best[name]:.4f}; "
            f"all-three {'beats' if beats_all else 'does NOT beat'} every single stream"
        )
        strongest = max(singles, key=lambda r: r[name])
        difference = paired_auc_difference(
            [
                PairedPrediction(
                    label=labels[i],
                    source_identity=data["identity"][i],
                    left_probability=float(a),
                    right_probability=float(b),
                )
                for i, a, b in zip(
                    ids,
                    scores[tuple(all_three["branches"])][name],
                    scores[tuple(strongest["branches"])][name],
                    strict=True,
                )
            ],
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        separated = difference.lower > 0 or difference.upper < 0
        paired_report[name] = {
            "strongest_single": strongest["branches"],
            "estimate": difference.estimate,
            "lower": difference.lower,
            "upper": difference.upper,
            "separated_from_zero": separated,
        }
        print(
            f"  all-three minus {' + '.join(strongest['branches'])}: "
            f"{difference.estimate:+.4f} "
            f"[{difference.lower:+.4f}, {difference.upper:+.4f}]"
            f"{' separated from zero' if separated else ' not separated from zero'}"
        )

    payload = {
        "branch_seed": arguments.branch_seed,
        "out_of_fold_clips": len(train_ids),
        "partitions": {n: len(p[0]) for n, p in partitions.items()},
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_seed": arguments.seed,
        "bootstrap_unit": "source_identity",
        "combinations": results,
        "raw_branch_auc": raw,
        "all_three_vs_strongest_single": paired_report,
    }
    out = arguments.output or run / "evaluation" / f"fusion-ablation{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
