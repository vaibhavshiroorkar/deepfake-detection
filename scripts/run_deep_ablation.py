"""Does fusing streams beat the best single stream?

Objective 3's success metric, measured rather than assumed, for the Design B
feature-level head. Every subset of the available streams gets its own
`StreamFusion`, fitted on the same holdout rows and scored on the same two
partitions, so the only thing varying is which streams are in the input.

Two partitions because they disagree, and the disagreement is the result. The
held-out in-domain test set says one thing; DFDC, the only audio-bearing corpus
here not built on VoxCeleb2, says another. Reporting only the first answers the
easier question.

Fitted on `holdout` rows: the streams trained on the training partition and have
never seen validation, so those rows are leakage-free without cross-fitting
every stream per fold.

    uv run python scripts/run_deep_ablation.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _rows(path: Path, streams: tuple[str, ...]):
    """Assembled rows restricted to `streams`, keeping partial coverage."""
    import dataclasses

    from deepfake_detection.fusion.store import FeatureStore

    assembled = FeatureStore(path).assemble(required_branches=streams, strict=False)
    # Drop the streams this subset excludes, so the head is built for exactly
    # the input being ablated rather than carrying dead columns.
    return [
        dataclasses.replace(
            row,
            branch_embeddings={
                k: v for k, v in row.branch_embeddings.items() if k in streams
            },
        )
        for row in assembled
    ]


@dataclass(frozen=True, slots=True)
class _Scored:
    """One fused clip, carrying the identity the bootstrap clusters on."""

    logit: float
    label: int
    source_identity: str


def _auc_of(items) -> float:
    """ROC-AUC over a resample. 0.5 when a draw lands on one class, so one
    degenerate sample does not discard the whole interval."""
    import torch

    from deepfake_detection.training.fusion import _auc

    labels = [item.label for item in items]
    if len(set(labels)) != 2:
        return 0.5
    return float(
        _auc(
            torch.tensor([item.logit for item in items]),
            torch.tensor([float(item.label) for item in items]),
        )
    )


def _score(model, rows, dims, device: str, *, samples: int, seed: int):
    """ROC-AUC with a confidence interval, or None when one class is present.

    Clustered on `source_identity`, not on the clip. The question a reader has
    is what happens on a different set of speakers, not on a different draw of
    clips from these speakers. On DFDC that happens to give a narrower interval
    than a clip-level bootstrap, 0.0776 against 0.1065, because identities there
    carry balanced class proportions and resampling whole identities preserves
    the ratio. Narrower is not the reason to choose it; matching the question is.
    """
    import torch

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.training.fusion import as_tensors

    usable = [row for row in rows if row.branch_embeddings]
    if not usable:
        return None
    values, presence, labels = as_tensors(usable, dims, device)
    if len(set(labels.tolist())) < 2:
        return None
    model.eval()
    with torch.inference_mode():
        output = model(values, presence)
    logits = output.logit.cpu().tolist()
    scored = [
        _Scored(logit, int(row.label), row.source_identity)
        for logit, row in zip(logits, usable, strict=False)
    ]
    return cluster_bootstrap_interval(scored, _auc_of, samples=samples, seed=seed)


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.training.fusion import (
        fit_stream_fusion,
        stream_dimensions,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Resamples per interval. 31 subsets times two partitions, so this "
        "is the run's cost driver.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    features = arguments.run_dir / "features"
    train_path = features / "holdout.parquet"
    if not train_path.is_file():
        print(f"No holdout store at {train_path}. Run scripts/score_streams.py first.")
        return 1

    from deepfake_detection.fusion.store import FeatureStore

    available = sorted(
        {row.branch for row in FeatureStore(train_path).read() if row.available}
    )
    print(f"streams: {', '.join(available)}")

    partitions = {
        name: features / f"{name}.parquet"
        for name in ("in-domain", "dfdc")
        if (features / f"{name}.parquet").is_file()
    }
    print(f"partitions: {', '.join(partitions)}\n")

    results = []
    for size in range(1, len(available) + 1):
        for subset in itertools.combinations(available, size):
            train_rows = _rows(train_path, subset)
            dims = stream_dimensions(train_rows)
            model, history = fit_stream_fusion(
                rows=train_rows,
                epochs=arguments.epochs,
                seed=arguments.seed,
                device=arguments.device,
                # One pattern: every stream in the subset present. The question
                # here is whether the combination helps, not how it behaves when
                # a stream is missing, and mixing presence patterns in would
                # change what is being compared between rows of the table.
                presence_patterns={"video": 1.0},
            )
            row = {"streams": list(subset), "size": size,
                   "train_clips": history.train_clips}
            for name, path in partitions.items():
                interval = _score(
                    model,
                    _rows(path, subset),
                    dims,
                    arguments.device,
                    samples=arguments.bootstrap_samples,
                    seed=arguments.seed,
                )
                row[name] = None if interval is None else interval.estimate
                if interval is not None:
                    row[f"{name}_ci"] = [interval.lower, interval.upper]
            results.append(row)
            cells = "  ".join(
                f"{name} {row[name]:.4f}" if row[name] is not None else f"{name} n/a"
                for name in partitions
            )
            print(f"  {' + '.join(subset):58} {cells}")

    print()
    width = max(len(" + ".join(r["streams"])) for r in results)
    header = f"{'streams':{width}}  " + "  ".join(f"{n:>10}" for n in partitions)
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: (r["size"], r["streams"])):
        cells = "  ".join(
            (
                f"{row[n]:.4f} [{row[n + '_ci'][0]:.4f}, {row[n + '_ci'][1]:.4f}]"
                if row.get(f"{n}_ci")
                else (f"{row[n]:.4f}" if row[n] is not None else "n/a")
            )
            for n in partitions
        )
        print(f"{' + '.join(row['streams']):{width}}  {cells}")

    print()
    for name in partitions:
        scored = [r for r in results if r[name] is not None]
        if not scored:
            continue
        best = max(scored, key=lambda r: r[name])
        singles = [r for r in scored if r["size"] == 1]
        full = next((r for r in scored if r["size"] == len(available)), None)
        beats = (
            full is not None
            and singles
            and all(full[name] > s[name] for s in singles)
        )
        # Two claims, because they are not the same claim. "Beats" compares
        # point estimates; "separated" asks whether the best combination's
        # interval clears the best single stream's estimate. A table of
        # overlapping intervals supports the first and not the second, and the
        # second is the one a reader should be given.
        best_single = max(singles, key=lambda r: r[name])
        interval = best.get(f"{name}_ci")
        separated = (
            interval is not None
            and best["size"] > 1
            and interval[0] > best_single[name]
        )
        print(
            f"{name}: best is {' + '.join(best['streams'])} at {best[name]:.4f}"
            + (f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "")
            + f"; best single is {' + '.join(best_single['streams'])} at "
            f"{best_single[name]:.4f}"
        )
        print(
            f"   all streams {'beats' if beats else 'does NOT beat'} every single "
            f"stream; best combination is "
            f"{'separated from' if separated else 'NOT separated from'} the best "
            f"single stream at 95 percent"
        )

    out = arguments.output or arguments.run_dir / "evaluation" / "deep-ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"combinations": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
