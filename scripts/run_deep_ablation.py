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


def _score(model, rows, dims, device: str) -> float | None:
    """ROC-AUC of a fitted head over one partition, or None for one class."""
    import torch

    from deepfake_detection.training.fusion import _auc, as_tensors

    usable = [row for row in rows if row.branch_embeddings]
    if not usable:
        return None
    values, presence, labels = as_tensors(usable, dims, device)
    if len(set(labels.tolist())) < 2:
        return None
    model.eval()
    with torch.inference_mode():
        output = model(values, presence)
    return float(_auc(output.logit, labels))


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
                row[name] = _score(model, _rows(path, subset), dims, arguments.device)
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
            f"{row[n]:10.4f}" if row[n] is not None else f"{'n/a':>10}"
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
        print(
            f"{name}: best is {' + '.join(best['streams'])} at {best[name]:.4f}; "
            f"all streams {'beats' if beats else 'does NOT beat'} every single stream"
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
