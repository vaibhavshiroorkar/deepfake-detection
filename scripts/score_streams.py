"""Score every trained Design B stream, and leave behind what fusion needs.

Two jobs in one pass, because they read the same forward passes.

The first is the number that actually answers "is this stream any good".
Training records loss only, and a validation loss of 0.27 says nothing about
ranking: the visual branch reached 0.9742 ROC-AUC in-domain and 0.7583 on DFDC
from loss curves that looked the same. Loss is what training optimises; AUC is
what the objective is stated in.

The second is the feature store `ddf train fusion --model deep` reads. Exporting
the embeddings is the same forward pass as scoring the logits, so doing them
separately would double the GPU time for nothing.

Partitions are the held-out in-domain test set and DFDC, which is the only
audio-bearing corpus here not built on VoxCeleb2. They disagree, and the
disagreement is the finding rather than a nuisance.

    uv run python scripts/score_streams.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Scored:
    """One scored clip, carrying the identity the bootstrap clusters on."""

    logit: float
    label: int
    source_identity: str


def _auc_of(items) -> float:
    """ROC-AUC over a resampled set, for `cluster_bootstrap_interval`.

    A resample can land on one class even when the full set has both, and a
    bootstrap that raised there would lose the whole interval. Returning 0.5
    keeps the sample and says the draw carried no ranking information.
    """
    from sklearn.metrics import roc_auc_score

    labels = [item.label for item in items]
    if len(set(labels)) != 2:
        return 0.5
    return float(roc_auc_score(labels, [item.logit for item in items]))


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.fusion.store import FeatureStore
    from deepfake_detection.fusion.stream_export import export_stream_features
    from deepfake_detection.fusion.stream_loading import load_streams
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path("runs/program-20260906"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Resamples for the confidence interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Bootstrap seed, so an interval is reproducible.",
    )
    arguments = parser.parse_args(argv)

    audit = json.loads(
        (arguments.source_run / "cache-audit.json").read_text(encoding="utf-8")
    )
    preprocessing_hash = audit["preprocessing_hash"]

    print("trained streams:")
    specs = load_streams(arguments.run_dir, arguments.device, report=print)
    if not specs:
        print("nothing trained yet")
        return 1

    cache_index = {}
    with (arguments.source_run / "cache-index.csv").open(encoding="utf-8") as handle:
        import csv

        for row in csv.DictReader(handle):
            cache_index[row["clip_id"]] = arguments.source_run / row["cache_path"]
    cache_store = CacheStore(arguments.source_run / "cache")

    # `holdout` is the validation partition. The streams trained on
    # train-usable and have never seen it, so fusion can be fitted on these rows
    # without the test partition being touched. That is holdout stacking rather
    # than out-of-fold stacking: it uses less data but costs one training run
    # per stream instead of one per fold.
    partitions = {
        "holdout": (
            arguments.source_run / "split" / "val-usable.csv",
            "FakeAVCeleb",
            "holdout",
        ),
        "in-domain": (
            arguments.source_run / "split" / "test-usable.csv",
            "FakeAVCeleb",
            "test",
        ),
        "dfdc": (arguments.source_run / "split" / "dfdc-visual.csv", "DFDC", "external"),
    }

    summary: dict[str, dict[str, float | None]] = {}
    features = arguments.run_dir / "features"
    features.mkdir(parents=True, exist_ok=True)

    for partition, (manifest, dataset, role) in partitions.items():
        if not manifest.is_file():
            print(f"\n{partition}: no manifest at {manifest}, skipping")
            continue
        records = load_manifest(manifest, dataset=dataset).records
        store_path = features / f"{partition}.parquet"
        # A scoring pass rebuilds the store rather than adding to it. The store
        # rejects a duplicate key, so re-scoring after training one more stream
        # would otherwise fail on the streams already in the file.
        store_path.unlink(missing_ok=True)
        store = FeatureStore(store_path)
        print(f"\n{partition}: {len(records):,} clips -> {store_path}")
        report = export_stream_features(
            records=records,
            cache_index=cache_index,
            cache_store=cache_store,
            feature_store=store,
            streams=specs,
            split_hash=arguments.run_dir.name,
            preprocessing_hash=preprocessing_hash,
            partition_role=role,
            run_id=f"score-{partition}",
            device=arguments.device,
        )
        print(
            f"  {report.exported_rows:,} rows, "
            f"{report.unavailable_rows:,} abstained"
        )

        by_stream: dict[str, list[_Scored]] = {}
        for row in store.read():
            if row.available:
                by_stream.setdefault(row.branch, []).append(
                    _Scored(row.logit, row.label, row.source_identity)
                )
        for name, scored in sorted(by_stream.items()):
            labels = np.array([item.label for item in scored])
            # One class means a ranking metric is undefined, which MNW's
            # fake-only lab set is the standing example of. Say so rather than
            # inventing a number.
            if len(set(labels.tolist())) != 2:
                summary.setdefault(name, {})[partition] = None
                print(f"  {name:24} n/a (one class)   ({len(scored):,} clips)")
                continue

            interval = cluster_bootstrap_interval(
                scored,
                _auc_of,
                samples=arguments.bootstrap_samples,
                seed=arguments.seed,
            )
            summary.setdefault(name, {})[partition] = interval.estimate
            summary[name][f"{partition}_ci"] = [interval.lower, interval.upper]
            print(
                f"  {name:24} {interval.estimate:.4f} "
                f"[{interval.lower:.4f}, {interval.upper:.4f}]   "
                f"({len(scored):,} clips, {len(set(i.source_identity for i in scored)):,} identities)"
            )

    print("\nROC-AUC")
    columns = [p for p in partitions if any(p in row for row in summary.values())]
    header = f"{'stream':24} " + "  ".join(f"{p:>12}" for p in columns)
    print(header)
    print("-" * len(header))
    for name, scores in sorted(summary.items()):
        cells = "  ".join(
            f"{scores.get(p):12.4f}"
            if isinstance(scores.get(p), float)
            else f"{'n/a':>12}"
            for p in columns
        )
        print(f"{name:24} {cells}")

    output = arguments.run_dir / "evaluation" / "stream-scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
