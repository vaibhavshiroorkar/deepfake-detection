"""Per-family ROC-AUC for a FaceForensics++ model, with intervals.

An aggregate number over six manipulation families hides the thing that matters.
A detector at 0.87 overall might be 0.99 on one family and 0.55 on another, and
only the second number tells you what happens when the next family arrives. The
same pattern is documented in the wild: a 2026 evaluation found one detector at
69.8 percent on one generator and 15.4 percent on another.

Each family is scored against the same real clips, so the rows are comparable to
each other and each is a proper two-class problem.

This is the "seen family" baseline. Pair it with `scripts/train_family_holdout.sh`,
which trains one model per held-out family, and the difference between the two
tables for the same family is the generalization measurement.

    uv run python scripts/score_by_family.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Scored:
    score: float
    label: int
    source_identity: str


def _auc_of(items) -> float:
    from sklearn.metrics import roc_auc_score

    labels = [item.label for item in items]
    if len(set(labels)) != 2:
        return 0.5
    return float(roc_auc_score(labels, [item.score for item in items]))


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.fusion.stream_loading import build_model
    from deepfake_detection.training.checkpoints import load_checkpoint
    from deepfake_detection.views.cache_store import CacheStore
    from deepfake_detection.views.equivalence import same_preprocessing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ffpp-20260913"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the run's visual-efficientnet checkpoint.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--label", default="seen", help="Row label in the output.")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    checkpoint = arguments.checkpoint or (
        arguments.run_dir / "checkpoints" / "visual-efficientnet.pt"
    )
    manifest = arguments.manifest or (
        arguments.run_dir / "split" / "test-usable.csv"
    )
    history = json.loads(
        checkpoint.with_name(checkpoint.stem + "-history.json").read_text(
            encoding="utf-8"
        )
    )
    expected_hash = history["metadata"]["preprocessing_hash"]
    model, _ = build_model("visual-efficientnet", history)
    load_checkpoint(checkpoint, model=model)
    model = model.to(arguments.device).eval()

    index: dict[str, Path] = {}
    with (arguments.run_dir / "cache-index.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index[row["clip_id"]] = arguments.run_dir / row["cache_path"]
    store = CacheStore(arguments.run_dir / "cache")

    records = []
    for record in load_manifest(manifest, dataset="FaceForensics++").records:
        path = index.get(record.clip_id)
        if path is None:
            continue
        prepared = store.load(path, views=("visual_view",))
        if prepared.visual_view is None:
            continue
        if not same_preprocessing(
            prepared.preprocessing_config_hash, expected_hash
        ):
            raise SystemExit(
                f"Cache hash {prepared.preprocessing_config_hash[:12]} does not "
                f"match the checkpoint's {expected_hash[:12]}"
            )
        records.append(record)
    if not records:
        print(f"No usable clip in {manifest}")
        return 1

    by_id = {record.clip_id: record for record in records}
    data = CachedBranchDataset(
        records=records, cache_index=index, cache_store=store, branch="visual"
    )
    batches = DataLoader(data, batch_size=8, collate_fn=collate_branch_items)

    scored: list[_Scored] = []
    with torch.inference_mode():
        for batch in batches:
            logit, _ = model(batch.values.to(arguments.device))
            for clip_id, value, label in zip(
                batch.clip_ids, logit.cpu().tolist(), batch.labels.tolist(),
                strict=True,
            ):
                scored.append(
                    _Scored(float(value), int(label), by_id[clip_id].source)
                )

    # The same real clips for every family, so the rows compare with each other
    # rather than each carrying its own baseline.
    real = [item for item in scored if item.label == 0]
    families: dict[str, list[_Scored]] = {}
    for item, record in zip(scored, records, strict=True):
        if record.clip_fake:
            families.setdefault(record.method, []).append(item)

    print(f"{len(real):,} real clips shared across {len(families)} families\n")
    results: dict[str, dict] = {}
    for family, fakes in sorted(families.items()):
        interval = cluster_bootstrap_interval(
            real + fakes,
            _auc_of,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        results[family] = {
            "roc_auc": interval.estimate,
            "ci": [interval.lower, interval.upper],
            "fake_clips": len(fakes),
            "real_clips": len(real),
        }
        print(
            f"{family:28} {interval.estimate:.4f} "
            f"[{interval.lower:.4f}, {interval.upper:.4f}]   {len(fakes):,} fakes"
        )

    output = (
        arguments.output
        or arguments.run_dir / "evaluation" / f"by-family-{arguments.label}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"condition": arguments.label, "families": results},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
