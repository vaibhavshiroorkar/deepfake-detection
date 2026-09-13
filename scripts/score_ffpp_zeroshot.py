"""Score the FaceForensics++ model on corpora it has never seen.

This is the protocol behind nearly every published cross-dataset table: train on
FF++ c23, evaluate zero-shot on Celeb-DF-v2 and DFDC. It is the first number
this project produces that a reader can compare with someone else's, because
every earlier model was trained on FakeAVCeleb, which almost nobody uses as a
training corpus.

No re-caching is involved. FF++ was cached under the program run's code version,
so all four corpora share one `preprocessing_config_hash` and one cache store
layout. The hash is checked here rather than assumed: a mismatch means the model
is being fed views it was not trained on, and the resulting number would look
plausible and mean nothing.

FakeAVCeleb appears as a fourth partition on purpose. It is the corpus every
other model here was trained on, so it turns the usual comparison around and
asks what an FF++ model makes of it.

    uv run python scripts/score_ffpp_zeroshot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROGRAM = Path("runs/program-20260906")

# name -> (run directory holding the cache, manifest, dataset name)
PARTITIONS = {
    "ffpp-test": (
        Path("runs/ffpp-20260913"),
        Path("runs/ffpp-20260913/split/test-usable.csv"),
        "FaceForensics++",
    ),
    "celebdf": (
        PROGRAM,
        Path("runs/full-20260904/celebdf-test-manifest.csv"),
        "Celeb-DF-v2",
    ),
    "dfdc": (PROGRAM, PROGRAM / "split" / "dfdc-visual.csv", "DFDC"),
    "fakeavceleb-test": (
        PROGRAM,
        PROGRAM / "split" / "test-usable.csv",
        "FakeAVCeleb",
    ),
}


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/ffpp-20260913/checkpoints/visual-efficientnet.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    history_path = arguments.checkpoint.with_name(
        arguments.checkpoint.stem + "-history.json"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    expected_hash = history["metadata"]["preprocessing_hash"]
    model, _ = build_model("visual-efficientnet", history)
    load_checkpoint(arguments.checkpoint, model=model)
    model = model.to(arguments.device).eval()
    best = history["epochs"][history["best_epoch"] - 1]
    print(
        f"FF++ model: epoch {history['best_epoch']} of {len(history['epochs'])}, "
        f"validation AUC {best['validation_auc']:.4f}\n"
    )

    stores: dict[Path, tuple[dict[str, Path], object]] = {}
    results: dict[str, dict] = {}

    for name, (run_dir, manifest, dataset) in PARTITIONS.items():
        if not manifest.is_file():
            print(f"{name}: no manifest at {manifest}, skipping")
            continue
        if run_dir not in stores:
            from deepfake_detection.views.cache_store import CacheStore

            index: dict[str, Path] = {}
            with (run_dir / "cache-index.csv").open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    index[row["clip_id"]] = run_dir / row["cache_path"]
            stores[run_dir] = (index, CacheStore(run_dir / "cache"))
        index, store = stores[run_dir]

        # Two filters, and both are the abstention policy rather than
        # convenience. A clip with no cache entry was never built; a clip whose
        # entry carries no visual view had no stable face track, which is
        # exactly the case the protocol says to report rather than drop
        # silently. Coverage is reported beside the metric for that reason.
        listed = load_manifest(manifest, dataset=dataset).records
        records = []
        no_entry = 0
        no_view = 0
        for record in listed:
            path = index.get(record.clip_id)
            if path is None:
                no_entry += 1
                continue
            prepared = store.load(path, views=("visual_view",))
            if prepared.visual_view is None:
                no_view += 1
                continue
            if prepared.preprocessing_config_hash != expected_hash:
                raise SystemExit(
                    f"{name} was cached under "
                    f"{prepared.preprocessing_config_hash[:12]} but the model "
                    f"was trained under {expected_hash[:12]}"
                )
            records.append(record)
        if not records:
            print(f"{name}: no usable clip in {manifest.name}, skipping")
            continue
        data = CachedBranchDataset(
            records=records, cache_index=index, cache_store=store, branch="visual"
        )
        batches = DataLoader(data, batch_size=8, collate_fn=collate_branch_items)

        by_id = {record.clip_id: record for record in records}
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

        labels = {item.label for item in scored}
        if len(labels) != 2:
            print(f"{name}: one class only, no ranking metric")
            results[name] = {"clips": len(scored), "roc_auc": None}
            continue
        interval = cluster_bootstrap_interval(
            scored, _auc_of, samples=arguments.bootstrap_samples, seed=arguments.seed
        )
        results[name] = {
            "clips": len(scored),
            "listed": len(listed),
            "abstained_no_entry": no_entry,
            "abstained_no_view": no_view,
            "coverage": len(scored) / len(listed),
            "identities": len({item.source_identity for item in scored}),
            "roc_auc": interval.estimate,
            "ci": [interval.lower, interval.upper],
        }
        print(
            f"{name:18} {interval.estimate:.4f} "
            f"[{interval.lower:.4f}, {interval.upper:.4f}]   "
            f"{len(scored):,} of {len(listed):,} clips, "
            f"coverage {len(scored) / len(listed):.1%}"
        )

    output = (
        arguments.output
        or Path("runs/ffpp-20260913/evaluation/zero-shot.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"trained_on": "FaceForensics++ c23", "partitions": results},
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
