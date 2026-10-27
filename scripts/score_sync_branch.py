"""Score a trained synchronization branch, which nothing else could do.

`ddf evaluate branch` takes `visual` or `audio` only, so the sync branch had no
evaluation path at all. Three sync checkpoints were trained on seeds 17, 29 and
43 and produced no metrics, which means they moved no result row: a checkpoint
without a number is not evidence of anything.

The reason for the gap is real rather than an oversight. Sync is not a binary
classifier. It predicts which of seven audio offsets aligns a mouth crop with
its waveform, plus a mismatch class, and `CachedSyncDataset` exists to train
that: it takes authentic clips only and synthesises the offsets. Neither the
head nor the loader has a fake-or-real opinion.

What turns it into one is `sync_anomaly_logit`, which the fusion feature export
already uses: the log-sum-exp of every non-aligned class minus the aligned
class. High means "this clip does not look aligned". That scalar is what fusion
reads, so scoring it directly is measuring the same quantity fusion consumes
rather than inventing a new one.

**Labels.** Scored against `clip_fake`, matching the label the fusion feature
store writes on its sync rows. That is the clip-level question, not a
sync-specific cue label, and it is the wrong question for a
`FakeVideo-FakeAudio` clip that was re-synchronised after manipulation: such a
clip is fake and may be perfectly aligned. The number below is therefore a floor
on what the branch could do, and F19 is the standing warning about comparing it
against a differently labelled one.

**Validation.** Passing `--check-store` scores the clips through the feature
store's recorded logits as well and compares. They must agree: the store was
written by the same model through the same function, so a mismatch means this
script is not reproducing the pipeline.

Only for the seed the store was built from, which is seed 17. Pointing it at a
seed-29 or seed-43 checkpoint reports a disagreement, and that is the check
working: those are different models and their logits should differ.

    uv run python scripts/score_sync_branch.py --run-dir runs/program-20260906 \
        --checkpoint runs/program-20260906/checkpoints/final-sync-seed17.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Scored:
    clip_id: str
    label: int
    logit: float
    probability: float
    source_identity: str
    method: str


def _interval(items, *, seed: int, samples: int):
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    def statistic(rows):
        labels = [r.label for r in rows]
        if len(set(labels)) < 2:
            raise ValueError("Bootstrap sample does not contain both classes")
        return float(roc_auc_score(labels, [r.probability for r in rows]))

    return cluster_bootstrap_interval(items, statistic, samples=samples, seed=seed)


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch

    from deepfake_detection.branches.sync import build_sync_branch
    from deepfake_detection.branches.sync_objective import sync_anomaly_logit
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.evaluation.metrics import binary_metrics
    from deepfake_detection.training.checkpoints import load_checkpoint
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset", default="FakeAVCeleb")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--check-store",
        type=Path,
        default=None,
        help="A feature store whose recorded sync logits must match these.",
    )
    arguments = parser.parse_args(argv)
    run = arguments.run_dir

    manifest = arguments.manifest or (run / "split" / "test-sync-usable.csv")
    if not manifest.is_file():
        print(f"No manifest at {manifest}", file=sys.stderr)
        return 1
    if not arguments.checkpoint.is_file():
        print(f"No checkpoint at {arguments.checkpoint}", file=sys.stderr)
        return 1

    from deepfake_detection.data.cache_index import read_cache_index

    index = read_cache_index(run / "cache-index.csv")
    records = [
        record
        for record in load_manifest(manifest, dataset=arguments.dataset).records
        if record.clip_id in index
    ]
    if not records:
        print("No manifest rows resolve to a cache entry", file=sys.stderr)
        return 1

    model = build_sync_branch(pretrained=False)
    load_checkpoint(arguments.checkpoint, model=model)
    model.to(arguments.device).eval()
    store = CacheStore(run)

    scored: list[Scored] = []
    skipped = 0
    batch: list[tuple] = []

    def flush() -> None:
        if not batch:
            return
        video = torch.from_numpy(np.stack([item[1] for item in batch])).float()
        audio = torch.from_numpy(np.stack([item[2] for item in batch])).float()
        with torch.inference_mode():
            output = model(
                mouth_video=video.to(arguments.device),
                waveform=audio.to(arguments.device),
            )
            logits = sync_anomaly_logit(output.offset_logits).cpu().numpy()
        for (record, _, _), logit in zip(batch, logits, strict=True):
            scored.append(
                Scored(
                    clip_id=record.clip_id,
                    label=int(record.clip_fake),
                    logit=float(logit),
                    probability=float(1.0 / (1.0 + np.exp(-float(logit)))),
                    source_identity=record.source,
                    method=record.method,
                )
            )
        batch.clear()

    for record in records:
        prepared = store.load(
            index[record.clip_id], views=("sync_video_view", "sync_audio_view")
        )
        if prepared.sync_video_view is None or prepared.sync_audio_view is None:
            skipped += 1
            continue
        batch.append((record, prepared.sync_video_view, prepared.sync_audio_view))
        if len(batch) == arguments.batch_size:
            flush()
    flush()

    labels = [item.label for item in scored]
    print(
        f"scored {len(scored):,} clips "
        f"(fake {sum(labels)}, real {len(labels) - sum(labels)}), "
        f"abstained on {skipped} with no sync view"
    )
    if len(set(labels)) < 2:
        print("Single class, no ranking metric exists", file=sys.stderr)
        return 1

    probabilities = [item.probability for item in scored]
    metrics = binary_metrics(labels=labels, probabilities=probabilities, threshold=0.5)
    interval = _interval(
        scored, seed=arguments.seed, samples=arguments.bootstrap_samples
    )
    print(
        f"ROC-AUC {metrics.roc_auc:.4f} "
        f"[{interval.lower:.4f}, {interval.upper:.4f}]"
        f"{'  contains chance' if interval.lower <= 0.5 <= interval.upper else ''}"
    )
    print(
        f"balanced accuracy {metrics.balanced_accuracy:.4f}  "
        f"ECE {metrics.expected_calibration_error:.4f}"
    )

    store_check = None
    if arguments.check_store is not None and arguments.check_store.is_file():
        from deepfake_detection.fusion.store import FeatureStore

        recorded = {
            row.clip_id: row.logit
            for row in FeatureStore(arguments.check_store).read()
            if row.branch == "sync" and row.available
        }
        shared = [item for item in scored if item.clip_id in recorded]
        if shared:
            from sklearn.metrics import roc_auc_score

            mine = [item.logit for item in shared]
            theirs = [recorded[item.clip_id] for item in shared]
            shared_labels = [item.label for item in shared]
            store_check = {
                "clips": len(shared),
                "this_script_auc": float(roc_auc_score(shared_labels, mine)),
                "feature_store_auc": float(roc_auc_score(shared_labels, theirs)),
            }
            delta = abs(
                store_check["this_script_auc"] - store_check["feature_store_auc"]
            )
            # Compared on logits rather than probabilities: the sigmoid
            # saturates at large magnitudes and creates ties the raw scores do
            # not have.
            #
            # The tolerance is 1e-4 of ROC-AUC, not zero. Convolution kernels
            # are chosen by size at runtime, so the store's batch size and this
            # script's produce float32 results that differ in the last bits:
            # measured here, logits differ by at most 8.5e-4 and the AUC by
            # 8.5e-6. Demanding bit-equality would fail on arithmetic rather
            # than on logic. The measured delta is recorded either way, so the
            # threshold does not have to be trusted.
            store_check["auc_delta"] = delta
            store_check["max_logit_delta"] = float(
                max(abs(m - t) for m, t in zip(mine, theirs, strict=True))
            )
            store_check["tolerance"] = 1e-4
            store_check["agrees"] = delta < 1e-4
            print(
                f"\nfeature-store check on {len(shared):,} shared clips: "
                f"this script {store_check['this_script_auc']:.4f}, "
                f"store {store_check['feature_store_auc']:.4f} "
                f"({'agree' if store_check['agrees'] else 'DISAGREE'} "
                f"within {store_check['tolerance']:.0e}, "
                f"delta {store_check['auc_delta']:.2e})"
            )

    payload = {
        "branch": "sync",
        "dataset": arguments.dataset,
        "checkpoint": str(arguments.checkpoint).replace("\\", "/"),
        "manifest": str(manifest).replace("\\", "/"),
        "label": "clip_fake",
        "scope": (
            "Scored on the sync anomaly logit, the same scalar the fusion "
            "feature store records, against the clip-level label. A "
            "re-synchronised FakeVideo-FakeAudio clip is fake and may be "
            "aligned, so this is a floor rather than a cue-specific number."
        ),
        "coverage": {
            "manifest_rows": len(records),
            "scored_rows": len(scored),
            "abstained_rows": skipped,
        },
        "class_balance": {"fake": sum(labels), "real": len(labels) - sum(labels)},
        "metrics": {
            "roc_auc": metrics.roc_auc,
            "roc_auc_lower": interval.lower,
            "roc_auc_upper": interval.upper,
            "contains_chance": interval.lower <= 0.5 <= interval.upper,
            "pr_auc": metrics.pr_auc,
            "balanced_accuracy": metrics.balanced_accuracy,
            "brier": metrics.brier,
            "expected_calibration_error": metrics.expected_calibration_error,
        },
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_unit": "source_identity",
        "feature_store_check": store_check,
    }
    out = arguments.output or (
        run / "evaluation" / f"sync-{arguments.checkpoint.stem}-metrics.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions = arguments.predictions or (
        run / "evaluation" / f"sync-{arguments.checkpoint.stem}-predictions.csv"
    )
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clip_id", "label", "logit", "probability", "source", "method"]
        )
        for item in scored:
            writer.writerow(
                [
                    item.clip_id,
                    item.label,
                    f"{item.logit:.6f}",
                    f"{item.probability:.6f}",
                    item.source_identity,
                    item.method,
                ]
            )
    print(f"\nwrote {out}\nwrote {predictions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
