from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import joblib

from deepfake_detection.benchmarks.detector_annotations import (
    annotation_audit_sha256,
    read_annotations,
    validate_annotations,
)
from deepfake_detection.benchmarks.detector_metrics import (
    BoundDetectorReport,
    DetectorDecision,
    compare_detectors,
    read_bound_detector_report,
)
from deepfake_detection.benchmarks.detector_runner import run_detector_benchmark
from deepfake_detection.benchmarks.detector_sample import (
    MINIMUM_REVIEW_CLIPS,
    MINIMUM_REVIEW_FRAMES,
    ReviewFrame,
    build_review_sample,
    read_review_sample,
    review_sample_sha256,
    write_review_sample,
)
from deepfake_detection.data.cache_build import build_cache
from deepfake_detection.data.guards import (
    reject_evaluation_only,
    reject_evaluation_only_datasets,
)
from deepfake_detection.data.manifest import load_manifest, write_manifest
from deepfake_detection.data.protocols import (
    audit_split,
    build_method_holdout_protocol,
    build_source_split,
    identity_strict_subset,
    split_hash,
    stratified_subsample,
)
from deepfake_detection.evaluation.bootstrap import (
    PairedPrediction,
    bootstrap_binary_metrics,
    paired_auc_difference,
)
from deepfake_detection.evaluation.metrics import (
    EvaluationItem,
    evaluate_items,
    per_method_metrics,
    select_balanced_accuracy_threshold,
    subgroup_metrics,
)
from deepfake_detection.experiments import (
    NullRunLogger,
    execute_configured_run,
    run_fusion_smoke,
    runtime,
)
from deepfake_detection.experiments.runner import _CONFIGURED_RUN_SENTINEL
from deepfake_detection.experiments.runtime import capture_runtime
from deepfake_detection.experiments.scopes import validate_evidence_scope
from deepfake_detection.experiments.training_log import (
    log_binary_training,
    log_detector_benchmark,
    log_fusion_training,
    log_sync_training,
)
from deepfake_detection.fusion.late import FusionArtifact, FusionSample, LateFusion
from deepfake_detection.fusion.store import FeatureStore
from deepfake_detection.inference.loading import (
    InferenceConfig,
    build_preprocessor,
    load_prediction_engine,
)
from deepfake_detection.training.crossfit import build_group_folds
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.media import FFmpegMediaDecoder
from deepfake_detection.views.model_assets import fetch_yunet_model


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_build(arguments: argparse.Namespace) -> int:
    result = load_manifest(arguments.input, dataset=arguments.dataset)
    write_manifest(result.records, arguments.output)
    _write_json(
        arguments.audit,
        {
            "records": len(result.records),
            "quarantined_paths": [str(path) for path in result.quarantined_paths],
        },
    )
    return 0


def _manifest_from_meta(arguments: argparse.Namespace) -> int:
    """Build a manifest straight from a FakeAVCeleb meta_data.csv.

    `manifest build` validates a manifest that already exists. Producing the
    first one from a raw drop used to be an unrecorded manual step, which is why
    no script for the original 2,000-row manifest survives in the repo. The
    conversion itself is data/meta.py, already used by the dashboard; this only
    puts a command in front of it and runs the result through the same
    quarantine pass every other manifest gets.
    """
    import pandas as pd

    from deepfake_detection.data.meta import manifest_from_meta

    meta = pd.read_csv(arguments.meta)
    frame = manifest_from_meta(
        meta,
        root=arguments.root,
        data_dir=arguments.data_dir,
        require_exists=not arguments.allow_missing,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)

    result = load_manifest(arguments.output, dataset=arguments.dataset)
    write_manifest(result.records, arguments.output)
    _write_json(
        arguments.audit,
        {
            "meta_rows": len(meta),
            "records": len(result.records),
            "dropped_missing_media": len(meta) - len(frame),
            "quarantined_paths": [str(path) for path in result.quarantined_paths],
        },
    )
    return 0


def _handoff_update(arguments: argparse.Namespace) -> int:
    """Regenerate the handoff's generated blocks from the working tree.

    Called after every stage of a long run, so the handoff describes the state
    that actually exists rather than the state someone last remembered to
    write down.
    """
    from deepfake_detection.documentation.handoff import update_handoff

    changed = update_handoff(arguments.root, arguments.path)
    print(f"{arguments.path}: {'updated' if changed else 'already current'}")
    return 0


# What each branch or stream must find in a cache entry to be usable. The
# audiovisual streams need both halves of a pair: a clip with mouth crops but no
# audio track, which is every MNW clip, can feed the sync branch's video encoder
# but has nothing for a cross-modal comparison to read.
BRANCH_VIEWS = {
    "visual": ("visual_view",),
    "audio": ("audio_view",),
    "sync": ("sync_video_view",),
    "lipsync": ("sync_video_view", "sync_audio_view"),
    "emotion": ("visual_view", "audio_view"),
}


def _partition_by_view(
    records,
    cache_index,
    cache_store,
    *,
    branch: str,
    preprocessing_hash: str | None,
):
    """Split records into those a branch can score and those it must abstain on.

    A clip is unusable when it was never cached, when its cache entry was built
    by different preprocessing, or when the view this branch reads is absent.
    The last case is the pipeline's abstention policy rather than a fault: a
    clip whose primary face track was unstable gets no visual view, because
    substituting a full-frame crop would bias the evaluation.

    Returns (usable, abstained) where abstained maps clip_id to a reason.
    """
    required = BRANCH_VIEWS[branch]
    usable = []
    abstained: dict[str, str] = {}
    for record in records:
        path = cache_index.get(record.clip_id)
        if path is None:
            abstained[record.clip_id] = "not_cached"
            continue
        try:
            views = cache_store.available_views(path)
            metadata = cache_store.load_metadata(path)
        except (OSError, ValueError, KeyError) as error:
            abstained[record.clip_id] = f"unreadable_cache: {error}"
            continue
        if (
            preprocessing_hash is not None
            and metadata.get("preprocessing_config_hash") != preprocessing_hash
        ):
            abstained[record.clip_id] = "preprocessing_hash_mismatch"
            continue
        missing = [name for name in required if name not in views]
        if missing:
            abstained[record.clip_id] = "no_" + "_and_".join(missing)
            continue
        usable.append(record)
    return usable, abstained


def _manifest_usable(arguments: argparse.Namespace) -> int:
    """Write the subset of a manifest that one branch can actually read.

    Training and evaluation both crash on a clip whose view is missing, so this
    filter used to live in an untracked one-off script pinned to a single run
    directory, a single hash and a single branch. Losing it is how a rebuilt
    cache turns into a mid-epoch exception.

    The dropped clips are not silently discarded: the audit records each one and
    its reason, so the abstention rate stays reportable rather than vanishing
    from the denominator.
    """
    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    cache_index = _read_cache_index(arguments.cache_index)
    usable, abstained = _partition_by_view(
        records,
        cache_index,
        CacheStore(arguments.cache_root),
        branch=arguments.branch,
        preprocessing_hash=arguments.preprocessing_hash,
    )
    write_manifest(usable, arguments.output)

    reasons: dict[str, int] = {}
    for reason in abstained.values():
        key = reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    labels = {record.video_fake for record in usable}
    _write_json(
        arguments.audit,
        {
            "branch": arguments.branch,
            "dataset": arguments.dataset,
            "input_rows": len(records),
            "usable_rows": len(usable),
            "abstained_rows": len(abstained),
            "abstention_rate": (
                len(abstained) / len(records) if records else 0.0
            ),
            "abstained_by_reason": dict(sorted(reasons.items())),
            "abstained_clip_ids": sorted(abstained),
            "both_classes_present": labels == {True, False},
        },
    )
    if not usable:
        raise ValueError(
            f"No clip in {arguments.manifest} has a {arguments.branch} view"
        )
    return 0


def _smoke(arguments: argparse.Namespace) -> int:
    run_fusion_smoke(
        arguments.output_dir,
        seed=arguments.seed,
        samples=arguments.samples,
        logger=getattr(arguments, "_run_logger", NullRunLogger()),
    )
    return 0


def _split_build(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="split building")
    result = load_manifest(arguments.manifest, dataset=arguments.dataset)
    split = build_source_split(result.records, seed=arguments.seed)
    strict = identity_strict_subset(split)
    audit = audit_split(split)
    for name, records in split.items():
        write_manifest(records, arguments.output_dir / f"{name}.csv")
        write_manifest(
            strict[name], arguments.output_dir / f"{name}-identity-strict.csv"
        )
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "seed": arguments.seed,
            "split_hash": split_hash(split),
            "rows": {name: len(records) for name, records in split.items()},
            "strict_rows": {name: len(records) for name, records in strict.items()},
            "source_overlaps": {
                f"{left}:{right}": sorted(values)
                for (left, right), values in audit.source_overlaps.items()
            },
            "all_identity_overlaps": {
                f"{left}:{right}": sorted(values)
                for (left, right), values in audit.all_identity_overlaps.items()
            },
            "method_counts": audit.method_counts,
        },
    )
    return 0


def _split_crossfit(arguments: argparse.Namespace) -> int:
    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    folds = build_group_folds(records, folds=arguments.folds, seed=arguments.seed)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, fold in enumerate(folds):
        train = tuple(records[row] for row in fold.train_indices)
        holdout = tuple(records[row] for row in fold.holdout_indices)
        write_manifest(train, arguments.output_dir / f"fold-{index}-train.csv")
        write_manifest(holdout, arguments.output_dir / f"fold-{index}-holdout.csv")
        summaries.append(
            {
                "fold": index,
                "train_sources": len(fold.train_sources),
                "holdout_sources": len(fold.holdout_sources),
                "train_rows": len(train),
                "holdout_rows": len(holdout),
            }
        )
    _write_json(
        arguments.output_dir / "crossfit-audit.json",
        {"seed": arguments.seed, "folds": summaries},
    )
    return 0


def _split_method_holdout(arguments: argparse.Namespace) -> int:
    split = {
        name: load_manifest(
            arguments.split_dir / f"{name}.csv",
            dataset=arguments.dataset,
        ).records
        for name in ("train", "val", "test")
    }
    protocol = build_method_holdout_protocol(
        split,
        heldout_methods=set(arguments.methods),
    )
    for name, records in protocol.items():
        write_manifest(records, arguments.output_dir / f"{name}.csv")
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "heldout_methods": sorted(arguments.methods),
            "rows": {name: len(records) for name, records in protocol.items()},
        },
    )
    return 0


def _split_subsample(arguments: argparse.Namespace) -> int:
    """Thin an existing frozen split without rebuilding or rehashing it.

    The pilot tier and the full tier have to stay comparable, so they share one
    split and one split hash. Only the number of rows fed to training changes.
    """
    rows = {}
    for name in ("train", "val", "test"):
        records = load_manifest(
            arguments.split_dir / f"{name}.csv", dataset=arguments.dataset
        ).records
        sampled = stratified_subsample(
            records, seed=arguments.seed, fake_ratio=arguments.fake_ratio
        )
        write_manifest(sampled, arguments.output_dir / f"{name}.csv")
        rows[name] = {
            "rows": len(sampled),
            "real": sum(1 for record in sampled if not record.video_fake),
            "fake": sum(1 for record in sampled if record.video_fake),
            "methods": len({record.method for record in sampled}),
        }
    _write_json(
        arguments.output_dir / "audit.json",
        {
            "seed": arguments.seed,
            "fake_ratio": arguments.fake_ratio,
            "source_split_dir": str(arguments.split_dir),
            "rows": rows,
        },
    )
    return 0


def _cache_build(arguments: argparse.Namespace) -> int:
    result = load_manifest(arguments.manifest, dataset=arguments.dataset)
    preprocessor = build_preprocessor(
        code_version=arguments.code_version,
        device=arguments.device,
        detector=arguments.detector,
        tracker=arguments.tracker,
        crop_mode=arguments.crop_mode,
        model_path=arguments.model_path,
        expected_model_hash=arguments.expected_model_hash,
        remove_leading_silence=not arguments.keep_leading_silence,
    )
    report = build_cache(
        records=result.records,
        dataset_root=arguments.dataset_root,
        preprocessor=preprocessor,
        cache_store=CacheStore(arguments.cache_root),
        skip_cached=arguments.skip_cached,
        shard=_parse_shard(arguments.shard),
    )
    arguments.index.parent.mkdir(parents=True, exist_ok=True)
    with arguments.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "cache_path"))
        writer.writeheader()
        for clip_id, cache_path in sorted(report.cache_index.items()):
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "cache_path": str(
                        _cache_index_value(cache_path, index=arguments.index)
                    ),
                }
            )
    _write_json(
        arguments.audit,
        {
            "succeeded": report.succeeded,
            "failed": report.failed,
            "skipped": report.skipped,
            "full_fusion_ready": report.full_fusion_ready,
            "blocker_counts": report.blocker_counts,
            "preprocessing_hash": report.preprocessing_hash,
            "failures": report.failures,
        },
    )
    return 2 if report.failed else 0


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    index, _, count = value.partition("/")
    try:
        shard = (int(index), int(count))
    except ValueError as error:
        raise ValueError(f"Shard must look like I/N, got {value!r}") from error
    return shard


def _cache_merge(arguments: argparse.Namespace) -> int:
    """Fold per-shard index and audit files into one of each.

    Sharding is what makes a 21,544-clip build finish in hours rather than a day
    and a half, but each worker can only write its own slice. Training reads one
    index, so the shards have to be reassembled before it can run.
    """
    merged_index: dict[str, Path] = {}
    for path in arguments.indexes:
        for clip_id, cache_path in _read_cache_index(path).items():
            merged_index[clip_id] = cache_path

    succeeded = failed = skipped = full_fusion_ready = 0
    blocker_counts: Counter[str] = Counter()
    failures: dict[str, str] = {}
    hashes: set[str] = set()
    for path in arguments.audits:
        audit = json.loads(path.read_text(encoding="utf-8"))
        succeeded += int(audit["succeeded"])
        failed += int(audit["failed"])
        skipped += int(audit.get("skipped", 0))
        full_fusion_ready += int(audit["full_fusion_ready"])
        blocker_counts.update(audit.get("blocker_counts", {}))
        failures.update(audit.get("failures", {}))
        if audit.get("preprocessing_hash"):
            hashes.add(audit["preprocessing_hash"])
    if len(hashes) > 1:
        raise ValueError(f"Shards disagree on the preprocessing hash: {sorted(hashes)}")

    arguments.index.parent.mkdir(parents=True, exist_ok=True)
    with arguments.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "cache_path"))
        writer.writeheader()
        for clip_id, cache_path in sorted(merged_index.items()):
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "cache_path": str(
                        _cache_index_value(cache_path, index=arguments.index)
                    ),
                }
            )
    _write_json(
        arguments.audit,
        {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "full_fusion_ready": full_fusion_ready,
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "preprocessing_hash": next(iter(hashes)) if hashes else None,
            "failures": failures,
            "shards": len(arguments.audits),
        },
    )
    return 2 if failed else 0


def _evaluate_branch(arguments: argparse.Namespace) -> int:
    """Score one trained branch on any manifest, in-domain or cross-dataset.

    This replaces the two near-identical `evaluate_visual.py` scripts under
    runs/. They hardcoded one checkpoint directory, one validation manifest and
    one dataset name, so every new evaluation target meant another copy. Cross-
    dataset work needs the manifest, the cache and the dataset name to be
    arguments, and needs the run recorded through the tracked path like every
    other run rather than by a direct mlflow call.
    """
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.training.checkpoints import load_checkpoint

    validate_evidence_scope(arguments.evidence_scope)
    manifest_records = load_manifest(
        arguments.manifest, dataset=arguments.dataset
    ).records
    cache_index = _read_cache_index(arguments.cache_index)
    model = _build_branch_model(arguments)
    state = load_checkpoint(arguments.checkpoint, model=model)
    preprocessing_hash = (
        arguments.preprocessing_hash or state.metadata.preprocessing_hash
    )

    # Clips the branch cannot read are abstained on, not dropped. A clip with an
    # unstable primary face track has no visual view, and on MNW that is 46 of
    # 131. Feeding them to the loader raises mid-epoch; removing them silently
    # would flatter the result by shrinking the denominator.
    records, abstained = _partition_by_view(
        manifest_records,
        cache_index,
        CacheStore(arguments.cache_root),
        branch=arguments.branch,
        preprocessing_hash=preprocessing_hash,
    )
    if not records:
        raise ValueError(
            f"No clip in {arguments.manifest} has a {arguments.branch} view"
        )

    dataset = CachedBranchDataset(
        records=records,
        cache_index=cache_index,
        cache_store=CacheStore(arguments.cache_root),
        branch=arguments.branch,
        preprocessing_hash=preprocessing_hash,
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    model.to(arguments.device).eval()

    by_clip = {record.clip_id: record for record in records}
    rows: list[dict[str, object]] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch.values.to(arguments.device)).logits
            loss_sum += float(
                functional.binary_cross_entropy_with_logits(
                    logits,
                    batch.labels.to(arguments.device),
                    reduction="sum",
                )
            )
            probabilities = torch.sigmoid(logits).cpu().tolist()
            labels = batch.labels.int().tolist()
            for clip_id, label, probability in zip(
                batch.clip_ids, labels, probabilities, strict=True
            ):
                # Keyed on clip_id rather than zipped positionally against the
                # manifest, so a reordered or filtered loader cannot silently
                # pair a score with the wrong clip's metadata.
                record = by_clip[clip_id]
                rows.append(
                    {
                        "clip_id": clip_id,
                        "label": label,
                        "probability": probability,
                        "predicted": int(probability >= arguments.threshold),
                        "source": record.source,
                        "manipulation_type": record.manipulation_type,
                        "method": record.method,
                    }
                )
    if not rows:
        raise ValueError("Evaluation produced no rows")

    report = _branch_evaluation_report(arguments, rows, state, preprocessing_hash)
    report["loss"] = loss_sum / len(rows)
    report["coverage"] = _coverage(manifest_records, rows, abstained)
    _write_json(arguments.output, report)
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    with arguments.predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger = getattr(arguments, "_run_logger", NullRunLogger())
    logger.log_params(
        {
            "evaluation.branch": arguments.branch,
            "evaluation.dataset": arguments.dataset,
            "evaluation.rows": len(rows),
            "evaluation.threshold": arguments.threshold,
            "evaluation.checkpoint_sha256": report["checkpoint_sha256"],
        }
    )
    logger.log_metrics(
        {f"evaluation.{name}": float(value) for name, value in _flat_metrics(report)}
    )
    logger.log_artifact(arguments.output, artifact_path="evaluation")
    logger.log_artifact(arguments.predictions, artifact_path="evaluation")
    return 0


def _coverage(manifest_records, rows, abstained: dict[str, str]) -> dict[str, object]:
    """How much of the manifest was actually scored, and why the rest was not.

    docs/data-card.md requires the abstention rate to be reported rather than
    the abstained clips being deleted from the denominator, because a detector
    that silently declines the hard clips looks better than it is.
    """
    reasons: dict[str, int] = {}
    for reason in abstained.values():
        key = reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    total = len(manifest_records)
    return {
        "manifest_rows": total,
        "scored_rows": len(rows),
        "abstained_rows": len(abstained),
        "abstention_rate": len(abstained) / total if total else 0.0,
        "abstained_by_reason": dict(sorted(reasons.items())),
    }


def _build_branch_model(arguments: argparse.Namespace):
    from deepfake_detection.branches.audio import build_wav2vec2_audio_branch
    from deepfake_detection.branches.visual import build_efficientnet_b0

    if arguments.branch == "visual":
        return build_efficientnet_b0(pretrained=False)
    return build_wav2vec2_audio_branch(
        model_name=arguments.audio_model,
        pretrained=False,
    )


# Below this many rows in the smaller class, a ranking metric is noise.
MINIMUM_CLASS_ROWS = 10


def _branch_evaluation_report(
    arguments: argparse.Namespace,
    rows: list[dict[str, object]],
    state,
    preprocessing_hash: str,
) -> dict[str, object]:
    from deepfake_detection.evaluation.metrics import binary_metrics

    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    confusion = {
        "true_positive": sum(
            row["label"] == 1 and row["predicted"] == 1 for row in rows
        ),
        "true_negative": sum(
            row["label"] == 0 and row["predicted"] == 0 for row in rows
        ),
        "false_positive": sum(
            row["label"] == 0 and row["predicted"] == 1 for row in rows
        ),
        "false_negative": sum(
            row["label"] == 1 and row["predicted"] == 0 for row in rows
        ),
    }
    report: dict[str, object] = {
        "branch": arguments.branch,
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "checkpoint_run_id": state.metadata.run_id,
        "dataset": arguments.dataset,
        "evidence_scope": arguments.evidence_scope,
        "fixed_threshold": arguments.threshold,
        "rows": len(rows),
        "confusion": confusion,
        "preprocessing_hash": preprocessing_hash,
        "trained_on_split_hash": state.metadata.split_hash,
        "per_method": _grouped_rates(rows, "method"),
        "per_manipulation_type": _grouped_rates(rows, "manipulation_type"),
    }
    positives = sum(labels)
    negatives = len(labels) - positives
    report["class_balance"] = {
        "fake": positives,
        "real": negatives,
        # A ranking metric needs enough of both classes to mean anything. MNW
        # scores 84 forgeries against 1 genuine clip, which yields an ROC-AUC
        # that is arithmetically defined and statistically worthless. Saying so
        # in the record is the difference between a caveat and a false result.
        "ranking_metrics_reliable": min(positives, negatives) >= MINIMUM_CLASS_ROWS,
        "minimum_rows_per_class": MINIMUM_CLASS_ROWS,
    }
    if set(labels) == {0, 1}:
        report["metrics"] = asdict(
            binary_metrics(
                labels=labels,
                probabilities=probabilities,
                threshold=arguments.threshold,
            )
        )
        if min(positives, negatives) < MINIMUM_CLASS_ROWS:
            report["class_balance"]["note"] = (
                f"Only {min(positives, negatives)} rows in the smaller class. "
                "Ranking metrics such as ROC-AUC are reported but must not be "
                "quoted as a result. Read the detection rate instead."
            )
        # The detection rate is the statistic that survives a lopsided set, so
        # it is recorded either way rather than only in the single-class case.
        report["detection_rate"] = (
            sum(row["label"] == 1 and row["predicted"] == 1 for row in rows) / positives
            if positives
            else None
        )
    else:
        # A single-class set has nothing to rank against, so ROC-AUC and every
        # other ranking metric is undefined. MNW's lab half is exactly this: 120
        # forgeries and no genuine video. The detection rate is the honest
        # statistic, and reporting an AUC here would be inventing one.
        present = labels[0]
        correct = sum(row["label"] == row["predicted"] for row in rows)
        rate_name = "detection_rate" if present == 1 else "specificity"
        report["metrics"] = None
        report["single_class"] = {
            "class": "fake" if present == 1 else "real",
            rate_name: correct / len(rows),
            "mean_probability": sum(probabilities) / len(rows),
            "note": (
                "Only one class is present, so ranking metrics such as ROC-AUC "
                "are undefined and are not reported."
            ),
        }
    return report


def _grouped_rates(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    """Per-group hit rate, which is how an unseen-generator result is read.

    On MNW this is the per-generator detection rate: the number that answers
    whether the detector catches forgeries it was never trained on.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "rows": len(group),
            "accuracy": sum(row["label"] == row["predicted"] for row in group)
            / len(group),
            "mean_probability": sum(float(row["probability"]) for row in group)
            / len(group),
        }
        for name, group in sorted(grouped.items())
    }


def _flat_metrics(report: dict[str, object]):
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            if isinstance(value, int | float):
                yield name, value
    single = report.get("single_class")
    if isinstance(single, dict):
        for name, value in single.items():
            if isinstance(value, int | float):
                yield name, value
    confusion = report["confusion"]
    if isinstance(confusion, dict):
        for name, value in confusion.items():
            yield f"confusion.{name}", value


def _select_threshold(arguments: argparse.Namespace) -> int:
    with arguments.predictions.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    scored = [row for row in rows if row.get("probability", "").strip()]
    selection = select_balanced_accuracy_threshold(
        labels=[int(row["label"]) for row in scored],
        probabilities=[float(row["probability"]) for row in scored],
    )
    _write_json(
        arguments.output,
        {
            "selection_set": str(arguments.predictions),
            "objective": "balanced_accuracy",
            "scored": len(scored),
            **asdict(selection),
        },
    )
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    with arguments.predictions.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    items = [
        EvaluationItem(
            label=int(row["label"]),
            probability=(
                float(row["probability"])
                if row.get("probability", "").strip()
                else None
            ),
            source_identity=row["source"],
            method=row["method"],
            race=row.get("race", "unknown"),
            gender=row.get("gender", "unknown"),
        )
        for row in rows
    ]
    overall = evaluate_items(items, threshold=arguments.threshold)
    methods = per_method_metrics(items, threshold=arguments.threshold)
    intervals = bootstrap_binary_metrics(
        items,
        threshold=arguments.threshold,
        samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    method_auc_values = [
        report.metrics.roc_auc
        for report in methods.values()
        if report.metrics is not None
    ]
    paired = [
        PairedPrediction(
            label=int(row["label"]),
            source_identity=row["source"],
            left_probability=float(row["probability"]),
            right_probability=float(row["visual_probability"]),
        )
        for row in rows
        if row.get("probability", "").strip()
        and row.get("visual_probability", "").strip()
    ]
    paired_interval = (
        paired_auc_difference(
            paired,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        if paired and set(item.label for item in paired) == {0, 1}
        else None
    )
    _write_json(
        arguments.output,
        {
            "threshold": arguments.threshold,
            "overall": asdict(overall),
            "confidence_intervals": {
                name: asdict(interval) for name, interval in intervals.items()
            },
            "per_method": {name: asdict(report) for name, report in methods.items()},
            "macro_method_roc_auc": (
                sum(method_auc_values) / len(method_auc_values)
                if method_auc_values
                else None
            ),
            "fusion_vs_visual_auc": (
                asdict(paired_interval) if paired_interval is not None else None
            ),
            "race": {
                name: asdict(report)
                for name, report in subgroup_metrics(
                    items, attribute="race", threshold=arguments.threshold
                ).items()
            },
            "gender": {
                name: asdict(report)
                for name, report in subgroup_metrics(
                    items, attribute="gender", threshold=arguments.threshold
                ).items()
            },
        },
    )
    return 0


def _train_fusion(arguments: argparse.Namespace) -> int:
    branches = tuple(arguments.branches)
    if len(set(branches)) != len(branches):
        raise ValueError("Fusion branch names must be unique")
    rows = FeatureStore(arguments.feature_store).assemble(required_branches=branches)
    if {row.partition_role for row in rows} != {"oof"}:
        raise ValueError("Fusion training requires out-of-fold feature rows")
    # The feature store is a blend, so the guard runs over every dataset that
    # contributed a row rather than over a single --dataset flag.
    reject_evaluation_only_datasets(
        (row.dataset for row in rows), operation="fusion training"
    )
    samples = [
        FusionSample(
            branch_logits=row.branch_logits,
            face_coverage=row.face_coverage,
            audio_clipped=row.audio_clipped,
            av_duration_delta_sec=row.av_duration_delta_sec,
        )
        for row in rows
    ]
    labels = [row.label for row in rows]
    model = LateFusion(
        branch_names=branches,
        classifier_kind=arguments.model,
    ).fit(samples, labels)
    split_hashes = {row.split_hash for row in rows}
    preprocessing_hashes = {row.preprocessing_hash for row in rows}
    if len(split_hashes) != 1:
        raise ValueError("Fusion rows use different split hashes")
    if len(preprocessing_hashes) != 1:
        raise ValueError("Fusion rows use different preprocessing hashes")
    split_hash = split_hashes.pop()
    preprocessing_hash = preprocessing_hashes.pop()
    artifact = FusionArtifact(
        model=model,
        split_hash=split_hash,
        preprocessing_hash=preprocessing_hash,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, arguments.output)
    _write_json(
        arguments.metadata,
        {
            "samples": len(samples),
            "branches": list(branches),
            "model": arguments.model,
            "feature_store": str(arguments.feature_store),
            "split_hash": split_hash,
            "preprocessing_hash": preprocessing_hash,
            "oof_run_ids": sorted({row.run_id for row in rows}),
            "branch_checkpoint_hashes": {
                branch: sorted({row.checkpoint_hashes[branch] for row in rows})
                for branch in branches
            },
        },
    )
    log_fusion_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        samples=len(samples),
        branches=branches,
        model_kind=arguments.model,
        split_hash=split_hash,
        preprocessing_hash=preprocessing_hash,
        model_path=arguments.output,
        metadata_path=arguments.metadata,
    )
    return 0


def _read_cache_index(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    return {
        row["clip_id"]: (
            Path(row["cache_path"])
            if Path(row["cache_path"]).is_absolute()
            else (path.parent / row["cache_path"]).resolve()
        )
        for row in rows
    }


def _cache_index_value(cache_path: Path, *, index: Path) -> Path:
    resolved = cache_path.resolve()
    try:
        return resolved.relative_to(index.parent.resolve())
    except ValueError:
        return resolved


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "uncommitted"
    # The command has fixed arguments and never invokes a shell.
    process = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "uncommitted"


def _training_cost_metrics(
    *,
    train_samples: int,
    completed_epochs: int,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
) -> dict[str, int | float]:
    training_examples = train_samples * completed_epochs
    return {
        "training_examples": training_examples,
        "samples_per_second": training_examples / elapsed_seconds,
        "peak_gpu_memory_mib": peak_gpu_memory_bytes / (1024**2),
    }


def _binary_branch_train(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from deepfake_detection.branches.audio import build_wav2vec2_audio_branch
    from deepfake_detection.branches.visual import build_efficientnet_b0
    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.training.binary import (
        BinaryTrainingConfig,
        fit_binary_branch,
    )
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = _read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)
    train_dataset = CachedBranchDataset(
        records=train_records,
        cache_index=index,
        cache_store=cache_store,
        branch=arguments.train_command,
        preprocessing_hash=arguments.preprocessing_hash,
    )
    validation_dataset = CachedBranchDataset(
        records=validation_records,
        cache_index=index,
        cache_store=cache_store,
        branch=arguments.train_command,
        preprocessing_hash=arguments.preprocessing_hash,
    )
    labels = [
        int(
            record.video_fake
            if arguments.train_command == "visual"
            else record.audio_fake
        )
        for record in train_records
    ]
    if set(labels) != {0, 1}:
        raise ValueError("Branch training requires both cue-specific classes")
    counts = {label: labels.count(label) for label in (0, 1)}
    weights = [1.0 / counts[label] for label in labels]
    generator = torch.Generator().manual_seed(arguments.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_branch_items,
        num_workers=arguments.workers,
    )
    if arguments.train_command == "visual":
        model = build_efficientnet_b0(pretrained=True)
    else:
        model = build_wav2vec2_audio_branch(model_name=arguments.audio_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = BinaryTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        freeze_epochs=arguments.freeze_epochs,
        early_stopping_patience=arguments.patience,
    )
    run_config = {
        "training": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "data": {
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "sampler": "inverse_frequency_with_replacement",
        },
        "model": {
            "branch": arguments.train_command,
            "audio_model": (
                arguments.audio_model if arguments.train_command == "audio" else None
            ),
            "pretrained": True,
        },
    }
    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_binary_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch=arguments.train_command,
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )
    log_binary_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
        samples_per_second=float(training_cost["samples_per_second"]),
        peak_gpu_memory_mib=float(training_cost["peak_gpu_memory_mib"]),
    )
    return 0


def _stream_train(arguments: argparse.Namespace) -> int:
    """Train one audiovisual cross-attention stream.

    Parallel to `_binary_branch_train` rather than folded into it: the batch
    carries a video and an audio tensor instead of one, and the history records
    diagonal attention mass per epoch, which the binary history has no field
    for.
    """
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedAVPairDataset,
        collate_av_pair_items,
    )
    from deepfake_detection.streams.cross_modal_stream import build_lipsync_stream
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )
    from deepfake_detection.training.streams import (
        StreamTrainingConfig,
        fit_stream,
        parameter_groups,
    )

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = _read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)

    def dataset_for(records):
        return CachedAVPairDataset(
            records=records,
            cache_index=index,
            cache_store=cache_store,
            stream=arguments.stream,
            preprocessing_hash=arguments.preprocessing_hash,
        )

    train_dataset = dataset_for(train_records)
    validation_dataset = dataset_for(validation_records)
    labels = [int(record.clip_fake) for record in train_records]
    if set(labels) != {0, 1}:
        raise ValueError("Stream training needs both real and fake clips")

    # Inverse-frequency sampling, matching the branch trainers. FakeAVCeleb is
    # roughly twenty to one fake, so uniform sampling would show the model a
    # real clip once a batch.
    counts = {label: labels.count(label) for label in set(labels)}
    generator = torch.Generator()
    generator.manual_seed(arguments.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=[1.0 / counts[label] for label in labels],
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=sampler,
        collate_fn=collate_av_pair_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_av_pair_items,
        num_workers=arguments.workers,
    )

    model = build_lipsync_stream(
        video_backbone=arguments.video_backbone,
        audio_model=arguments.audio_model,
        pretrained=True,
        common_dim=arguments.common_dim,
        attention_heads=arguments.attention_heads,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            head_lr=arguments.learning_rate,
            encoder_lr=arguments.encoder_learning_rate,
        ),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = StreamTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        freeze_epochs=arguments.freeze_epochs,
        early_stopping_patience=arguments.patience,
    )
    run_config = {
        "stream": arguments.stream,
        "dataset": arguments.dataset,
        "training": asdict(config),
        "optimizer": {
            "name": "adamw",
            "learning_rate": arguments.learning_rate,
            "encoder_learning_rate": arguments.encoder_learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "model": {
            "video_backbone": arguments.video_backbone,
            "audio_model": arguments.audio_model,
            "common_dim": arguments.common_dim,
            "attention_heads": arguments.attention_heads,
            "pretrained": True,
        },
    }

    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_stream(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch=arguments.stream,
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    best = history.epochs[history.best_epoch - 1]
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )

    logger = getattr(arguments, "_run_logger", NullRunLogger())
    logger.log_params({f"stream.{key}": value for key, value in run_config["model"].items()})
    logger.log_metrics(
        {
            "training.loss": best.train_loss,
            "validation.loss": best.validation_loss,
            # The stream's independent check. A falling loss beside a flat
            # diagonal mass means a shortcut was found, not synchronisation.
            "validation.diagonal_mass": best.validation_diagonal_mass,
            "training.best_epoch": float(history.best_epoch),
            "training.samples_per_second": float(training_cost["samples_per_second"]),
            "training.peak_gpu_memory_mib": float(
                training_cost["peak_gpu_memory_mib"]
            ),
            "training.elapsed_seconds": time.perf_counter() - started_at,
        }
    )
    logger.log_artifact(arguments.checkpoint, artifact_path="checkpoints")
    logger.log_artifact(arguments.history, artifact_path="history")
    return 0


def _sync_branch_train(arguments: argparse.Namespace) -> int:
    reject_evaluation_only(arguments.dataset, operation="training")
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)
    runtime.require_research_cuda(arguments.device)

    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.branches.sync import build_sync_branch
    from deepfake_detection.branches.sync_objective import OFFSET_MILLISECONDS
    from deepfake_detection.data.datasets import (
        CachedGlobalSyncDataset,
        CachedSyncDataset,
        collate_sync_items,
    )
    from deepfake_detection.training.checkpoints import (
        RunMetadata,
        hash_config,
        save_checkpoint,
    )
    from deepfake_detection.training.sync import SyncTrainingConfig, fit_sync_branch

    train_records = load_manifest(
        arguments.train_manifest, dataset=arguments.dataset
    ).records
    validation_records = load_manifest(
        arguments.validation_manifest, dataset=arguments.dataset
    ).records
    index = _read_cache_index(arguments.cache_index)
    cache_store = CacheStore(arguments.cache_root)
    dataset_type = (
        CachedSyncDataset
        if arguments.label_mode == "authentic-offset"
        else CachedGlobalSyncDataset
    )
    dataset_arguments = {
        "cache_index": index,
        "cache_store": cache_store,
        "preprocessing_hash": arguments.preprocessing_hash,
    }
    train_dataset = dataset_type(
        records=train_records,
        **dataset_arguments,
        **({"sample_rate": 16_000} if dataset_type is CachedSyncDataset else {}),
    )
    validation_dataset = dataset_type(
        records=validation_records,
        **dataset_arguments,
        **({"sample_rate": 16_000} if dataset_type is CachedSyncDataset else {}),
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sync_items,
        num_workers=arguments.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_sync_items,
        num_workers=arguments.workers,
    )
    model = build_sync_branch(audio_model_name=arguments.audio_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    config = SyncTrainingConfig(
        epochs=arguments.epochs,
        accumulation_steps=arguments.accumulation_steps,
        heads_epochs=arguments.heads_epochs,
        early_stopping_patience=arguments.patience,
        contrastive_weight=arguments.contrastive_weight,
    )
    run_config = {
        "training": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "data": {
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "offset_milliseconds": (
                list(OFFSET_MILLISECONDS)
                if arguments.label_mode == "authentic-offset"
                else []
            ),
            "mismatch": (
                "cross_identity"
                if arguments.label_mode == "authentic-offset"
                else "global_fake_label"
            ),
            "label_mode": arguments.label_mode,
        },
        "model": {
            "branch": "sync",
            "audio_model": arguments.audio_model,
            "pretrained": True,
        },
    }
    torch.cuda.reset_peak_memory_stats(arguments.device)
    training_started = time.perf_counter()
    history = fit_sync_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
    )
    training_cost = _training_cost_metrics(
        train_samples=len(train_dataset),
        completed_epochs=len(history.epochs),
        elapsed_seconds=time.perf_counter() - training_started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(arguments.device),
    )
    metadata = RunMetadata(
        run_id=arguments.run_id,
        branch="sync",
        git_commit=_git_commit(),
        split_hash=arguments.split_hash,
        preprocessing_hash=arguments.preprocessing_hash,
        config_hash=_configuration_hash(arguments, run_config, hash_config),
        seed=arguments.seed,
    )
    checkpoint_hash = save_checkpoint(
        arguments.checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        epoch=history.best_epoch,
    )
    _write_json(
        arguments.history,
        {
            "metadata": asdict(metadata),
            "config": run_config,
            "checkpoint_hash": checkpoint_hash,
            "best_epoch": history.best_epoch,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "hardware": training_cost,
        },
    )
    log_sync_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
        samples_per_second=float(training_cost["samples_per_second"]),
        peak_gpu_memory_mib=float(training_cost["peak_gpu_memory_mib"]),
    )
    return 0


def _run_configured(arguments: argparse.Namespace) -> int:
    return execute_configured_run(
        arguments.config,
        root=arguments.root,
        parser_factory=build_parser,
        disable_tracking=arguments.no_tracking,
    )


def _configuration_hash(
    arguments: argparse.Namespace,
    run_config: object,
    fallback: Callable[[object], str],
) -> str:
    resolved = getattr(arguments, "_config_hash", None)
    return resolved if resolved is not None else fallback(run_config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logger(arguments: argparse.Namespace):
    return getattr(arguments, "_run_logger", NullRunLogger())


def _detector_fetch_yunet(arguments: argparse.Namespace) -> int:
    model = fetch_yunet_model(arguments.destination, force=arguments.force)
    digest = _sha256(model)
    payload = {
        "asset": "opencv-zoo-yunet-2026may",
        "sha256": digest,
    }
    if arguments.report is not None:
        _write_json(arguments.report, payload)
        _run_logger(arguments).log_artifact(
            arguments.report,
            artifact_path="detector/aggregate",
        )
    _run_logger(arguments).log_params(
        {
            "detector.asset": payload["asset"],
            "detector.model_sha256": digest,
        }
    )
    return 0


def _manifest_media_path(record, dataset_root: Path) -> Path:
    path = Path(record.video_path)
    return path if path.is_absolute() else dataset_root / path


def _load_frozen_split(split_dir: Path, dataset: str):
    return {
        name: load_manifest(split_dir / f"{name}.csv", dataset=dataset).records
        for name in ("train", "val", "test")
    }


def _detector_sample(arguments: argparse.Namespace) -> int:
    frozen_split = _load_frozen_split(arguments.split_dir, arguments.dataset)
    observed_split_hash = split_hash(frozen_split)
    if observed_split_hash != arguments.expected_split_hash:
        raise ValueError("Frozen split does not match the expected split hash")
    strict_split = identity_strict_subset(frozen_split)
    records = strict_split["train"]
    decoder = FFmpegMediaDecoder()

    def duration_reader(record) -> float:
        return decoder.probe(
            _manifest_media_path(record, arguments.dataset_root)
        ).duration_sec

    def frame_reader(record, timestamp: float):
        return decoder.read_frames(
            _manifest_media_path(record, arguments.dataset_root),
            (timestamp,),
        )[0]

    sample = build_review_sample(
        records,
        partition=arguments.partition,
        frozen_split=frozen_split,
        expected_split_hash=arguments.expected_split_hash,
        duration_reader=duration_reader,
        frame_reader=frame_reader,
        frame_count=arguments.frames,
        clip_count=arguments.clips,
        double_review_fraction=arguments.double_review_fraction,
        seed=arguments.seed,
    )
    write_review_sample(sample, arguments.output)
    records_by_id = {record.clip_id: record for record in records}
    arguments.review_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for row in sample:
        record = records_by_id[row.clip_id]
        frame = frame_reader(record, row.timestamp_sec)
        destination = arguments.review_dir / f"{row.frame_id}.png"
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"Cannot write review image: {destination}")
    payload = {
        "evidence_scope": "research_evidence_pending_human_review",
        "frame_count": len(sample),
        "clip_count": len({row.clip_id for row in sample}),
        "source_count": len({row.source_hash for row in sample}),
        "double_review_count": sum(row.double_review for row in sample),
        "comparison_frame_count": sum(row.split_role == "comparison" for row in sample),
        "comparison_clip_count": len(
            {row.clip_id for row in sample if row.split_role == "comparison"}
        ),
        "sample_sha256": review_sample_sha256(sample),
        "split_hash": observed_split_hash,
        "identity_strict_split_hash": split_hash(strict_split),
        "seed": arguments.seed,
    }
    _write_json(arguments.report, payload)
    logger = _run_logger(arguments)
    logger.log_params(
        {
            "detector.sample_sha256": payload["sample_sha256"],
            "detector.evidence_scope": payload["evidence_scope"],
            "detector.sample_seed": arguments.seed,
        }
    )
    logger.log_metrics(
        {
            "detector.review_frames": float(payload["frame_count"]),
            "detector.review_clips": float(payload["clip_count"]),
            "detector.review_sources": float(payload["source_count"]),
        }
    )
    logger.log_artifact(arguments.report, artifact_path="detector/aggregate")
    return 0


def _detector_validate_annotations(arguments: argparse.Namespace) -> int:
    sample = read_review_sample(arguments.sample)
    annotations = read_annotations(arguments.annotations)
    audit = validate_annotations(sample, annotations)
    _write_json(arguments.report, asdict(audit))
    logger = _run_logger(arguments)
    logger.log_params(
        {
            "detector.annotation_audit_valid": audit.valid,
            "detector.annotation_audit_sha256": annotation_audit_sha256(audit),
            "detector.reviewed_sample_sha256": audit.reviewed_sample_sha256,
            "detector.split_hash": audit.split_hash,
            "detector.identity_strict_split_hash": (audit.identity_strict_split_hash),
        }
    )
    logger.log_metrics(
        {
            "detector.annotation_frames": float(audit.frame_count),
            "detector.annotation_reviews": float(audit.review_count),
            "detector.annotation_disagreements": float(len(audit.disagreements)),
            "detector.comparison_frames": float(audit.comparison_frame_count),
            "detector.comparison_clips": float(audit.comparison_clip_count),
        }
    )
    logger.log_artifact(arguments.report, artifact_path="detector/aggregate")
    return 0 if audit.valid else 2


def _detector_frame_reader(
    *,
    manifest: Path,
    dataset: str,
    dataset_root: Path,
):
    records = load_manifest(manifest, dataset=dataset).records
    records_by_id = {record.clip_id: record for record in records}
    decoder = FFmpegMediaDecoder()

    def read(row: ReviewFrame):
        try:
            record = records_by_id[row.clip_id]
        except KeyError as error:
            raise ValueError(
                f"Review sample clip is absent from the manifest: {row.clip_id}"
            ) from error
        return decoder.read_frames(
            _manifest_media_path(record, dataset_root),
            (row.timestamp_sec,),
        )[0]

    return read


def _detector_run(arguments: argparse.Namespace) -> int:
    logger = _run_logger(arguments)
    sample = read_review_sample(arguments.sample)
    frozen_split = _load_frozen_split(arguments.split_dir, arguments.dataset)
    strict_split = identity_strict_subset(frozen_split)
    sample_split_hashes = {row.split_hash for row in sample}
    if sample_split_hashes != {split_hash(frozen_split)}:
        raise ValueError("Review sample does not match the frozen split artifact")
    sample_strict_hashes = {row.identity_strict_split_hash for row in sample}
    if sample_strict_hashes != {split_hash(strict_split)}:
        raise ValueError("Review sample does not match identity-strict training")
    frozen_train = {
        (
            record.clip_id,
            hashlib.sha256(f"{record.dataset}\0{record.source}".encode()).hexdigest(),
        )
        for record in strict_split["train"]
    }
    if any((row.clip_id, row.source_hash) not in frozen_train for row in sample):
        raise ValueError("Review sample contains a source outside frozen training")
    annotations = read_annotations(arguments.annotations)
    preprocessor = build_preprocessor(
        code_version=arguments.code_version,
        device=arguments.device,
        detector=arguments.detector,
        model_path=arguments.model_path,
        expected_model_hash=arguments.expected_model_hash,
        detector_confidence=arguments.collection_threshold,
    )
    report = run_detector_benchmark(
        sample=sample,
        annotations=annotations,
        detector=preprocessor.detector,
        detector_name=arguments.detector,
        detector_revision=arguments.detector_revision,
        model_sha256=preprocessor.config.detector_model_sha256,
        frame_reader=_detector_frame_reader(
            manifest=arguments.split_dir / "train.csv",
            dataset=arguments.dataset,
            dataset_root=arguments.dataset_root,
        ),
        raw_output=arguments.predictions,
        runtime_snapshot=capture_runtime(Path.cwd()),
        collection_threshold=arguments.collection_threshold,
        warmup_frames=arguments.warmup_frames,
        evidence_scope=arguments.evidence_scope,
        source_run_id=(
            logger.run_id
            or arguments.source_run_id
            or (
                "software-fixture"
                if arguments.evidence_scope == "software_fixture_only"
                else ""
            )
        ),
        environment_lock_sha256=_sha256(Path.cwd() / "uv.lock"),
    )
    _write_json(arguments.report, asdict(report))
    log_detector_benchmark(
        logger,
        report=report,
        report_path=arguments.report,
        predictions_path=arguments.predictions,
    )
    return 0


def _detector_report_from_path(path: Path) -> BoundDetectorReport:
    return read_bound_detector_report(path)


def _detector_compare(arguments: argparse.Namespace) -> int:
    reports = tuple(_detector_report_from_path(path) for path in arguments.reports)
    decision: DetectorDecision = compare_detectors(reports)
    _write_json(arguments.output, asdict(decision))
    logger = _run_logger(arguments)
    logger.log_params(
        {
            "detector.comparison_rule_revision": decision.rule_revision,
            "detector.comparison_reason": decision.reason,
            "detector.selected": decision.selected_detector or "none",
            "detector.selected_association": (decision.selected_association or "none"),
        }
    )
    logger.log_params(
        {
            **{
                f"detector.input_report_sha256.{name}": digest
                for name, digest in decision.input_report_sha256.items()
            },
            **{
                f"detector.source_run_id.{name}": run_id
                for name, run_id in decision.source_run_ids.items()
            },
            **{
                f"detector.common_evidence.{name}": digest
                for name, digest in decision.common_evidence_hashes.items()
            },
        }
    )
    logger.log_artifact(arguments.output, artifact_path="detector/aggregate")
    return 0


def _load_trained_branches(arguments: argparse.Namespace):
    from deepfake_detection.branches.audio import build_wav2vec2_audio_branch
    from deepfake_detection.branches.sync import build_sync_branch
    from deepfake_detection.branches.visual import build_efficientnet_b0
    from deepfake_detection.training.checkpoints import (
        load_checkpoint,
        validate_branch_states,
    )

    visual = build_efficientnet_b0(pretrained=False)
    audio = build_wav2vec2_audio_branch(
        model_name=arguments.audio_model,
        pretrained=False,
    )
    sync = build_sync_branch(
        audio_model_name=arguments.audio_model,
        pretrained=False,
    )
    states = {
        "visual": load_checkpoint(arguments.visual_checkpoint, model=visual),
        "audio": load_checkpoint(arguments.audio_checkpoint, model=audio),
        "sync": load_checkpoint(arguments.sync_checkpoint, model=sync),
    }
    validate_branch_states(states)
    return visual, audio, sync, states


def _features_export(arguments: argparse.Namespace) -> int:
    from deepfake_detection.fusion.export import export_features

    if arguments.partition_role != "external":
        # An external-role export only scores a frozen model. Any other role
        # feeds fusion training or model selection, which a locked dataset must
        # never influence.
        reject_evaluation_only(
            arguments.dataset,
            operation=f"a {arguments.partition_role}-role feature export",
        )

    records = load_manifest(arguments.manifest, dataset=arguments.dataset).records
    cache_index = _read_cache_index(arguments.cache_index)
    visual, audio, sync, states = _load_trained_branches(arguments)
    report = export_features(
        records=records,
        cache_index=cache_index,
        cache_store=CacheStore(arguments.cache_root),
        feature_store=FeatureStore(arguments.feature_store),
        visual_model=visual,
        audio_model=audio,
        sync_model=sync,
        checkpoint_hashes={
            "visual": _sha256(arguments.visual_checkpoint),
            "audio": _sha256(arguments.audio_checkpoint),
            "sync": _sha256(arguments.sync_checkpoint),
        },
        split_hash=states["visual"].metadata.split_hash,
        preprocessing_hash=states["visual"].metadata.preprocessing_hash,
        partition_role=arguments.partition_role,
        run_id=arguments.run_id,
        device=arguments.device,
    )
    _write_json(arguments.report, asdict(report))
    return 2 if report.unavailable_rows else 0


def _features_score(arguments: argparse.Namespace) -> int:
    artifact = joblib.load(arguments.fusion_model)
    if not isinstance(artifact, FusionArtifact):
        raise ValueError("Fusion model does not contain provenance metadata")
    rows = FeatureStore(arguments.feature_store).assemble(
        required_branches=artifact.branch_names,
        strict=False,
    )
    if not rows:
        raise ValueError("Feature store has no complete fusion rows")
    if any(not row.source_identity for row in rows):
        raise ValueError("Fusion rows require source identity metadata")
    split_hashes = {row.split_hash for row in rows}
    preprocessing_hashes = {row.preprocessing_hash for row in rows}
    if len(split_hashes) != 1 or len(preprocessing_hashes) != 1:
        raise ValueError("Fusion scoring rows contain mixed provenance")
    artifact.validate_provenance(
        split_hash=split_hashes.pop(),
        preprocessing_hash=preprocessing_hashes.pop(),
    )

    def sample(row) -> FusionSample:
        return FusionSample(
            branch_logits=row.branch_logits,
            face_coverage=row.face_coverage,
            audio_clipped=row.audio_clipped,
            av_duration_delta_sec=row.av_duration_delta_sec,
        )

    probabilities: list[float | None] = [None] * len(rows)
    complete_indices = [index for index, row in enumerate(rows) if row.available]
    if complete_indices:
        complete_probabilities = artifact.predict_proba(
            [sample(rows[index]) for index in complete_indices]
        )
        for index, probability in zip(
            complete_indices,
            complete_probabilities,
            strict=True,
        ):
            probabilities[index] = float(probability)

    visual_probabilities: list[float | None] = [None] * len(rows)
    visual_indices = [
        index for index, row in enumerate(rows) if "visual" in row.branch_logits
    ]
    if "visual" in artifact.branch_names and visual_indices:
        available_visual = artifact.model.predict_branch_proba(
            [sample(rows[index]) for index in visual_indices],
            branch="visual",
        )
        for index, probability in zip(
            visual_indices,
            available_visual,
            strict=True,
        ):
            visual_probabilities[index] = float(probability)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "clip_id",
            "segment_id",
            "label",
            "probability",
            "visual_probability",
            "source",
            "method",
            "race",
            "gender",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability, visual_probability in zip(
            rows,
            probabilities,
            visual_probabilities,
            strict=True,
        ):
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "segment_id": row.segment_id,
                    "label": row.label,
                    "probability": "" if probability is None else probability,
                    "visual_probability": (
                        "" if visual_probability is None else float(visual_probability)
                    ),
                    "source": row.source_identity,
                    "method": row.method,
                    "race": row.race,
                    "gender": row.gender,
                }
            )
    return 0


def _predict(arguments: argparse.Namespace) -> int:
    engine = load_prediction_engine(
        InferenceConfig(
            visual_checkpoint=arguments.visual_checkpoint,
            audio_checkpoint=arguments.audio_checkpoint,
            sync_checkpoint=arguments.sync_checkpoint,
            fusion_model=arguments.fusion_model,
            code_version=arguments.code_version,
            threshold=arguments.threshold,
            audio_model=arguments.audio_model,
            device=arguments.device,
            detector=arguments.detector,
            tracker=arguments.tracker,
            crop_mode=arguments.crop_mode,
            model_path=arguments.model_path,
            expected_model_hash=arguments.expected_model_hash,
        )
    )
    result = engine.predict(arguments.video)
    _write_json(arguments.output, asdict(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddf")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--config", type=Path, action="append", required=True)
    run.add_argument("--no-tracking", action="store_true")
    run.set_defaults(
        handler=_run_configured,
        _configured_run_sentinel=_CONFIGURED_RUN_SENTINEL,
    )

    handoff = commands.add_parser("handoff")
    handoff_commands = handoff.add_subparsers(dest="handoff_command", required=True)
    handoff_update = handoff_commands.add_parser("update")
    handoff_update.add_argument("--root", type=Path, default=Path("."))
    handoff_update.add_argument(
        "--path", type=Path, default=Path("docs/handoff.md")
    )
    handoff_update.set_defaults(handler=_handoff_update)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--samples", type=int, default=32)
    smoke.set_defaults(handler=_smoke)

    manifest = commands.add_parser("manifest")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_build = manifest_commands.add_parser("build")
    manifest_build.add_argument("--input", type=Path, required=True)
    manifest_build.add_argument("--output", type=Path, required=True)
    manifest_build.add_argument("--audit", type=Path, required=True)
    manifest_build.add_argument("--dataset", required=True)

    manifest_usable = manifest_commands.add_parser("usable")
    manifest_usable.add_argument("--manifest", type=Path, required=True)
    manifest_usable.add_argument("--cache-index", type=Path, required=True)
    manifest_usable.add_argument("--cache-root", type=Path, required=True)
    manifest_usable.add_argument("--output", type=Path, required=True)
    manifest_usable.add_argument("--audit", type=Path, required=True)
    manifest_usable.add_argument("--dataset", required=True)
    manifest_usable.add_argument(
        "--branch",
        choices=("visual", "audio", "sync", "lipsync", "emotion"),
        default="visual",
    )
    manifest_usable.add_argument("--preprocessing-hash")
    manifest_usable.set_defaults(handler=_manifest_usable)

    manifest_meta = manifest_commands.add_parser("from-meta")
    manifest_meta.add_argument("--meta", type=Path, required=True)
    manifest_meta.add_argument("--root", type=Path, required=True)
    manifest_meta.add_argument("--data-dir", type=Path, default=Path("data"))
    manifest_meta.add_argument("--output", type=Path, required=True)
    manifest_meta.add_argument("--audit", type=Path, required=True)
    manifest_meta.add_argument("--dataset", required=True)
    manifest_meta.add_argument("--allow-missing", action="store_true")
    manifest_meta.set_defaults(handler=_manifest_from_meta)
    manifest_build.set_defaults(handler=_manifest_build)

    split = commands.add_parser("split")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    split_build = split_commands.add_parser("build")
    split_build.add_argument("--manifest", type=Path, required=True)
    split_build.add_argument("--output-dir", type=Path, required=True)
    split_build.add_argument("--dataset", required=True)
    split_build.add_argument("--seed", type=int, required=True)
    split_build.set_defaults(handler=_split_build)
    split_crossfit = split_commands.add_parser("crossfit")
    split_crossfit.add_argument("--manifest", type=Path, required=True)
    split_crossfit.add_argument("--output-dir", type=Path, required=True)
    split_crossfit.add_argument("--dataset", required=True)
    split_crossfit.add_argument("--folds", type=int, default=3)
    split_crossfit.add_argument("--seed", type=int, required=True)
    split_crossfit.set_defaults(handler=_split_crossfit)
    split_holdout = split_commands.add_parser("method-holdout")
    split_holdout.add_argument("--split-dir", type=Path, required=True)
    split_holdout.add_argument("--output-dir", type=Path, required=True)
    split_holdout.add_argument("--dataset", required=True)
    split_holdout.add_argument("--methods", nargs="+", required=True)

    split_subsample = split_commands.add_parser("subsample")
    split_subsample.add_argument("--split-dir", type=Path, required=True)
    split_subsample.add_argument("--output-dir", type=Path, required=True)
    split_subsample.add_argument("--dataset", required=True)
    split_subsample.add_argument("--seed", type=int, required=True)
    split_subsample.add_argument("--fake-ratio", type=float, default=3.0)
    split_subsample.set_defaults(handler=_split_subsample)
    split_holdout.set_defaults(handler=_split_method_holdout)

    cache = commands.add_parser("cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_build = cache_commands.add_parser("build")
    cache_build.add_argument("--manifest", type=Path, required=True)
    cache_build.add_argument("--dataset-root", type=Path, required=True)
    cache_build.add_argument("--cache-root", type=Path, required=True)
    cache_build.add_argument("--index", type=Path, required=True)
    cache_build.add_argument("--audit", type=Path, required=True)
    cache_build.add_argument("--dataset", required=True)
    cache_build.add_argument("--device", default="cpu")
    cache_build.add_argument("--code-version", required=True)
    cache_build.add_argument("--keep-leading-silence", action="store_true")
    cache_build.add_argument(
        "--skip-cached",
        action="store_true",
        help="Reuse clips already present in the cache root instead of rebuilding them.",
    )
    cache_build.add_argument(
        "--shard",
        help="Process only this shard, written I/N. Shards may run concurrently.",
    )

    cache_merge = cache_commands.add_parser("merge")
    cache_merge.add_argument("--indexes", type=Path, nargs="+", required=True)
    cache_merge.add_argument("--audits", type=Path, nargs="+", required=True)
    cache_merge.add_argument("--index", type=Path, required=True)
    cache_merge.add_argument("--audit", type=Path, required=True)
    cache_merge.set_defaults(handler=_cache_merge)

    def add_preprocessing_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--detector",
            choices=("mtcnn", "yunet"),
            default="mtcnn",
        )
        command.add_argument(
            "--tracker",
            choices=("greedy_iou", "constant_velocity"),
            default="greedy_iou",
        )
        command.add_argument(
            "--crop-mode",
            choices=("box", "landmark"),
            default="box",
        )
        command.add_argument("--model-path", type=Path)
        command.add_argument("--expected-model-hash")

    add_preprocessing_arguments(cache_build)
    cache_build.set_defaults(handler=_cache_build)

    detector = commands.add_parser("detector")
    detector_commands = detector.add_subparsers(
        dest="detector_command",
        required=True,
    )
    detector_fetch = detector_commands.add_parser("fetch-yunet")
    detector_fetch.add_argument(
        "--destination",
        type=Path,
        default=Path("models/face_detection_yunet_2026may.onnx"),
    )
    detector_fetch.add_argument("--force", action="store_true")
    detector_fetch.add_argument("--report", type=Path)
    detector_fetch.set_defaults(handler=_detector_fetch_yunet)

    detector_sample = detector_commands.add_parser("sample")
    detector_sample.add_argument("--split-dir", type=Path, required=True)
    detector_sample.add_argument("--expected-split-hash", required=True)
    detector_sample.add_argument("--dataset-root", type=Path, required=True)
    detector_sample.add_argument("--dataset", required=True)
    detector_sample.add_argument("--output", type=Path, required=True)
    detector_sample.add_argument("--review-dir", type=Path, required=True)
    detector_sample.add_argument("--report", type=Path, required=True)
    detector_sample.add_argument("--partition", default="train")
    detector_sample.add_argument("--frames", type=int, default=MINIMUM_REVIEW_FRAMES)
    detector_sample.add_argument("--clips", type=int, default=MINIMUM_REVIEW_CLIPS)
    detector_sample.add_argument(
        "--double-review-fraction",
        type=float,
        default=0.10,
    )
    detector_sample.add_argument("--seed", type=int, default=17)
    detector_sample.set_defaults(handler=_detector_sample)

    detector_validate = detector_commands.add_parser("validate-annotations")
    detector_validate.add_argument("--sample", type=Path, required=True)
    detector_validate.add_argument("--annotations", type=Path, required=True)
    detector_validate.add_argument("--report", type=Path, required=True)
    detector_validate.set_defaults(handler=_detector_validate_annotations)

    detector_run = detector_commands.add_parser("run")
    detector_run.add_argument("--sample", type=Path, required=True)
    detector_run.add_argument("--annotations", type=Path, required=True)
    detector_run.add_argument("--split-dir", type=Path, required=True)
    detector_run.add_argument("--dataset-root", type=Path, required=True)
    detector_run.add_argument("--dataset", required=True)
    detector_run.add_argument("--predictions", type=Path, required=True)
    detector_run.add_argument("--report", type=Path, required=True)
    detector_run.add_argument(
        "--detector",
        choices=("mtcnn", "yunet"),
        required=True,
    )
    detector_run.add_argument("--detector-revision", required=True)
    detector_run.add_argument("--model-path", type=Path)
    detector_run.add_argument("--expected-model-hash")
    detector_run.add_argument("--code-version", default="detector-benchmark-v1")
    detector_run.add_argument("--device", default="cpu")
    detector_run.add_argument("--collection-threshold", type=float, default=0.0)
    detector_run.add_argument("--warmup-frames", type=int, default=3)
    detector_run.add_argument("--source-run-id")
    detector_run.add_argument(
        "--evidence-scope",
        choices=("research_evidence", "software_fixture_only"),
        default="research_evidence",
    )
    detector_run.set_defaults(handler=_detector_run)

    detector_compare = detector_commands.add_parser("compare")
    detector_compare.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
    )
    detector_compare.add_argument("--output", type=Path, required=True)
    detector_compare.set_defaults(handler=_detector_compare)

    evaluate = commands.add_parser("evaluate")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)

    evaluate_branch = evaluate_commands.add_parser("branch")
    evaluate_branch.add_argument(
        "--branch", choices=("visual", "audio"), required=True
    )
    evaluate_branch.add_argument("--checkpoint", type=Path, required=True)
    evaluate_branch.add_argument("--manifest", type=Path, required=True)
    evaluate_branch.add_argument("--cache-index", type=Path, required=True)
    evaluate_branch.add_argument("--cache-root", type=Path, required=True)
    evaluate_branch.add_argument("--dataset", required=True)
    evaluate_branch.add_argument("--threshold", type=float, required=True)
    evaluate_branch.add_argument("--output", type=Path, required=True)
    evaluate_branch.add_argument("--predictions", type=Path, required=True)
    evaluate_branch.add_argument("--evidence-scope", required=True)
    evaluate_branch.add_argument("--preprocessing-hash")
    evaluate_branch.add_argument("--device", default="cuda")
    evaluate_branch.add_argument("--batch-size", type=int, default=8)
    evaluate_branch.add_argument("--workers", type=int, default=0)
    evaluate_branch.add_argument("--audio-model", default="facebook/wav2vec2-base")
    evaluate_branch.set_defaults(handler=_evaluate_branch)

    evaluate_predictions = evaluate_commands.add_parser("predictions")
    evaluate_predictions.add_argument("--predictions", type=Path, required=True)
    evaluate_predictions.add_argument("--output", type=Path, required=True)
    evaluate_predictions.add_argument("--threshold", type=float, required=True)
    evaluate_predictions.add_argument("--bootstrap-samples", type=int, default=1_000)
    evaluate_predictions.add_argument("--seed", type=int, default=17)
    evaluate_predictions.set_defaults(handler=_evaluate)

    threshold = commands.add_parser("threshold")
    threshold.add_argument("--predictions", type=Path, required=True)
    threshold.add_argument("--output", type=Path, required=True)
    threshold.set_defaults(handler=_select_threshold)

    train = commands.add_parser("train")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_fusion = train_commands.add_parser("fusion")
    train_fusion.add_argument("--feature-store", type=Path, required=True)
    train_fusion.add_argument("--output", type=Path, required=True)
    train_fusion.add_argument("--metadata", type=Path, required=True)
    train_fusion.add_argument(
        "--model",
        choices=("logistic", "mlp"),
        default="logistic",
    )
    train_fusion.add_argument(
        "--branches",
        nargs="+",
        choices=("visual", "audio", "sync"),
        default=("visual", "audio", "sync"),
    )
    train_fusion.set_defaults(handler=_train_fusion)

    def add_branch_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--train-manifest", type=Path, required=True)
        command.add_argument("--validation-manifest", type=Path, required=True)
        command.add_argument("--cache-index", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, required=True)
        command.add_argument("--dataset", required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--history", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--split-hash", required=True)
        command.add_argument("--preprocessing-hash", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--epochs", type=int, default=12)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--accumulation-steps", type=int, default=4)
        command.add_argument("--learning-rate", type=float, default=1e-4)
        command.add_argument("--weight-decay", type=float, default=1e-4)
        command.add_argument("--patience", type=int, default=3)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--seed", type=int, default=17)

    train_visual = train_commands.add_parser("visual")
    add_branch_arguments(train_visual)
    train_visual.add_argument("--freeze-epochs", type=int, default=3)
    train_visual.set_defaults(handler=_binary_branch_train)

    train_audio = train_commands.add_parser("audio")
    add_branch_arguments(train_audio)
    train_audio.add_argument("--freeze-epochs", type=int, default=3)
    train_audio.add_argument(
        "--audio-model",
        default="facebook/wav2vec2-base",
    )
    train_audio.set_defaults(handler=_binary_branch_train)

    train_stream = train_commands.add_parser("stream")
    add_branch_arguments(train_stream)
    train_stream.add_argument(
        "--stream", choices=("lipsync", "emotion"), default="lipsync"
    )
    train_stream.add_argument("--freeze-epochs", type=int, default=3)
    train_stream.add_argument(
        "--video-backbone", default="tf_efficientnet_b0.ns_jft_in1k"
    )
    train_stream.add_argument("--audio-model", default="facebook/wav2vec2-base")
    train_stream.add_argument("--common-dim", type=int, default=256)
    train_stream.add_argument("--attention-heads", type=int, default=4)
    train_stream.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for the pretrained encoders, kept well below the head's.",
    )
    train_stream.set_defaults(handler=_stream_train)

    train_sync = train_commands.add_parser("sync")
    add_branch_arguments(train_sync)
    train_sync.add_argument("--heads-epochs", type=int, default=3)
    train_sync.add_argument("--contrastive-weight", type=float, default=0.1)
    train_sync.add_argument(
        "--label-mode",
        choices=("authentic-offset", "global-fake"),
        default="authentic-offset",
    )
    train_sync.add_argument(
        "--audio-model",
        default="facebook/wav2vec2-base",
    )
    train_sync.set_defaults(handler=_sync_branch_train)

    def add_checkpoint_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--visual-checkpoint", type=Path, required=True)
        command.add_argument("--audio-checkpoint", type=Path, required=True)
        command.add_argument("--sync-checkpoint", type=Path, required=True)
        command.add_argument(
            "--audio-model",
            default="facebook/wav2vec2-base",
        )
        command.add_argument("--device", default="cuda")

    features = commands.add_parser("features")
    feature_commands = features.add_subparsers(dest="feature_command", required=True)
    feature_export = feature_commands.add_parser("export")
    feature_export.add_argument("--manifest", type=Path, required=True)
    feature_export.add_argument("--cache-index", type=Path, required=True)
    feature_export.add_argument("--cache-root", type=Path, required=True)
    feature_export.add_argument("--feature-store", type=Path, required=True)
    feature_export.add_argument("--report", type=Path, required=True)
    feature_export.add_argument("--dataset", required=True)
    feature_export.add_argument("--run-id", required=True)
    feature_export.add_argument(
        "--partition-role",
        required=True,
        choices=("oof", "validation", "test", "external", "stress"),
    )
    add_checkpoint_arguments(feature_export)
    feature_export.set_defaults(handler=_features_export)
    feature_score = feature_commands.add_parser("score")
    feature_score.add_argument("--feature-store", type=Path, required=True)
    feature_score.add_argument("--fusion-model", type=Path, required=True)
    feature_score.add_argument("--output", type=Path, required=True)
    feature_score.set_defaults(handler=_features_score)

    predict = commands.add_parser("predict")
    predict.add_argument("video", type=Path)
    predict.add_argument("--fusion-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--threshold", type=float, required=True)
    predict.add_argument("--code-version", required=True)
    add_checkpoint_arguments(predict)
    add_preprocessing_arguments(predict)
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
