from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
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
    DetectorBenchmarkReport,
    DetectorDecision,
    compare_detectors,
    read_detector_report,
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
from deepfake_detection.data.manifest import load_manifest, write_manifest
from deepfake_detection.data.protocols import (
    audit_split,
    build_method_holdout_protocol,
    build_source_split,
    identity_strict_subset,
    split_hash,
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


def _smoke(arguments: argparse.Namespace) -> int:
    run_fusion_smoke(
        arguments.output_dir,
        seed=arguments.seed,
        samples=arguments.samples,
        logger=getattr(arguments, "_run_logger", NullRunLogger()),
    )
    return 0


def _split_build(arguments: argparse.Namespace) -> int:
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
    )
    arguments.index.parent.mkdir(parents=True, exist_ok=True)
    with arguments.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "cache_path"))
        writer.writeheader()
        for clip_id, cache_path in sorted(report.cache_index.items()):
            writer.writerow({"clip_id": clip_id, "cache_path": str(cache_path)})
    _write_json(
        arguments.audit,
        {
            "succeeded": report.succeeded,
            "failed": report.failed,
            "full_fusion_ready": report.full_fusion_ready,
            "blocker_counts": report.blocker_counts,
            "preprocessing_hash": report.preprocessing_hash,
            "failures": report.failures,
        },
    )
    return 2 if report.failed else 0


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


def _binary_branch_train(arguments: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)

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
    history = fit_binary_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
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
        },
    )
    log_binary_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    return 0


def _sync_branch_train(arguments: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    runtime.seed_everything(arguments.seed, deterministic=True)

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
    history = fit_sync_branch(
        model=model,
        train_batches=train_loader,
        validation_batches=validation_loader,
        optimizer=optimizer,
        config=config,
        device=arguments.device,
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
        },
    )
    log_sync_training(
        getattr(arguments, "_run_logger", NullRunLogger()),
        history=history,
        configuration_hash=metadata.config_hash,
        checkpoint=arguments.checkpoint,
        history_path=arguments.history,
        elapsed_seconds=time.perf_counter() - started_at,
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


def _detector_report_from_path(path: Path) -> DetectorBenchmarkReport:
    return read_detector_report(path)


def _detector_compare(arguments: argparse.Namespace) -> int:
    reports = tuple(_detector_report_from_path(path) for path in arguments.reports)
    report_hashes = {
        report.detector_name: _sha256(path)
        for path, report in zip(arguments.reports, reports, strict=True)
    }
    decision: DetectorDecision = compare_detectors(
        reports,
        input_report_sha256=report_hashes,
    )
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
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=1_000)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.set_defaults(handler=_evaluate)

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
