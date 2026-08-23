from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import joblib

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
from deepfake_detection.fusion.late import FusionArtifact, FusionSample, LateFusion
from deepfake_detection.fusion.store import FeatureStore
from deepfake_detection.training.crossfit import build_group_folds
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.face_detector import MTCNNFaceDetector
from deepfake_detection.views.media import FFmpegMediaDecoder
from deepfake_detection.views.preprocessor import Preprocessor
from deepfake_detection.views.timeline import ViewConfig


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
    config = ViewConfig(remove_leading_silence=not arguments.keep_leading_silence)
    preprocessor = Preprocessor(
        decoder=FFmpegMediaDecoder(),
        detector=MTCNNFaceDetector(
            confidence=config.detector_confidence,
            device=arguments.device,
        ),
        config=config,
        code_version=arguments.code_version,
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


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_branch_train(arguments: argparse.Namespace) -> int:
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

    _seed_everything(arguments.seed)
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
        config_hash=hash_config(run_config),
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
    return 0


def _sync_branch_train(arguments: argparse.Namespace) -> int:
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

    _seed_everything(arguments.seed)
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
        config_hash=hash_config(run_config),
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
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    from deepfake_detection.inference.predictor import PredictionEngine
    from deepfake_detection.training.checkpoints import validate_branch_states
    from deepfake_detection.views.cache import preprocessing_config_hash

    visual, audio, sync, states = _load_trained_branches(arguments)
    config = ViewConfig()
    preprocessor = Preprocessor(
        decoder=FFmpegMediaDecoder(),
        detector=MTCNNFaceDetector(
            confidence=config.detector_confidence,
            device=arguments.device,
        ),
        config=config,
        code_version=arguments.code_version,
    )
    fusion = joblib.load(arguments.fusion_model)
    if not isinstance(fusion, FusionArtifact):
        raise ValueError("Fusion model does not contain provenance metadata")
    provenance = validate_branch_states(states)
    runtime_preprocessing_hash = preprocessing_config_hash(
        config=config,
        code_version=arguments.code_version,
    )
    if runtime_preprocessing_hash != provenance.preprocessing_hash:
        raise ValueError("Runtime preprocessing does not match the checkpoints")
    fusion.validate_provenance(
        split_hash=provenance.split_hash,
        preprocessing_hash=provenance.preprocessing_hash,
    )
    engine = PredictionEngine(
        preprocessor=preprocessor,
        visual_model=visual,
        audio_model=audio,
        sync_model=sync,
        fusion=fusion,
        threshold=arguments.threshold,
        device=arguments.device,
    )
    result = engine.predict(arguments.video)
    _write_json(arguments.output, asdict(result))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddf")
    commands = parser.add_subparsers(dest="command", required=True)

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
    cache_build.set_defaults(handler=_cache_build)

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
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
