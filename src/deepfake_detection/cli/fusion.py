"""Fusion heads and their feature stores: `ddf train fusion`, `ddf features`."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import joblib

from deepfake_detection.cli.common import _write_json, add_checkpoint_arguments
from deepfake_detection.data.cache_index import read_cache_index
from deepfake_detection.data.guards import (
    reject_evaluation_only,
    reject_evaluation_only_datasets,
)
from deepfake_detection.data.manifest import load_manifest
from deepfake_detection.digest import file_sha256
from deepfake_detection.experiments import (
    NullRunLogger,
    runtime,
)
from deepfake_detection.experiments.training_log import (
    log_fusion_training,
)
from deepfake_detection.fusion.late import FusionArtifact, FusionSample, LateFusion
from deepfake_detection.fusion.store import FeatureStore
from deepfake_detection.views.cache_store import CacheStore


def _stream_checkpoint_contract(arguments, rows, branches):
    """Keep fold training inputs distinct from the final serving checkpoints."""
    training = {
        name: sorted(
            {
                row.checkpoint_hashes[name]
                for row in rows
                if name in row.checkpoint_hashes
            }
        )
        for name in branches
    }
    if any(
        not hashes or any(not value for value in hashes) for hashes in training.values()
    ):
        raise ValueError("Deep fusion requires checkpoint hashes for every stream")
    binding = getattr(arguments, "stream_checkpoints", None)
    if binding is not None:
        paths = json.loads(Path(binding).read_text(encoding="utf-8"))
        if set(paths) != set(branches):
            raise ValueError(
                "Stream checkpoint mapping must name every fusion stream exactly once"
            )
        serving = {name: file_sha256(Path(path)) for name, path in paths.items()}
    elif {row.partition_role for row in rows} == {"holdout"} and all(
        len(hashes) == 1 for hashes in training.values()
    ):
        serving = {name: hashes[0] for name, hashes in training.items()}
    else:
        raise ValueError(
            "OOF deep fusion requires --stream-checkpoints with final checkpoint paths; fold hashes cannot bind the final models"
        )
    if {row.partition_role for row in rows} == {"holdout"} and any(
        training[name] != [serving[name]] for name in branches
    ):
        raise ValueError(
            "Holdout fusion must serve the exact checkpoints that produced its training features"
        )
    return training, serving


def _deep_fusion(arguments: argparse.Namespace, rows, branches) -> int:
    """Feature-level fusion: the whole embedding per stream, not one scalar.

    Split out from `_train_fusion` rather than folded into it because almost
    nothing is shared past the row assembly. This fits a torch module with early
    stopping against a grouped validation slice and saves a state dict; the
    sklearn path fits in closed form and saves a joblib artifact.
    """
    import torch

    from deepfake_detection.training.fusion import fit_stream_fusion

    training_hashes, serving_hashes = _stream_checkpoint_contract(
        arguments, rows, branches
    )
    runtime.seed_everything(arguments.seed, deterministic=True)
    model, history = fit_stream_fusion(
        rows=rows,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        stream_dropout=arguments.stream_dropout,
        dropout=arguments.dropout,
        hidden_sizes=tuple(arguments.hidden_sizes),
        common_dim=arguments.common_dim,
        validation_fraction=arguments.validation_fraction,
        patience=arguments.patience,
        seed=arguments.seed,
        device=arguments.device,
    )
    best = history.epochs[history.best_epoch - 1]
    print(
        f"best epoch {history.best_epoch}/{len(history.epochs)}  "
        f"val loss {best.validation_loss:.4f}  val AUC {best.validation_auc:.4f}",
        flush=True,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stream_dims": history.stream_dims,
            "common_dim": arguments.common_dim,
            "hidden_sizes": list(arguments.hidden_sizes),
            "split_hash": rows[0].split_hash,
            "preprocessing_hash": rows[0].preprocessing_hash,
            "training_checkpoint_hashes": training_hashes,
            "stream_checkpoint_hashes": serving_hashes,
        },
        arguments.output,
    )
    _write_json(
        arguments.metadata,
        {
            "samples": len(rows),
            "branches": list(branches),
            "model": "deep",
            "feature_store": str(arguments.feature_store),
            "split_hash": rows[0].split_hash,
            "preprocessing_hash": rows[0].preprocessing_hash,
            "stream_dims": history.stream_dims,
            "best_epoch": history.best_epoch,
            "train_clips": history.train_clips,
            "validation_clips": history.validation_clips,
            "epochs": [asdict(epoch) for epoch in history.epochs],
            "oof_run_ids": sorted({row.run_id for row in rows}),
            "training_checkpoint_hashes": training_hashes,
            "stream_checkpoint_hashes": serving_hashes,
        },
    )
    logger = getattr(arguments, "_run_logger", NullRunLogger())
    logger.log_params({"fusion.model": "deep", "fusion.streams": ",".join(branches)})
    logger.log_metrics(
        {
            "validation.loss": best.validation_loss,
            "validation.auc": best.validation_auc,
            "training.loss": best.train_loss,
            "training.best_epoch": float(history.best_epoch),
        }
    )
    logger.log_artifact(arguments.output, artifact_path="checkpoints")
    logger.log_artifact(arguments.metadata, artifact_path="history")
    return 0


def _train_fusion(arguments: argparse.Namespace) -> int:
    branches = tuple(arguments.branches)
    if len(set(branches)) != len(branches):
        raise ValueError("Fusion branch names must be unique")
    # Deep fusion tolerates a clip that one stream cannot read, zero-filling it
    # with a presence flag, so it does not need every stream on every row. The
    # scalar path has no such input, and strict assembly is what keeps a missing
    # branch from silently becoming a zero logit there. --drop-incomplete fits
    # the scalar head on the clips every stream could read, and the metadata
    # records how many it left out.
    drop_incomplete = arguments.drop_incomplete and arguments.model != "deep"
    rows = FeatureStore(arguments.feature_store).assemble(
        required_branches=branches,
        strict=arguments.model != "deep" and not drop_incomplete,
    )
    dropped_incomplete = 0
    if drop_incomplete:
        complete = tuple(row for row in rows if row.available)
        dropped_incomplete = len(rows) - len(complete)
        rows = complete
    # Two ways to get features a stream never trained on. `oof` cross-fits the
    # branches so every clip is scored by a checkpoint that did not see it, and
    # `holdout` scores the validation partition with a stream trained only on
    # the training partition. Cross-fitting uses more of the data; holdout costs
    # one training run instead of one per fold. Both keep the test partition
    # untouched, which is the property that matters.
    roles = {row.partition_role for row in rows}
    if not roles <= {"oof", "holdout"} or not roles:
        raise ValueError(
            "Fusion training needs feature rows a branch never trained on: "
            f"partition_role must be 'oof' or 'holdout', got {sorted(roles)}"
        )
    if roles == {"oof", "holdout"}:
        raise ValueError(
            "Fusion rows mix out-of-fold and holdout roles, so the same clip "
            "may appear under two different provenance rules"
        )
    # The feature store is a blend, so the guard runs over every dataset that
    # contributed a row rather than over a single --dataset flag.
    reject_evaluation_only_datasets(
        (row.dataset for row in rows), operation="fusion training"
    )
    split_hashes = {row.split_hash for row in rows}
    preprocessing_hashes = {row.preprocessing_hash for row in rows}
    if len(split_hashes) != 1:
        raise ValueError("Fusion rows use different split hashes")
    if len(preprocessing_hashes) != 1:
        raise ValueError("Fusion rows use different preprocessing hashes")
    if arguments.model == "deep":
        return _deep_fusion(arguments, rows, branches)
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
            "dropped_incomplete": dropped_incomplete,
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


def _load_trained_branches(arguments: argparse.Namespace):
    from deepfake_detection.branches.registry import load_design_a_branches
    from deepfake_detection.training.checkpoints import validate_branch_states

    models, states = load_design_a_branches(
        visual=arguments.visual_checkpoint,
        audio=arguments.audio_checkpoint,
        sync=arguments.sync_checkpoint,
        audio_fallback=arguments.audio_model,
    )
    validate_branch_states(states)
    return models["visual"], models["audio"], models["sync"], states


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
    cache_index = read_cache_index(arguments.cache_index)
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
            "visual": file_sha256(arguments.visual_checkpoint),
            "audio": file_sha256(arguments.audio_checkpoint),
            "sync": file_sha256(arguments.sync_checkpoint),
        },
        split_hash=states["visual"].metadata.split_hash,
        preprocessing_hash=states["visual"].metadata.preprocessing_hash,
        partition_role=arguments.partition_role,
        run_id=arguments.run_id,
        device=arguments.device,
        trained_source_identities={
            branch: state.metadata.trained_source_identities
            for branch, state in states.items()
        },
        selection_source_identities={
            branch: state.metadata.selection_source_identities
            for branch, state in states.items()
        },
    )
    _write_json(arguments.report, asdict(report))
    return (
        2
        if report.unavailable_rows
        and not getattr(arguments, "allow_unavailable", False)
        else 0
    )


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


def register_features(commands) -> None:
    features = commands.add_parser("features")
    feature_commands = features.add_subparsers(dest="feature_command", required=True)
    feature_export = feature_commands.add_parser("export")
    feature_export.add_argument("--manifest", type=Path, required=True)
    feature_export.add_argument("--cache-index", type=Path, required=True)
    feature_export.add_argument("--cache-root", type=Path, required=True)
    feature_export.add_argument("--feature-store", type=Path, required=True)
    feature_export.add_argument("--report", type=Path, required=True)
    feature_export.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success after recording unavailable modality rows and coverage.",
    )
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
