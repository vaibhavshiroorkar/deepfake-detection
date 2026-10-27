"""Scoring and thresholds: `ddf evaluate` and `ddf threshold`."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from deepfake_detection.cli.common import _write_json
from deepfake_detection.cli.data import _partition_by_view
from deepfake_detection.data.cache_index import read_cache_index
from deepfake_detection.data.manifest import load_manifest
from deepfake_detection.evaluation.bootstrap import (
    PairedPrediction,
    bootstrap_binary_metrics,
    paired_auc_difference,
)
from deepfake_detection.evaluation.branch_report import (
    branch_evaluation_report,
    coverage,
    flat_metrics,
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
)
from deepfake_detection.experiments.scopes import validate_evidence_scope
from deepfake_detection.views.cache_store import CacheStore


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

    from deepfake_detection.branches.registry import load_branch
    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )

    validate_evidence_scope(arguments.evidence_scope)
    manifest_records = load_manifest(
        arguments.manifest, dataset=arguments.dataset
    ).records
    cache_index = read_cache_index(arguments.cache_index)
    # The flags only decide for an old checkpoint that records no architecture.
    model, state = load_branch(
        arguments.checkpoint,
        arguments.branch,
        fallback=(
            arguments.architecture
            if arguments.branch == "visual"
            else arguments.audio_model
        ),
    )
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
            values = batch.values.to(arguments.device)
            from deepfake_detection.training.input_contract import audio_forward

            logits = audio_forward(model, values, batch.valid_samples).logits
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

    report = branch_evaluation_report(arguments, rows, state, preprocessing_hash)
    report["loss"] = loss_sum / len(rows)
    report["coverage"] = coverage(manifest_records, rows, abstained)
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
        {f"evaluation.{name}": float(value) for name, value in flat_metrics(report)}
    )
    logger.log_artifact(arguments.output, artifact_path="evaluation")
    logger.log_artifact(arguments.predictions, artifact_path="evaluation")
    return 0


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
    intervals = (
        bootstrap_binary_metrics(
            items,
            threshold=arguments.threshold,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        if overall.metrics is not None
        else None
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
            }
            if intervals is not None
            else None,
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


def register_evaluate(commands) -> None:
    evaluate = commands.add_parser("evaluate")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_branch = evaluate_commands.add_parser("branch")
    evaluate_branch.add_argument("--branch", choices=("visual", "audio"), required=True)
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
    # Used only for a checkpoint that records no architecture.
    evaluate_branch.add_argument(
        "--architecture",
        choices=("convnext-tiny", "efficientnet-b0"),
        default="efficientnet-b0",
    )
    evaluate_branch.set_defaults(handler=_evaluate_branch)
    evaluate_predictions = evaluate_commands.add_parser("predictions")
    evaluate_predictions.add_argument("--predictions", type=Path, required=True)
    evaluate_predictions.add_argument("--output", type=Path, required=True)
    evaluate_predictions.add_argument("--threshold", type=float, required=True)
    evaluate_predictions.add_argument("--bootstrap-samples", type=int, default=1_000)
    evaluate_predictions.add_argument("--seed", type=int, default=17)
    evaluate_predictions.set_defaults(handler=_evaluate)


def register_threshold(commands) -> None:
    threshold = commands.add_parser("threshold")
    threshold.add_argument("--predictions", type=Path, required=True)
    threshold.add_argument("--output", type=Path, required=True)
    threshold.set_defaults(handler=_select_threshold)
