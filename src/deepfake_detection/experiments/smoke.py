from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deepfake_detection.evaluation.metrics import (
    binary_metrics,
    select_balanced_accuracy_threshold,
)
from deepfake_detection.experiments.tracking import NullRunLogger, RunLogger
from deepfake_detection.fusion.late import FusionSample, LateFusion

_BRANCH_NAMES = ("visual", "audio", "sync")
_EVIDENCE_SCOPE = "software_fixture_only"
_PREDICTION_FIELDS = (
    "source_identity",
    "partition",
    "label",
    "visual_logit",
    "audio_logit",
    "sync_logit",
    "face_coverage",
    "audio_clipped",
    "av_duration_delta_sec",
    "probability",
)


@dataclass(frozen=True, slots=True)
class SmokeReport:
    seed: int
    samples: int
    train_samples: int
    validation_samples: int
    threshold: float
    metrics: dict[str, float]
    artifact_hashes: dict[str, str]
    evidence_scope: str = _EVIDENCE_SCOPE


@dataclass(frozen=True, slots=True)
class _FixtureRow:
    source_identity: str
    partition: str
    label: int
    sample: FusionSample


def run_fusion_smoke(
    output_dir: Path,
    *,
    seed: int,
    samples: int,
    logger: RunLogger | None = None,
) -> SmokeReport:
    """Run a deterministic software fixture, not a research experiment."""
    _validate_samples(samples)
    active_logger = logger if logger is not None else NullRunLogger()
    rows = _fixture_rows(seed=seed, samples=samples)
    fit_rows = [row for row in rows if row.partition == "fit"]
    validation_rows = [row for row in rows if row.partition == "validation"]
    _assert_disjoint_balanced_groups(fit_rows, validation_rows)

    fusion = LateFusion(branch_names=_BRANCH_NAMES).fit(
        [row.sample for row in fit_rows], [row.label for row in fit_rows]
    )
    fit_probabilities = fusion.predict_proba([row.sample for row in fit_rows])
    selection = select_balanced_accuracy_threshold(
        labels=[row.label for row in fit_rows],
        probabilities=fit_probabilities,
    )
    validation_probabilities = fusion.predict_proba(
        [row.sample for row in validation_rows]
    )
    metrics = asdict(
        binary_metrics(
            labels=[row.label for row in validation_rows],
            probabilities=validation_probabilities,
            threshold=selection.threshold,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_path = output_dir / "fusion.joblib"
    predictions_path = output_dir / "predictions.csv"
    report_path = output_dir / "smoke-report.json"
    _atomic_write(fusion_path, lambda temporary: joblib.dump(fusion, temporary))
    _atomic_write(
        predictions_path,
        lambda temporary: _write_predictions(
            temporary,
            rows,
            fit_probabilities=fit_probabilities,
            validation_probabilities=validation_probabilities,
        ),
    )
    payload_artifact_hashes = {
        fusion_path.name: _file_sha256(fusion_path),
        predictions_path.name: _file_sha256(predictions_path),
    }
    report = SmokeReport(
        seed=seed,
        samples=samples,
        train_samples=len(fit_rows),
        validation_samples=len(validation_rows),
        threshold=selection.threshold,
        metrics=metrics,
        artifact_hashes=payload_artifact_hashes,
    )
    _atomic_write(
        report_path,
        lambda temporary: _write_report(
            temporary,
            report,
            fit_rows=fit_rows,
            validation_rows=validation_rows,
        ),
    )
    report_byte_hash = _file_sha256(report_path)
    _log_smoke_evidence(
        active_logger,
        report,
        report_byte_hash=report_byte_hash,
        artifact_paths=(fusion_path, predictions_path, report_path),
    )
    return report


def _validate_samples(samples: int) -> None:
    if samples < 16 or samples % 8:
        raise ValueError("Smoke samples must be at least 16 and a multiple of 8")


def _fixture_rows(*, seed: int, samples: int) -> list[_FixtureRow]:
    generator = np.random.default_rng(seed)
    pairs = samples // 2
    pairs_per_group = [pairs // 8 + int(index < pairs % 8) for index in range(8)]
    rows: list[_FixtureRow] = []
    for group_index, group_pairs in enumerate(pairs_per_group):
        partition = "fit" if group_index < 6 else "validation"
        source_identity = f"fixture-source-{group_index}"
        for label in (0, 1):
            for _ in range(group_pairs):
                rows.append(
                    _FixtureRow(
                        source_identity=source_identity,
                        partition=partition,
                        label=label,
                        sample=_fixture_sample(generator, label),
                    )
                )
    return rows


def _fixture_sample(generator: np.random.Generator, label: int) -> FusionSample:
    signal = 1.0 if label else -1.0
    branch_logits = {
        "visual": float(1.6 * signal + generator.normal(0.0, 0.22)),
        "audio": float(1.3 * signal + generator.normal(0.0, 0.24)),
        "sync": float(1.1 * signal + generator.normal(0.0, 0.26)),
    }
    return FusionSample(
        branch_logits=branch_logits,
        face_coverage=float(np.clip(generator.normal(0.88, 0.04), 0.6, 0.99)),
        audio_clipped=bool(generator.random() < 0.08),
        av_duration_delta_sec=float(generator.normal(0.0, 0.012)),
    )


def _assert_disjoint_balanced_groups(
    fit_rows: Sequence[_FixtureRow], validation_rows: Sequence[_FixtureRow]
) -> None:
    fit_sources = {row.source_identity for row in fit_rows}
    validation_sources = {row.source_identity for row in validation_rows}
    if fit_sources & validation_sources:
        raise RuntimeError("Smoke fixture source groups must be disjoint")
    for source in fit_sources | validation_sources:
        source_rows = [
            row
            for row in (*fit_rows, *validation_rows)
            if row.source_identity == source
        ]
        if [row.label for row in source_rows].count(0) != [
            row.label for row in source_rows
        ].count(1):
            raise RuntimeError("Smoke fixture source groups must balance binary labels")


def _write_predictions(
    path: Path,
    rows: Sequence[_FixtureRow],
    *,
    fit_probabilities: np.ndarray,
    validation_probabilities: np.ndarray,
) -> None:
    probabilities = iter((*fit_probabilities, *validation_probabilities))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PREDICTION_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            probability = float(next(probabilities))
            writer.writerow(
                {
                    "source_identity": row.source_identity,
                    "partition": row.partition,
                    "label": row.label,
                    "visual_logit": row.sample.branch_logits["visual"],
                    "audio_logit": row.sample.branch_logits["audio"],
                    "sync_logit": row.sample.branch_logits["sync"],
                    "face_coverage": row.sample.face_coverage,
                    "audio_clipped": row.sample.audio_clipped,
                    "av_duration_delta_sec": row.sample.av_duration_delta_sec,
                    "probability": probability,
                }
            )


def _write_report(
    path: Path,
    report: SmokeReport,
    *,
    fit_rows: Sequence[_FixtureRow],
    validation_rows: Sequence[_FixtureRow],
) -> None:
    payload: dict[str, Any] = {
        "artifact_hashes": report.artifact_hashes,
        "evidence_scope": report.evidence_scope,
        "evaluation_partition": "validation",
        "fit_source_identities": sorted({row.source_identity for row in fit_rows}),
        "metric_evidence_scope": {
            name: report.evidence_scope for name in sorted(report.metrics)
        },
        "metrics": report.metrics,
        "samples": report.samples,
        "seed": report.seed,
        "threshold": report.threshold,
        "threshold_selection_partition": "fit",
        "train_samples": report.train_samples,
        "validation_samples": report.validation_samples,
        "validation_source_identities": sorted(
            {row.source_identity for row in validation_rows}
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        write(temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_smoke_evidence(
    logger: RunLogger,
    report: SmokeReport,
    *,
    report_byte_hash: str,
    artifact_paths: Sequence[Path],
) -> None:
    logger.log_params(
        {
            "smoke.evidence_scope": report.evidence_scope,
            "smoke.seed": report.seed,
            "smoke.samples": report.samples,
            "smoke.fit_samples": report.train_samples,
            "smoke.validation_samples": report.validation_samples,
            **{
                f"smoke.payload.{name}.sha256": value
                for name, value in report.artifact_hashes.items()
            },
            "smoke.report_sha256": report_byte_hash,
        }
    )
    logger.log_metrics(
        {f"smoke.validation.{name}": value for name, value in report.metrics.items()}
    )
    for path in artifact_paths:
        logger.log_artifact(path)
