"""The report `ddf evaluate branch` writes for one branch on one partition.

Separate from the command so scripts and tests can build the same report
without going through argparse.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from deepfake_detection.digest import file_sha256


def coverage(manifest_records, rows, abstained: dict[str, str]) -> dict[str, object]:
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


# Below this many rows in the smaller class, a ranking metric is noise.
MINIMUM_CLASS_ROWS = 10


def branch_evaluation_report(
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
        "checkpoint_sha256": file_sha256(arguments.checkpoint),
        "checkpoint_run_id": state.metadata.run_id,
        "dataset": arguments.dataset,
        "evidence_scope": arguments.evidence_scope,
        "fixed_threshold": arguments.threshold,
        "rows": len(rows),
        "confusion": confusion,
        "preprocessing_hash": preprocessing_hash,
        "trained_on_split_hash": state.metadata.split_hash,
        "per_method": grouped_rates(rows, "method"),
        "per_manipulation_type": grouped_rates(rows, "manipulation_type"),
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


def grouped_rates(rows: list[dict[str, object]], key: str) -> dict[str, object]:
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


def flat_metrics(report: dict[str, object]):
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
