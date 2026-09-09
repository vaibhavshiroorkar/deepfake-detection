"""Record a run directory's training and evaluation artefacts into MLflow.

`ddf run --config` opens a tracked run around a command, so anything launched
that way lands in MLflow automatically. A supervisor that invokes
`ddf train visual` directly does not, and `runs/program-20260906` was built that
way: fourteen checkpoints, a fusion model and six evaluations that exist only as
files. The dashboard reads MLflow, so none of it was visible.

This reads the history and metrics JSON those commands already write and
replays them into MLflow. It is idempotent by run name, so re-running it does
not duplicate anything.

    uv run python scripts/backfill_mlflow.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepfake_detection.experiments.scopes import validate_evidence_scope


def _numeric(values: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a nested report into the scalar metrics MLflow accepts."""
    out: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(_numeric(value, prefix=f"{name}."))
    return out


def backfill(run_dir: Path, experiment: str, tracking_uri: str) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient()
    known = {
        run.info.run_name
        for run in client.search_runs(
            [client.get_experiment_by_name(experiment).experiment_id], max_results=500
        )
    }

    recorded = 0
    for history_path in sorted((run_dir / "checkpoints").glob("*-history.json")):
        name = history_path.name.replace("-history.json", "")
        if name in known:
            print(f"  skip {name} (already recorded)")
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metadata = history.get("metadata", {})
        epochs = history.get("epochs", [])
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(
                {
                    "project": "deepfake-generalization",
                    "environment": "local",
                    "evidence_scope": validate_evidence_scope(
                        "development_comparison"
                    ),
                    "dataset": history.get("config", {}).get("dataset", "FakeAVCeleb"),
                    # Fold models exist only to produce honest out-of-fold
                    # features; separating them keeps a fold's weaker numbers
                    # from being read as a candidate result.
                    "ablation_group": (
                        "program-crossfit" if name.startswith("fold") else "program-final"
                    ),
                    "tier": "program",
                    "branch": metadata.get("branch", "unknown"),
                    "backfilled": "true",
                }
            )
            mlflow.log_params(
                {f"metadata.{k}": v for k, v in metadata.items() if v is not None}
            )
            for key, value in history.get("config", {}).items():
                if isinstance(value, dict):
                    mlflow.log_params(
                        {f"config.{key}.{k}": v for k, v in value.items()}
                    )
                else:
                    mlflow.log_param(f"config.{key}", value)
            # Per-epoch curves, so the dashboard can plot them like a live run.
            for record in epochs:
                step = record.get("epoch", 0)
                mlflow.log_metrics(
                    {
                        k: float(v)
                        for k, v in record.items()
                        if isinstance(v, int | float) and k != "epoch"
                    },
                    step=step,
                )
            if epochs:
                best = epochs[min(history.get("best_epoch", 1), len(epochs)) - 1]
                mlflow.log_metrics(
                    {
                        "training.loss": float(best.get("train_loss", 0.0)),
                        "validation.loss": float(best.get("validation_loss", 0.0)),
                        "training.best_epoch": float(history.get("best_epoch", 0)),
                    }
                )
            mlflow.log_metrics(_numeric(history.get("hardware", {}), "hardware."))
            mlflow.log_artifact(str(history_path), artifact_path="history")
        recorded += 1
        print(f"  recorded {name}")

    for metrics_path in sorted((run_dir / "evaluation").glob("*-metrics.json")):
        name = metrics_path.name.replace("-metrics.json", "") + "-evaluation"
        if name in known:
            print(f"  skip {name} (already recorded)")
            continue
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        scope = report.get("evidence_scope", "development_test")
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(
                {
                    "project": "deepfake-generalization",
                    "environment": "local",
                    "evidence_scope": validate_evidence_scope(scope),
                    "dataset": report.get("dataset", "FakeAVCeleb"),
                    "ablation_group": "program-evaluation",
                    "tier": "program",
                    "backfilled": "true",
                }
            )
            source = report.get("overall", report)
            metrics = source.get("metrics") or {}
            mlflow.log_metrics(_numeric(metrics, "evaluation."))
            for extra in ("confusion", "coverage", "class_balance"):
                if isinstance(report.get(extra), dict):
                    mlflow.log_metrics(_numeric(report[extra], f"{extra}."))
            if isinstance(report.get("detection_rate"), int | float):
                mlflow.log_metric("evaluation.detection_rate", report["detection_rate"])
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        recorded += 1
        print(f"  recorded {name}")
    return recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--tracking-uri", default=None)
    arguments = parser.parse_args(argv)

    experiment = arguments.experiment or arguments.run_dir.name
    uri = arguments.tracking_uri or f"sqlite:///{Path('mlflow.db').resolve()}"
    print(f"backfilling {arguments.run_dir} into experiment {experiment!r}")
    count = backfill(arguments.run_dir, experiment, uri)
    print(f"done: {count} runs recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
