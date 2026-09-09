"""Read-only access to the local MLflow store.

The dashboard never writes a run. It reads what `ddf run` already recorded, so
the Experiments page can compare runs and the Streams pages can pull a
checkpoint straight out of one instead of asking for a file path.

mlflow is imported inside each function, so the dashboard still starts in an
environment without the tracking extra installed and says so at the point of
use rather than at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepfake_detection.dashboard.paths import MLFLOW_DB

# Preferred column order, not the set of columns. Anything a run recorded that
# is not named here still appears, after these, sorted. That matters because
# these tuples used to be the whole vocabulary: a cross-dataset metric or a new
# ablation parameter was dropped from the table with no error, so the page
# quietly showed less than the store held.
COMPARISON_METRICS = (
    "validation.loss",
    "training.loss",
    "training.samples_per_second",
    "evaluation.roc_auc",
    "evaluation.pr_auc",
    "evaluation.balanced_accuracy",
    "evaluation.f1",
)

# Parameters that distinguish one training run from another. `ddf run` records
# every CLI flag under an "arguments." prefix, which is dropped in the table.
COMPARISON_PARAMS = (
    "arguments.seed",
    "arguments.learning-rate",
    "arguments.epochs",
    "arguments.batch-size",
    "arguments.freeze-epochs",
    "training.best_epoch",
    "training.elapsed_seconds",
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    run_name: str
    experiment: str
    status: str
    start_time: int
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]


def tracking_uri(database: Path | None = None) -> str:
    """The local SQLite tracking store, as an MLflow URI."""
    path = (database or MLFLOW_DB).resolve().as_posix()
    return f"sqlite:///{path}"


def _client(database: Path | None = None):
    try:
        from mlflow import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "mlflow is not installed in this environment, so the tracking store "
            "cannot be read. Install it with `uv sync --extra tracking`."
        ) from error
    return MlflowClient(tracking_uri=tracking_uri(database))


def experiments(database: Path | None = None) -> list[str]:
    """Experiment names in the store, newest first, excluding the empty default."""
    client = _client(database)
    found = client.search_experiments()
    names = [experiment.name for experiment in found if experiment.name != "Default"]
    return names


def runs(
    experiment: str | None = None,
    *,
    database: Path | None = None,
    limit: int = 200,
) -> list[RunSummary]:
    """Runs in one experiment, or in every experiment, newest first."""
    client = _client(database)
    found = client.search_experiments()
    by_id = {item.experiment_id: item.name for item in found}
    if experiment is not None:
        by_id = {key: name for key, name in by_id.items() if name == experiment}
    if not by_id:
        return []
    rows = client.search_runs(
        list(by_id),
        max_results=limit,
        order_by=["attributes.start_time DESC"],
    )
    return [
        RunSummary(
            run_id=row.info.run_id,
            run_name=row.info.run_name or "",
            experiment=by_id.get(row.info.experiment_id, ""),
            status=row.info.status,
            start_time=int(row.info.start_time or 0),
            params=dict(row.data.params),
            metrics=dict(row.data.metrics),
            tags=dict(row.data.tags),
        )
        for row in rows
    ]


def comparison_rows(summaries: list[RunSummary]) -> list[dict[str, object]]:
    """One flat dict per run: identity, the params that vary, then the metrics.

    Built for `st.dataframe`, so every row carries the same keys and a value the
    run did not record comes back as None rather than being absent.
    """
    params = _ordered_columns(
        COMPARISON_PARAMS, (summary.params for summary in summaries)
    )
    metrics = _ordered_columns(
        COMPARISON_METRICS, (summary.metrics for summary in summaries)
    )
    rows = []
    for summary in summaries:
        row: dict[str, object] = {
            "run": summary.run_name or summary.run_id[:8],
            "status": summary.status,
            "experiment": summary.experiment,
        }
        for name in params:
            row[name.removeprefix("arguments.")] = summary.params.get(name)
        for name in metrics:
            row[name] = summary.metrics.get(name)
        row["run_id"] = summary.run_id
        rows.append(row)
    return rows


def _ordered_columns(preferred, recorded) -> tuple[str, ...]:
    """Preferred names first, then whatever else the runs actually recorded.

    Keeping the preferred order means the familiar columns stay where they were,
    while a metric nobody thought to hardcode still reaches the table.
    """
    seen: set[str] = set()
    for mapping in recorded:
        seen.update(mapping)
    # Preferred columns are kept even when no run recorded them, so the table
    # keeps a stable shape and a missing value reads as blank rather than as a
    # vanished column.
    extra = sorted(seen.difference(preferred))
    return tuple(preferred) + tuple(extra)


def download_artifacts(reference: str, destination: Path) -> Path:
    """Download one run's artifacts, or the single file a runs:/ URI names.

    Returns the local path. A bare run id gives a directory; a
    `runs:/<run_id>/<path>` URI gives whatever that path is.
    """
    try:
        from mlflow.artifacts import download_artifacts as _download
    except ImportError as error:
        raise RuntimeError(
            "mlflow is not installed in this environment, so an artifact cannot "
            "be pulled. Install it with `uv sync --extra tracking`, or point at a "
            "local file under checkpoints/ instead."
        ) from error

    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    reference = reference.strip()
    destination.mkdir(parents=True, exist_ok=True)
    if reference.startswith("runs:/"):
        local = _download(artifact_uri=reference, dst_path=str(destination))
    else:
        local = _download(run_id=reference, dst_path=str(destination))
    return Path(local)
