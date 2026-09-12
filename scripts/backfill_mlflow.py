"""Put the Design B runs into MLflow, from the records they already wrote.

Tracking is wired to `ddf run --config`, and Design B was trained by calling the
CLI directly, so none of its five streams has an MLflow run. Every number the
paper will quote from them is therefore untraceable by the rule
`docs/research/result-traceability.md` sets: a result must resolve to an
analysis command, content hashes, and a run in the store.

Nothing here is recomputed or invented. Each run is built from one
`*-history.json` and the checkpoint it names, both written by the trainer at the
time, and the run is tagged `backfilled` with the file it came from and that
file's SHA-256. A reader can tell a backfilled run from a live one, which is the
point: a backfill that looked like a live run would be worse than no run.

Re-running is safe. A stream whose checkpoint hash already has a run is skipped,
so after a retrain this adds the new checkpoints and leaves the old ones alone.

    uv run python scripts/backfill_mlflow.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_EXPERIMENT = "design-b-20260910"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_params(history: dict) -> dict[str, str]:
    values: dict[str, str] = {}
    for section, entries in history.get("config", {}).items():
        if isinstance(entries, dict):
            for key, value in entries.items():
                values[f"{section}.{key}"] = str(value)
        else:
            values[section] = str(entries)
    for key, value in history.get("metadata", {}).items():
        values[f"metadata.{key}"] = str(value)
    return values


def _epoch_metrics(history: dict) -> list[dict[str, float]]:
    """Per-epoch metrics, so a backfilled run carries its curve and not a point.

    `diagonal_mass` is included where the trainer recorded it. It is the measure
    that showed a stream reaching 0.9991 AUC with attention that never moved,
    and it is only evidence because it was never optimised, so it belongs in the
    store beside the loss it was recorded next to.
    """
    rows = []
    for epoch in history.get("epochs", []):
        row = {
            key: float(value)
            for key, value in epoch.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default=None)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Directory of checkpoints and history files. Defaults to the run's "
        "own, and is how the frozen-BatchNorm arm gets its own runs.",
    )
    arguments = parser.parse_args(argv)

    directory = arguments.checkpoints or arguments.run_dir / "checkpoints"
    histories = sorted(directory.glob("*-history.json"))
    if not histories:
        print(f"No history files in {directory}")
        return 1

    mlflow.set_tracking_uri(arguments.tracking_uri)
    experiment = arguments.experiment or arguments.run_dir.name
    mlflow.set_experiment(experiment)

    existing = set()
    found = mlflow.search_runs(
        experiment_names=[experiment], output_format="list", max_results=5000
    )
    for run in found:
        value = run.data.tags.get("checkpoint_hash")
        if value:
            existing.add(value)

    written = 0
    for history_path in histories:
        name = history_path.name.removesuffix("-history.json")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        checkpoint = history_path.with_name(f"{name}.pt")
        checkpoint_hash = str(history.get("checkpoint_hash", ""))
        if checkpoint_hash in existing:
            print(f"{name}: already in {experiment}, skipping")
            continue

        with mlflow.start_run(run_name=name):
            mlflow.log_params(_flat_params(history))
            for index, row in enumerate(_epoch_metrics(history), start=1):
                mlflow.log_metrics(row, step=index)
            best = history.get("best_epoch")
            if best and history.get("epochs"):
                final = history["epochs"][int(best) - 1]
                mlflow.log_metrics(
                    {
                        f"best.{key}": float(value)
                        for key, value in final.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
                mlflow.log_metric("best.epoch", float(best))
            for key, value in (history.get("hardware") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(f"hardware.{key}", float(value))
            mlflow.set_tags(
                {
                    "backfilled": "true",
                    "backfill_source": str(history_path.as_posix()),
                    "backfill_source_sha256": _sha256(history_path),
                    "checkpoint_hash": checkpoint_hash,
                    "checkpoint_present": str(checkpoint.is_file()).lower(),
                    "split_hash": str(
                        history.get("metadata", {}).get("split_hash", "")
                    ),
                    "preprocessing_hash": str(
                        history.get("metadata", {}).get("preprocessing_hash", "")
                    ),
                    "git_commit": str(
                        history.get("metadata", {}).get("git_commit", "")
                    ),
                }
            )
            mlflow.log_dict(history, "history.json")
            print(f"{name}: logged {len(history.get('epochs', []))} epochs")
            written += 1

    print(f"\n{written} run(s) written to experiment {experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
