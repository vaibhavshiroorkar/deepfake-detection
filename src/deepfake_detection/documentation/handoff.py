"""Regenerate the parts of the handoff that describe the filesystem and the runs.

The handoff went stale the moment the data directory changed under it: it stated
that `data/` was empty and that no real-video inference was possible, while both
datasets sat on disk. Every fact in it was written by hand, so nothing kept it
honest.

The fix is to stop hand-writing the facts. The dataset inventory, the evidence
table and the MLflow summary are regenerated from what is actually there,
between markers, the same way `docs/reference/cli.md` regenerates its command
list. Prose outside the markers is left alone, because judgement does not
regenerate.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

BLOCK_MARKERS = {
    "datasets": (
        "<!-- BEGIN GENERATED DATASETS -->",
        "<!-- END GENERATED DATASETS -->",
    ),
    "evidence": (
        "<!-- BEGIN GENERATED EVIDENCE -->",
        "<!-- END GENERATED EVIDENCE -->",
    ),
    "mlflow": (
        "<!-- BEGIN GENERATED MLFLOW -->",
        "<!-- END GENERATED MLFLOW -->",
    ),
}

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# Datasets the project expects, in the order the protocol reads them.
KNOWN_DATASETS = (
    ("FakeAVCeleb_v1.2", "Primary development set"),
    ("LAV-DF", "Cross-modal stream training, localized forgeries"),
    ("Celeb-DF-v2", "Cross-dataset generalization, no audio"),
    ("DFDC", "Cross-corpus test, the only one with audio not from VoxCeleb2"),
    ("FaceForensics++", "Declared visual experiments"),
    ("MNW", "Locked external benchmark, evaluation only"),
)


@dataclass(frozen=True, slots=True)
class DatasetState:
    name: str
    role: str
    present: bool
    videos: int
    bytes_on_disk: int


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def survey_datasets(data_dir: Path) -> tuple[DatasetState, ...]:
    """Count the video files actually present under each expected dataset."""
    states = []
    for name, role in KNOWN_DATASETS:
        root = Path(data_dir) / name
        if not root.is_dir():
            states.append(DatasetState(name, role, False, 0, 0))
            continue
        videos = 0
        total = 0
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                videos += 1
                total += path.stat().st_size
        states.append(DatasetState(name, role, True, videos, total))
    return tuple(states)


def render_datasets(states: Iterable[DatasetState]) -> str:
    lines = [
        "| Dataset directory | Role | State | Videos | Size |",
        "|---|---|---|---:|---:|",
    ]
    for state in states:
        if not state.present:
            lines.append(f"| `data/{state.name}` | {state.role} | absent | 0 | 0 B |")
            continue
        lines.append(
            f"| `data/{state.name}` | {state.role} | present | {state.videos:,} | "
            f"{_human_bytes(state.bytes_on_disk)} |"
        )
    return "\n".join(lines)


def collect_evidence(runs_dir: Path) -> tuple[dict, ...]:
    """Every evaluation report under runs/, newest directory first."""
    reports = []
    for path in sorted(Path(runs_dir).rglob("*-metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("overall"), dict):
            overall = payload["overall"]
            payload = {
                **payload,
                "metrics": overall.get("metrics"),
                "rows": overall.get("scored", 0),
                "dataset": payload.get("dataset", "FakeAVCeleb"),
                "evidence_scope": payload.get("evidence_scope", "fusion"),
            }
        if "metrics" not in payload and "single_class" not in payload:
            continue
        payload["_path"] = str(path).replace("\\", "/")
        reports.append(payload)
    return tuple(reports)


def render_evidence(reports: Iterable[dict]) -> str:
    rows = [
        "| Report | Dataset | Scope | Rows | ROC-AUC | Balanced accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    found = False
    for report in reports:
        found = True
        metrics = report.get("metrics")
        if isinstance(metrics, dict):
            roc = f"{metrics.get('roc_auc', float('nan')):.4f}"
            balanced = f"{metrics.get('balanced_accuracy', float('nan')):.4f}"
        else:
            single = report.get("single_class", {})
            rate = single.get("detection_rate", single.get("specificity"))
            # One class means no ranking metric exists. Say so rather than
            # leaving a blank that reads like a missing measurement.
            roc = "undefined"
            balanced = f"{rate:.4f} ({single.get('class', '?')} only)" if rate else "n/a"
        rows.append(
            f"| `{report['_path']}` | {report.get('dataset', '?')} | "
            f"{report.get('evidence_scope', '?')} | {report.get('rows', 0):,} | "
            f"{roc} | {balanced} |"
        )
    if not found:
        return "No evaluation reports are present under `runs/`."
    return "\n".join(rows)


def collect_mlflow(database: Path) -> tuple[dict, ...]:
    """Experiment names with run counts and statuses, read without MLflow."""
    if not Path(database).is_file():
        return ()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT e.name, r.status, COUNT(*) "
            "FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
            "GROUP BY e.name, r.status ORDER BY e.name, r.status"
        ).fetchall()
    except sqlite3.DatabaseError:
        return ()
    finally:
        connection.close()
    summary: dict[str, dict[str, int]] = {}
    for name, status, count in rows:
        summary.setdefault(name, {})[status] = count
    return tuple(
        {"experiment": name, "statuses": statuses}
        for name, statuses in sorted(summary.items())
    )


def render_mlflow(experiments: Iterable[dict]) -> str:
    entries = list(experiments)
    if not entries:
        return "No MLflow database is present."
    lines = ["| Experiment | Runs |", "|---|---|"]
    for entry in entries:
        statuses = entry["statuses"]
        total = sum(statuses.values())
        detail = ", ".join(
            f"{count} {status.lower()}" for status, count in sorted(statuses.items())
        )
        lines.append(f"| `{entry['experiment']}` | {total} ({detail}) |")
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    """Swap one marked block's contents, leaving surrounding prose untouched."""
    start, end = BLOCK_MARKERS[name]
    if start not in text or end not in text:
        raise ValueError(f"Handoff is missing the {name} markers ({start} ... {end})")
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{body}\n{end}{tail}"


def render_handoff(root: Path, text: str) -> str:
    root = Path(root)
    text = replace_block(text, "datasets", render_datasets(survey_datasets(root / "data")))
    text = replace_block(text, "evidence", render_evidence(collect_evidence(root / "runs")))
    text = replace_block(text, "mlflow", render_mlflow(collect_mlflow(root / "mlflow.db")))
    return text


def update_handoff(root: Path, path: Path) -> bool:
    """Rewrite the generated blocks. True when the file changed."""
    current = Path(path).read_text(encoding="utf-8")
    updated = render_handoff(root, current)
    if updated == current:
        return False
    Path(path).write_text(updated, encoding="utf-8")
    return True
