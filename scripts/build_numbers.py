"""Collect every reported number from results/ into one file, with its source.

The paper, the report and the slides never contain a typed number. They hold
placeholders such as {{audit.fakeavceleb.raw.clip}}, filled from
results/numbers.json at build time. Rerun this after any result changes, then
rebuild the deliverables, and every document agrees with the evidence.

    uv run python scripts/build_numbers.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _put(numbers: dict, key: str, value, source: Path) -> None:
    numbers[key] = {"value": value, "source": _rel(source)}


def audits(numbers: dict) -> None:
    names = {"fakeavceleb": "fakeavceleb.raw", "fakeavceleb-neutral-v1": "fakeavceleb.v1",
             "fakeavceleb-neutral": "fakeavceleb.neutral", "fakeavceleb-neutral-v4-check": "fakeavceleb.v4check",
             "lavdf-files": "lavdf.raw", "dfdc": "dfdc.raw", "hifiavdf": "hifiavdf.raw",
             "hifiavdf-neutral": "hifiavdf.neutral", "dfdc-neutral": "dfdc.neutral",
             "lavdf-test-neutral": "lavdf.neutral"}
    for folder, key in names.items():
        path = RESULTS / "shortcut-audit" / folder / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        _put(numbers, f"audit.{key}.clips", summary["clips_read"], path)
        for label, short in (("clip_fake", "clip"), ("video_fake", "video"), ("audio_fake", "audio")):
            if label in summary:
                _put(numbers, f"audit.{key}.{short}", round(summary[label]["shortcut_ceiling_auc"], 3), path)
        census = summary.get("codec_census", {})
        for kind, entry in census.items():
            _put(numbers, f"audit.{key}.census.{kind}.clips", entry["clips"], path)


def neutralization(numbers: dict) -> None:
    path = RESULTS / "shortcut-audit" / "fakeavceleb-neutral" / "neutralize-summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        _put(numbers, "neutral.fakeavceleb.kept", summary["rows_out"], path)
        _put(numbers, "neutral.fakeavceleb.refused", summary["rows_failed"], path)


def metrics_file(numbers: dict, prefix: str, path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["overall"]["metrics"]
    intervals = report["confidence_intervals"]
    _put(numbers, f"{prefix}.auc", round(overall["roc_auc"], 3), path)
    _put(numbers, f"{prefix}.auc_low", round(intervals["roc_auc"]["lower"], 3), path)
    _put(numbers, f"{prefix}.auc_high", round(intervals["roc_auc"]["upper"], 3), path)
    _put(numbers, f"{prefix}.fpr", round(overall["fpr"], 3), path)
    _put(numbers, f"{prefix}.balanced_accuracy", round(overall["balanced_accuracy"], 3), path)
    _put(numbers, f"{prefix}.n", report["overall"]["total"], path)
    for method, entry in report.get("per_method", {}).items():
        if entry.get("metrics"):
            _put(numbers, f"{prefix}.method.{method}", round(entry["metrics"]["roc_auc"], 3), path)


def sync(numbers: dict) -> None:
    for folder, base in (("p001-sync", "sync"), ("p001-sync-targets", "sync")):
        directory = RESULTS / folder
        if not directory.exists():
            continue
        for path in sorted(directory.glob("sync-*-seed*-metrics.json")):
            part, seed = path.stem.removeprefix("sync-").removesuffix("-metrics").rsplit("-seed", 1)
            metrics_file(numbers, f"{base}.{part}.seed{seed}", path)


def h4(numbers: dict) -> None:
    directory = RESULTS / "p001-h4"
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = path.stem.replace("sync-", "h4.")
        for name, values in report["results"].items():
            _put(numbers, f"{key}.{name}.fpr", round(values["false_positive_rate"], 3), path)
            _put(numbers, f"{key}.{name}.tpr", round(values["true_positive_rate"], 3), path)
        _put(numbers, f"{key}.success", report["h4_success"], path)


def main() -> int:
    numbers: dict = {}
    audits(numbers)
    neutralization(numbers)
    sync(numbers)
    h4(numbers)
    out = RESULTS / "numbers.json"
    out.write_text(json.dumps(dict(sorted(numbers.items())), indent=1) + "\n", encoding="utf-8")
    print(f"{len(numbers)} numbers -> {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
