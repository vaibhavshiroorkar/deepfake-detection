"""Collect every reported number from results/ into one file, with its source.

The paper, the report and the slides never contain a typed number. They hold
placeholders such as {{audit.fakeavceleb.raw.clip}}, filled from
results/numbers.json at build time. Rerun this after any result changes, then
rebuild the deliverables, and every document agrees with the evidence.

    uv run python scripts/build_numbers.py
"""

from __future__ import annotations

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
                for cue, auc in summary[label]["cue_auc"].items():
                    if auc == auc:  # skip NaN cues
                        _put(numbers, f"audit.{key}.cue.{short}.{cue}", round(auc, 3), path)
        census = summary.get("codec_census", {})
        for kind, entry in census.items():
            _put(numbers, f"audit.{key}.census.{kind}.clips", entry["clips"], path)
            for field in ("video_codec", "audio_codec"):
                for codec, count in entry.get(field, {}).items():
                    _put(numbers, f"audit.{key}.census.{kind}.{field}.{codec or 'none'}", count, path)


def neutralization(numbers: dict) -> None:
    path = RESULTS / "shortcut-audit" / "fakeavceleb-neutral" / "neutralize-summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        _put(numbers, "neutral.fakeavceleb.kept", summary["rows_out"], path)
        _put(numbers, "neutral.fakeavceleb.refused", summary["rows_failed"], path)
        for kind, reasons in summary.get("refused_by_category", {}).items():
            _put(numbers, f"neutral.fakeavceleb.refused.{kind}", sum(reasons.values()), path)
        for kind, count in summary.get("rows_in_by_category", {}).items():
            _put(numbers, f"neutral.fakeavceleb.in.{kind}", count, path)


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


def sync_training(numbers: dict) -> None:
    path = RESULTS / "p001-sync" / "training.json"
    if path.exists():
        log = json.loads(path.read_text(encoding="utf-8"))
        first = next(iter(log.values()))
        _put(numbers, "sync.train.real_windows", first["real_train_windows"], path)
        _put(numbers, "sync.train.real_val_windows", first["real_val_windows"], path)


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
        _put(numbers, f"{key}.success", _yes(report["h4_success"]), path)


def _yes(flag: bool) -> str:
    # Pass and fail print as words; a bare True or False would land in the paper as code.
    return "yes" if flag else "no"


def branches(numbers: dict) -> None:
    """Visual and audio paths: branch.part.seedS.*, scored on the label each path was trained for."""
    directory = RESULTS / "p001-branches"
    if not directory.exists():
        return
    for path in sorted(directory.glob("*-seed*-*-metrics.json")):
        branch, rest = path.stem.removesuffix("-metrics").split("-seed", 1)
        seed, part = rest.split("-", 1)
        metrics_file(numbers, f"{branch}.{part}.seed{seed}", path)


def h2(numbers: dict) -> None:
    """H2 (amendment A8): fusion minus visual-only per target, the Holm decision and the fusion weights."""
    path = RESULTS / "p001-h2" / "h2.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    for target, result in report["targets"].items():
        key = f"h2.{target}"
        for field in ("estimate", "lower", "upper"):
            _put(numbers, f"{key}.{field}", round(result[field], 3), path)
        _put(numbers, f"{key}.p", round(result["p_one_sided"], 3), path)
        _put(numbers, f"{key}.clips", result["clips"], path)
        _put(numbers, f"{key}.identities", result["identities"], path)
        for seed, diff in zip(report["seeds"], result["per_seed"], strict=True):
            _put(numbers, f"{key}.seed{seed}", round(diff, 3), path)
    for target, decision in report["holm"].items():
        _put(numbers, f"h2.{target}.holm", _yes(decision["rejected_at_0.05"]), path)
    _put(numbers, "h2.success", _yes(report["h2_success"]), path)
    for seed, entry in report["weights"].items():
        for branch, coef in zip(entry["branches"], entry["coef"], strict=True):
            _put(numbers, f"h2.weights.seed{seed}.{branch}", round(coef, 3), path)


def main() -> int:
    numbers: dict = {}
    audits(numbers)
    neutralization(numbers)
    sync(numbers)
    sync_training(numbers)
    h4(numbers)
    branches(numbers)
    h2(numbers)
    out = RESULTS / "numbers.json"
    out.write_text(json.dumps(dict(sorted(numbers.items())), indent=1) + "\n", encoding="utf-8")
    print(f"{len(numbers)} numbers -> {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
