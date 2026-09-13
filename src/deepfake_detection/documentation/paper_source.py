"""Regenerate the measured parts of the paper source book.

`docs/research/paper-source.md` is the single place to write the paper from. The
judgement in it is written by hand. Every number in it is generated from the
artifacts under `runs/`, between markers, the same way the handoff regenerates
its inventory.

The reason is the one the handoff already learned: a document with hand-typed
numbers is wrong the first time a run is repeated, and a wrong number that looks
like evidence is worse than a missing one. A table here that says "not generated
yet" names the command that would fill it, which is the honest state when an
experiment has not been run.

Every renderer takes the parsed artifact and returns markdown. None of them
raises when a file is absent: a source book that cannot be rebuilt while half
the experiments are running is a source book nobody rebuilds.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

NEWLINE = "\n"

BLOCK_MARKERS = {
    name: (f"<!-- BEGIN GENERATED {name.upper()} -->", f"<!-- END GENERATED {name.upper()} -->")
    for name in (
        "streams",
        "training",
        "batchnorm",
        "ablation",
        "correlation",
        "fusionablations",
        "thresholds",
        "designa",
        "mnw",
        "motion",
        "operating",
        "registry",
    )
}

# Where each block comes from, printed when the artifact is missing so the
# reader is told what to run rather than left with an empty table.
COMMANDS = {
    "streams": "python scripts/score_streams.py --run-dir {run}",
    "training": "pwsh scripts/train_design_b.ps1",
    "batchnorm": "python scripts/score_batchnorm_arms.py",
    "ablation": "python scripts/run_deep_ablation.py --run-dir {run}",
    "correlation": "python scripts/stream_correlation.py --run-dir {run}",
    "fusionablations": "python scripts/run_fusion_ablations.py --run-dir {run}",
    "thresholds": "python scripts/fit_gate_fusion.py --run-dir {run}",
    "designa": "pwsh scripts/run_program.ps1",
    "mnw": "ddf evaluate branch --branch visual --dataset MNW",
    "motion": "python scripts/motion_vs_error.py --dataset dfdc",
    "operating": "pwsh scripts/run_program.ps1",
    "registry": "python scripts/update_result_registry.py",
}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _missing(name: str, run_dir: Path) -> str:
    command = COMMANDS[name].format(run=run_dir.as_posix())
    return f"Not generated yet. Run `{command}`."


def _percent(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _cell(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _interval(scores: Mapping, partition: str) -> str:
    """One partition's cell, with its interval when the artifact carries one."""
    value = scores.get(partition)
    if value is None:
        return "n/a"
    bounds = scores.get(f"{partition}_ci")
    if not bounds:
        return _cell(value)
    return f"{value:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"


def render_streams(run_dir: Path) -> str:
    scores = _load(run_dir / "evaluation" / "stream-scores.json")
    if not scores:
        return _missing("streams", run_dir)
    lines = [
        "| Stream | holdout | in-domain | DFDC |",
        "|---|---|---|---|",
    ]
    for name in sorted(scores):
        row = scores[name]
        lines.append(
            f"| `{name}` | {_interval(row, 'holdout')} | "
            f"{_interval(row, 'in-domain')} | {_interval(row, 'dfdc')} |"
        )
    lines.append("")
    lines.append(
        "Labels are `clip_fake`, which the fusion store carries. The visual "
        "stream optimises `video_fake`, so its in-domain figure differs there; "
        "see the BatchNorm table."
    )
    return "\n".join(lines)


def render_training(run_dir: Path) -> str:
    """What each stream actually is, read from the history the trainer wrote."""
    directory = run_dir / "checkpoints"
    histories = sorted(directory.glob("*-history.json"))
    if not histories:
        return _missing("training", run_dir)
    lines = [
        "| Stream | Model | Optimizer | Schedule | Best epoch | Validation AUC "
        "| Diagonal mass |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for path in histories:
        history = _load(path)
        if not history or not history.get("epochs"):
            continue
        name = path.name.removesuffix("-history.json")
        config = history.get("config", {})
        model = config.get("model", {})
        optimizer = config.get("optimizer", {})
        training = config.get("training", {})
        best = history["epochs"][history["best_epoch"] - 1]
        model_text = ", ".join(
            f"{key}={value}"
            for key, value in sorted(model.items())
            if key not in {"pretrained"}
        )
        optimizer_text = (
            f"{optimizer.get('name', '?')}, head {optimizer.get('learning_rate', '?')}"
            f", encoder {optimizer.get('encoder_learning_rate', 'n/a')}"
        )
        schedule = (
            f"{len(history['epochs'])} of {training.get('epochs', '?')} epochs, "
            f"freeze {training.get('freeze_epochs', '?')}, "
            f"patience {training.get('early_stopping_patience', '?')}"
        )
        diagonal = best.get("validation_diagonal_mass")
        lines.append(
            f"| `{name}` | {model_text} | {optimizer_text} | {schedule} | "
            f"{history['best_epoch']} | {_cell(best.get('validation_auc'))} | "
            f"{_cell(diagonal) if diagonal else 'not applicable'} |"
        )
    return "\n".join(lines)


def render_batchnorm(run_dir: Path) -> str:
    payload = _load(run_dir / "evaluation" / "batchnorm-arms.json")
    if not payload:
        return _missing("batchnorm", run_dir)
    lines = [
        "| Arm | Stream | Validation | In-domain (`video_fake`) | DFDC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload.get("arms", []):
        lines.append(
            f"| {row['arm']} | `{row['stream']}` | {_cell(row.get('validation'))} | "
            f"{_cell(row.get('in-domain'))} | {_cell(row.get('dfdc'))} |"
        )
    sweep = _load(run_dir / "evaluation" / "batchnorm-sweep.json")
    if sweep:
        lines.append("")
        lines.append(
            f"Controlled sweep, {sweep.get('train_clips', '?')} training clips, "
            "epoch budget held fixed:"
        )
        lines.append("")
        lines.append("| Arm | Validation | DFDC |")
        lines.append("|---|---:|---:|")
        for arm in sweep.get("arms", []):
            lines.append(
                f"| {arm['arm']} | {_cell(arm.get('validation_auc'))} | "
                f"{_cell(arm.get('dfdc_auc'))} |"
            )
    else:
        lines.append("")
        lines.append(
            "The controlled sweep that holds the epoch budget fixed has not run. "
            f"`{COMMANDS['batchnorm'].format(run=run_dir.as_posix())}` measures "
            "the arms; `python scripts/batchnorm_sweep.py --run-dir "
            f"{run_dir.as_posix()}` adds the epoch control."
        )
    return "\n".join(lines)


def render_ablation(run_dir: Path) -> str:
    payload = _load(run_dir / "evaluation" / "deep-ablation.json")
    if not payload:
        return _missing("ablation", run_dir)
    combinations = payload.get("combinations", [])
    if not combinations:
        return _missing("ablation", run_dir)
    partitions = [p for p in ("in-domain", "dfdc") if any(p in c for c in combinations)]
    lines = [
        "| Streams | " + " | ".join(partitions) + " |",
        "|---|" + "|".join(["---"] * len(partitions)) + "|",
    ]
    for row in sorted(combinations, key=lambda r: (r["size"], r["streams"])):
        cells = " | ".join(_interval(row, p) for p in partitions)
        lines.append(f"| {' + '.join(row['streams'])} | {cells} |")

    lines.append("")
    for partition in partitions:
        scored = [c for c in combinations if c.get(partition) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda r: r[partition])
        singles = [c for c in scored if c["size"] == 1]
        best_single = max(singles, key=lambda r: r[partition]) if singles else None
        full = next((c for c in scored if c["size"] == len(best["streams"])), None)
        full = max(scored, key=lambda r: r["size"])
        lines.append(
            f"- {partition}: best is {' + '.join(best['streams'])} at "
            f"{best[partition]:.4f}; best single is "
            f"{' + '.join(best_single['streams'])} at {best_single[partition]:.4f}; "
            f"all {full['size']} streams reach {full[partition]:.4f}."
        )
    return "\n".join(lines)


def render_correlation(run_dir: Path) -> str:
    payload = _load(run_dir / "evaluation" / "stream-correlation.json")
    if not payload:
        return _missing("correlation", run_dir)
    lines = []
    for partition in sorted(payload):
        pairs = payload[partition].get("pairs", [])
        if not pairs:
            continue
        lines.append(f"**{partition}**")
        lines.append("")
        lines.append("| Pair | Logit correlation | Shared errors | Clips |")
        lines.append("|---|---:|---:|---:|")
        for pair in sorted(pairs, key=lambda p: -abs(p["logit_correlation"])):
            lines.append(
                f"| `{pair['left']}` and `{pair['right']}` | "
                f"{pair['logit_correlation']:+.3f} | "
                f"{pair['error_overlap']:.1%} | {pair['clips']:,} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_fusionablations(run_dir: Path) -> str:
    payload = _load(run_dir / "evaluation" / "fusion-ablations.json")
    if not payload:
        return _missing("fusionablations", run_dir)
    labels = {
        "late_fusion": "late fusion, complete clips only",
        "deep_abstaining": "deep fusion, abstains on partial coverage",
        "deep_masking": "deep fusion, answers from what ran",
    }
    lines = ["| Partition | Head | ROC-AUC | Coverage | Clips scored |", "|---|---|---|---:|---:|"]
    for partition, entry in sorted(payload.get("partitions", {}).items()):
        for key, label in labels.items():
            row = entry.get(key)
            if not row:
                continue
            bounds = row.get("ci")
            auc = (
                f"{row['auc']:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"
                if row.get("auc") is not None and bounds
                else _cell(row.get("auc"))
            )
            lines.append(
                f"| {partition} | {label} | {auc} | {row.get('coverage', 0):.1%} | "
                f"{row.get('scored', 0):,} |"
            )
    return "\n".join(lines)


def render_thresholds(run_dir: Path) -> str:
    payload = _load(run_dir / "checkpoints" / "gate-fusion.json")
    if not payload:
        return _missing("thresholds", run_dir)
    lines = ["| Media kind | Threshold | Rows it was chosen on |", "|---|---:|---:|"]
    for kind, value in sorted(payload.get("thresholds", {}).items()):
        rows = payload.get("threshold_rows", {}).get(kind, 0)
        lines.append(f"| {kind} | {value:.4f} | {rows:,} |")
    lines.append("")
    lines.append(
        f"Head fitted on {payload.get('fit_rows', 0):,} holdout rows over "
        f"{len(payload.get('streams', []))} streams, best epoch "
        f"{payload.get('best_epoch', '?')}, presence patterns "
        f"{payload.get('presence_patterns', {})}."
    )
    return "\n".join(lines)


def render_designa(program_run: Path) -> str:
    directory = Path(program_run) / "evaluation"
    reports = sorted(directory.glob("*-metrics.json"))
    if not reports:
        return _missing("designa", Path(program_run))
    lines = ["| Report | Dataset | Rows | ROC-AUC | Balanced accuracy |", "|---|---|---:|---:|---:|"]
    for path in reports:
        payload = _load(path)
        if not payload:
            continue
        if isinstance(payload.get("overall"), dict):
            overall = payload["overall"]
            metrics = overall.get("metrics")
            rows = overall.get("scored", 0)
            # The fusion reports nest everything under "overall" and keep the
            # dataset there, so reading only the top level prints "?".
            # The fusion reports carry no dataset field anywhere, so the
            # partition in the filename is the only honest source.
            dataset = payload.get("dataset") or overall.get("dataset") or (
                "DFDC" if "dfdc" in path.name else "FakeAVCeleb"
            )
        else:
            metrics = payload.get("metrics")
            rows = payload.get("rows", 0)
            dataset = payload.get("dataset", "?")
        if isinstance(metrics, dict):
            roc = _cell(metrics.get("roc_auc"))
            balanced = _cell(metrics.get("balanced_accuracy"))
        else:
            single = payload.get("single_class", {})
            rate = single.get("detection_rate")
            roc = "undefined, one class"
            balanced = f"{rate:.4f} detected" if rate is not None else "n/a"
        lines.append(
            f"| `{path.name}` | {dataset} | {rows:,} | {roc} | {balanced} |"
        )
    return "\n".join(lines)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """95 percent Wilson interval for a proportion.

    Wilson rather than the normal approximation because these counts are tiny,
    4 to 10 clips per generator, and the normal interval on 0 of 10 is [0, 0],
    which reads as certainty where there is none.
    """
    if total == 0:
        return (0.0, 1.0)
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def render_mnw(program_run: Path) -> str:
    """Detection per generator, which the aggregate detection rate hides."""
    path = Path(program_run) / "evaluation" / "visual-mnw-predictions.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return _missing("mnw", Path(program_run))
    if not rows:
        return _missing("mnw", Path(program_run))
    # Split on the label. MNW is described as fake-only and is not quite: one
    # clip is genuine, and counting its prediction as a detection turns a false
    # positive into a success.
    counts: dict[str, list[int]] = {}
    authentic: list[int] = [0, 0]
    for row in rows:
        called = int(row.get("predicted", 0))
        if int(row.get("label", 1)) == 0:
            authentic[1] += 1
            authentic[0] += called
            continue
        key = row.get("method") or "unknown"
        entry = counts.setdefault(key, [0, 0])
        entry[1] += 1
        entry[0] += called
    lines = [
        "| Generator | Detected | Clips | Rate | 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for key, (hit, total) in sorted(
        counts.items(), key=lambda item: item[1][0] / item[1][1]
    ):
        low, high = _wilson(hit, total)
        lines.append(
            f"| `{key}` | {hit} | {total} | {hit / total:.0%} | "
            f"[{low:.0%}, {high:.0%}] |"
        )
    detected = sum(v[0] for v in counts.values())
    total = sum(v[1] for v in counts.values())
    low, high = _wilson(detected, total)
    lines.append(
        f"| **all** | **{detected}** | **{total}** | **{detected / total:.0%}** | "
        f"[{low:.0%}, {high:.0%}] |"
    )
    named = {k: v for k, v in counts.items() if "wild" not in k}
    wild = {k: v for k, v in counts.items() if "wild" in k}
    lines.append("")
    for label, group in (("named generators", named), ("in the wild", wild)):
        if not group:
            continue
        hit = sum(v[0] for v in group.values())
        total = sum(v[1] for v in group.values())
        low, high = _wilson(hit, total)
        lines.append(
            f"- {label}: {hit} of {total}, {hit / total:.0%} "
            f"[{low:.0%}, {high:.0%}]"
        )
    if authentic[1]:
        lines.append(
            f"- the {authentic[1]} genuine clip(s) in the set: {authentic[0]} "
            "called fake"
        )
    lines.append("")
    lines.append(
        "Almost fake-only, so no ranking metric exists and the column is a "
        "detection count at the fixed threshold. Intervals are Wilson, because "
        "the per-generator counts are 4 to 10 clips and a normal interval on 0 "
        "of 10 would read as certainty. The honest reading of a single "
        "generator's row is therefore weak; the pattern across rows is what "
        "carries."
    )
    return "\n".join(lines)


def render_motion(program_run: Path) -> str:
    """Motion against the score, per corpus, with the distributions beside it."""
    directory = Path(program_run) / "evaluation"
    payloads = [
        _load(directory / f"motion-{name}.json")
        for name in ("in-domain", "dfdc", "celebdf")
    ]
    payloads = [payload for payload in payloads if payload]
    if not payloads:
        return _missing("motion", Path(program_run))
    lines = [
        "| Corpus | Clips | Mean motion | Top quartile | Correlation, "
        "manipulated | Correlation, authentic | False alarms, calm | "
        "False alarms, moving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        quartiles = payload.get("motion_quartiles") or [None, None, None]
        lines.append(
            f"| {payload.get('dataset', '?')} | {payload.get('clips', 0):,} | "
            f"{_cell(payload.get('motion_mean'))} | {_cell(quartiles[2])} | "
            f"{_cell(payload.get('spearman_manipulated'), 3)} | "
            f"{_cell(payload.get('spearman_authentic'), 3)} | "
            f"{_percent(payload.get('false_alarm_low_motion'))} | "
            f"{_percent(payload.get('false_alarm_high_motion'))} |"
        )
    lines.append("")
    lines.append(
        "Motion is the mean absolute difference between consecutive frames of "
        "the cached view, which is the tensor the model is handed. A negative "
        "correlation means more motion pushes the score towards `real`."
    )
    return NEWLINE.join(lines)


def render_operating(program_run: Path) -> str:
    """What the pipeline does at its fixed threshold, and how well calibrated."""
    directory = Path(program_run) / "evaluation"
    wanted = (
        ("fusion-test-metrics.json", "FakeAVCeleb, in-domain"),
        ("fusion-dfdc-metrics.json", "DFDC, cross-corpus"),
    )
    rows = []
    for name, label in wanted:
        payload = _load(directory / name)
        if not payload:
            continue
        overall = payload.get("overall", {})
        metrics = overall.get("metrics", {})
        paired = payload.get("fusion_vs_visual_auc", {})
        rows.append((label, metrics, paired, overall))
    if not rows:
        return _missing("operating", Path(program_run))

    lines = [
        "| Partition | ROC-AUC | Precision | Recall | FPR | FPR at 95% TPR | "
        "EER | Brier | Calibration error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics, _, _ in rows:
        lines.append(
            f"| {label} | {_cell(metrics.get('roc_auc'))} | "
            f"{_cell(metrics.get('precision'))} | {_cell(metrics.get('recall'))} | "
            f"{_cell(metrics.get('fpr'))} | {_cell(metrics.get('fpr_at_95_tpr'))} | "
            f"{_cell(metrics.get('eer'))} | {_cell(metrics.get('brier'))} | "
            f"{_cell(metrics.get('expected_calibration_error'))} |"
        )
    lines.append("")
    lines.append("Fusion against the visual baseline, paired source bootstrap:")
    lines.append("")
    lines.append("| Partition | Difference in ROC-AUC | 95% interval | Separated |")
    lines.append("|---|---:|---|---|")
    for label, _, paired, _ in rows:
        if not paired:
            continue
        lower, upper = paired.get("lower"), paired.get("upper")
        separated = (
            "yes" if lower is not None and upper is not None and lower * upper > 0
            else "no, the interval includes zero"
        )
        lines.append(
            f"| {label} | {_cell(paired.get('estimate'))} | "
            f"[{_cell(lower)}, {_cell(upper)}] | {separated} |"
        )
    lines.append("")
    lines.append(
        "The paired bootstrap is the protocol's comparison and the one the "
        "research question turns on: it resamples identities and takes the "
        "difference within each resample, so the two systems are never compared "
        "across different draws."
    )
    return NEWLINE.join(lines)


def render_registry(registry_path: Path) -> str:
    try:
        text = Path(registry_path).read_text(encoding="utf-8")
    except OSError:
        return _missing("registry", Path("runs"))
    start, end = "<!-- BEGIN GENERATED REGISTRY -->", "<!-- END GENERATED REGISTRY -->"
    if start not in text or end not in text:
        return _missing("registry", Path("runs"))
    block = text.split(start, 1)[1].split(end, 1)[0]
    rows = [line for line in block.splitlines() if line.startswith("| `")]
    statuses: dict[str, int] = {}
    for line in rows:
        status = line.rstrip("|").rsplit("|", 1)[-1].strip()
        statuses[status] = statuses.get(status, 0) + 1
    lines = [f"{len(rows)} registered results."]
    lines.append("")
    for status, count in sorted(statuses.items()):
        lines.append(f"- {count} `{status}`")
    lines.append("")
    lines.append(
        "The rows themselves, with each artifact's SHA-256 and MLflow run, are "
        "in [result traceability](result-traceability.md)."
    )
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    start, end = BLOCK_MARKERS[name]
    if start not in text or end not in text:
        raise ValueError(f"Paper source is missing the {name} markers")
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{body}\n{end}{tail}"


def build_blocks(
    *,
    run_dir: Path,
    program_run: Path,
    registry: Path,
) -> dict[str, str]:
    return {
        "streams": render_streams(run_dir),
        "training": render_training(run_dir),
        "batchnorm": render_batchnorm(run_dir),
        "ablation": render_ablation(run_dir),
        "correlation": render_correlation(run_dir),
        "fusionablations": render_fusionablations(run_dir),
        "thresholds": render_thresholds(run_dir),
        "designa": render_designa(program_run),
        "mnw": render_mnw(program_run),
        "motion": render_motion(program_run),
        "operating": render_operating(program_run),
        "registry": render_registry(registry),
    }


def update_paper_source(
    path: Path,
    *,
    run_dir: Path,
    program_run: Path,
    registry: Path,
) -> bool:
    """Rewrite every generated block. True when the file changed."""
    text = Path(path).read_text(encoding="utf-8")
    updated = text
    for name, body in build_blocks(
        run_dir=run_dir, program_run=program_run, registry=registry
    ).items():
        updated = replace_block(updated, name, body)
    if updated == text:
        return False
    Path(path).write_text(updated, encoding="utf-8")
    return True


def missing_blocks(path: Path) -> tuple[str, ...]:
    """Markers the document does not carry, for the test that guards it."""
    text = Path(path).read_text(encoding="utf-8")
    return tuple(
        name
        for name, (start, end) in BLOCK_MARKERS.items()
        if start not in text or end not in text
    )


def _iter_names() -> Iterable[str]:
    return BLOCK_MARKERS.keys()
