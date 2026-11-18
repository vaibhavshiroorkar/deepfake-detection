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
             "lavdf-test-neutral": "lavdf.neutral",
             "lavdf-window-neutral": "lavdf.window",   # LAV-DF by window label (A9)
             "fakeavceleb-neutral-cbr-check": "fakeavceleb.cbrcheck",   # A18, check sample
             "fakeavceleb-neutral-cbr": "fakeavceleb.cbr"}              # A18, all of FakeAVCeleb
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
    _put(numbers, f"{prefix}.n", report["overall"]["total"], path)
    # n counts every clip in the set; scored leaves out clips with no usable window (no face, no audio).
    _put(numbers, f"{prefix}.scored", report["overall"].get("scored", report["overall"]["total"]), path)
    if overall is None:
        # A set with one class has no AUC (the audio path on DFDC: every clip has real audio).
        _put(numbers, f"{prefix}.single_class", "yes", path)
        print(f"{prefix}: one class in the set, no AUC ({_rel(path)})")
        return
    intervals = report["confidence_intervals"]
    _put(numbers, f"{prefix}.auc", round(overall["roc_auc"], 3), path)
    _put(numbers, f"{prefix}.auc_low", round(intervals["roc_auc"]["lower"], 3), path)
    _put(numbers, f"{prefix}.auc_high", round(intervals["roc_auc"]["upper"], 3), path)
    _put(numbers, f"{prefix}.fpr", round(overall["fpr"], 3), path)
    _put(numbers, f"{prefix}.balanced_accuracy", round(overall["balanced_accuracy"], 3), path)
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


def _auc(labels: list[int], scores: list[float]) -> float:
    """ROC-AUC as the Mann-Whitney statistic, with tied scores sharing their average rank."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def h2_levels(numbers: dict) -> None:
    """Each path's AUC on the clip label, on the clips H2 compares, so the differences can be read
    against their levels: h2.{target}.seed{S}.auc.{path}, and the min and max over seeds."""
    directory = RESULTS / "p001-h2"
    paths = {"fusion": "fusion", "visual": "visual_logit", "audio": "audio_logit", "sync": "sync_logit"}
    by_target: dict[str, dict[str, list[float]]] = {}
    for path in sorted(directory.glob("fusion-*-seed*-predictions.csv")):
        target, seed = path.stem.removeprefix("fusion-").removesuffix("-predictions").rsplit("-seed", 1)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = [int(row["label"]) for row in rows]
        for name, column in paths.items():
            value = round(_auc(labels, [float(row[column]) for row in rows]), 3)
            _put(numbers, f"h2.{target}.seed{seed}.auc.{name}", value, path)
            by_target.setdefault(target, {}).setdefault(name, []).append(value)
    for target, levels in by_target.items():
        for name, values in levels.items():
            source = directory / f"fusion-{target}-seed*-predictions.csv"
            _put(numbers, f"h2.{target}.auc.{name}.min", min(values), source)
            _put(numbers, f"h2.{target}.auc.{name}.max", max(values), source)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _put_range(numbers: dict, key: str, values: list[float], source: Path) -> None:
    _put(numbers, f"{key}.min", min(values), source)
    _put(numbers, f"{key}.max", max(values), source)


def h2_checks(numbers: dict) -> None:
    """Post-hoc checks on H2, run after its verdict (exploratory).

    1. The FakeAVCeleb gain with the cloned-voice clips (real video, fake audio) left out. The visual
       path is trained never to flag those, so any gain on them comes by construction.
    2. The visual path on each LAV-DF fake type against the reals, on its own predictions. A type
       whose video is untouched should sit near 0.5 if the path reacts to face edits.
    """
    h2_dir, branch_dir = RESULTS / "p001-h2", RESULTS / "p001-branches"
    cloned = "RealVideo-FakeAudio"
    gains = []
    for path in sorted(h2_dir.glob("fusion-test-identity-strict-seed*-predictions.csv")):
        seed = path.stem.removesuffix("-predictions").rsplit("-seed", 1)[1]
        types_file = branch_dir / f"visual-seed{seed}-test-identity-strict-predictions.csv"
        types = {row["clip_id"]: row["manipulation_type"] for row in _rows(types_file)}
        rows = _rows(path)
        kept = [row for row in rows if types[row["clip_id"]] != cloned]
        for name, subset in (("diff_all", rows), ("diff_without_cloned_voice", kept)):
            labels = [int(row["label"]) for row in subset]
            diff = _auc(labels, [float(r["fusion"]) for r in subset]) - _auc(labels, [float(r["visual_logit"]) for r in subset])
            _put(numbers, f"h2.test-identity-strict.seed{seed}.{name}", round(diff, 3), path)
        gains.append(numbers[f"h2.test-identity-strict.seed{seed}.diff_without_cloned_voice"]["value"])
    if gains:
        _put_range(numbers, "h2.test-identity-strict.without_cloned_voice", gains, h2_dir / "fusion-test-identity-strict-seed*-predictions.csv")
    by_type: dict[str, list[float]] = {}
    for path in sorted(branch_dir.glob("visual-seed*-lavdf-test-predictions.csv")):
        seed = path.stem.removeprefix("visual-seed").split("-", 1)[0]
        rows = _rows(path)
        reals = [float(r["probability"]) for r in rows if r["manipulation_type"] == "RealVideo-RealAudio"]
        for kind in sorted({r["manipulation_type"] for r in rows} - {"RealVideo-RealAudio"}):
            fakes = [float(r["probability"]) for r in rows if r["manipulation_type"] == kind]
            value = round(_auc([0] * len(reals) + [1] * len(fakes), reals + fakes), 3)
            _put(numbers, f"visual.lavdf-test.seed{seed}.type.{kind}", value, path)
            by_type.setdefault(kind, []).append(value)
    source = branch_dir / "visual-seed*-lavdf-test-predictions.csv"
    for kind, values in by_type.items():
        _put_range(numbers, f"visual.lavdf-test.type.{kind}", values, source)
    if by_type:
        _put_range(numbers, "visual.lavdf-test.type", [v for values in by_type.values() for v in values], source)


def a12(numbers: dict) -> None:
    """A12 (exploratory): fusion fit on other corpora. AUC as the mean over seeds, and each paired
    difference with its identity-bootstrap interval: a12.{target}.{first}-minus-{second}.*"""
    path = RESULTS / "p001-a12" / "a12.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    for target, result in report["targets"].items():
        _put(numbers, f"a12.{target}.clips", result["clips"], path)
        for name, values in result["auc"].items():
            _put(numbers, f"a12.{target}.auc.{name}", round(sum(values) / len(values), 3), path)
        for name, diff in result["differences"].items():
            for field in ("estimate", "lower", "upper"):
                _put(numbers, f"a12.{target}.{name}.{field}", round(diff[field], 3), path)


def a11(numbers: dict) -> None:
    """A11 (exploratory): each upgraded path against its P-001 counterpart, paired by clip.
    a11.{new}-vs-{old}.{part}.auc_new / auc_old (means over seeds), .estimate/.lower/.upper, .seeds;
    a set with one class records .skipped instead."""
    path = RESULTS / "p001-a11" / "compare.json"
    if not path.exists():
        return
    for name, result in json.loads(path.read_text(encoding="utf-8"))["results"].items():
        key = f"a11.{name}"
        if "skipped" in result:
            _put(numbers, f"{key}.skipped", result["skipped"], path)
            continue
        for side in ("auc_new", "auc_old"):
            _put(numbers, f"{key}.{side}", round(sum(result[side]) / len(result[side]), 3), path)
        for field in ("estimate", "lower", "upper"):
            _put(numbers, f"{key}.{field}", round(result["difference"][field], 3), path)
        _put(numbers, f"{key}.seeds", len(result["seeds"]), path)


def a13(numbers: dict) -> None:
    """A13 (exploratory): frame statistics, real clips against video fakes. a13.{lavdf|fakeavceleb}.{stat}.*"""
    path = RESULTS / "p001-a13" / "a13.json"
    if not path.exists():
        return
    names = {"lavdf_unedited_video_fakes_vs_reals": "lavdf", "fakeavceleb_video_fakes_vs_reals": "fakeavceleb"}
    for block, result in json.loads(path.read_text(encoding="utf-8")).items():
        if block not in names:
            continue
        key = f"a13.{names[block]}"
        for stat, entry in result["statistics"].items():
            _put(numbers, f"{key}.{stat}.auc", round(entry["auc"], 3), path)
            _put(numbers, f"{key}.{stat}.low", round(entry["interval"][0], 3), path)
            _put(numbers, f"{key}.{stat}.high", round(entry["interval"][1], 3), path)
            if "spearman_with_visual" in entry:
                _put(numbers, f"{key}.{stat}.rho", round(entry["spearman_with_visual"], 2), path)
                _put(numbers, f"{key}.{stat}.candidate", _yes(entry["candidate"]), path)
        aucs = [round(entry["auc"], 3) for entry in result["statistics"].values()]
        _put(numbers, f"{key}.auc_min", min(aucs), path)
        _put(numbers, f"{key}.auc_max", max(aucs), path)
        _put(numbers, f"{key}.all_six", round(result["all_six_cross_fit_auc"], 3), path)
        if "visual_path_auc" in result:
            _put(numbers, f"{key}.visual", round(result["visual_path_auc"], 3), path)


def a14(numbers: dict) -> None:
    """A14 (exploratory): the audio path under audio perturbations. a14.{part}.{kind}.seed{S}.*"""
    directory = RESULTS / "p001-a14"
    if not directory.exists():
        return
    perturbed: dict[str, list[float]] = {}
    for path in sorted(directory.glob("audio-seed*-metrics.json")):
        seed, rest = path.stem.removeprefix("audio-seed").removesuffix("-metrics").split("-", 1)
        part, kind = rest.rsplit("-", 1)
        metrics_file(numbers, f"a14.{part}.{kind}.seed{seed}", path)
        if kind != "none":           # "none" is the unperturbed rerun
            perturbed.setdefault(part, []).append(numbers[f"a14.{part}.{kind}.seed{seed}.auc"]["value"])
    for part, values in perturbed.items():
        _put(numbers, f"a14.{part}.perturbed_min", min(values), directory / f"audio-seed*-{part}-*-metrics.json")
        _put(numbers, f"a14.{part}.perturbed_max", max(values), directory / f"audio-seed*-{part}-*-metrics.json")


def _load(folder: str, name: str) -> tuple[dict | None, Path]:
    path = RESULTS / folder / name
    return (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None), path


def a15(numbers: dict) -> None:
    """A15 (exploratory): H3 one direction. a15.{label}.{heads.min|heads.max|baseline|estimate|...}"""
    report, path = _load("p001-a15", "a15.json")
    if report is None:
        return
    for label, r in report["labels"].items():
        key = f"a15.{label}"
        _put(numbers, f"{key}.clips", r["clips"], path)
        _put_range(numbers, f"{key}.heads", [round(v, 3) for v in r["macro_f1_heads"].values()], path)
        _put(numbers, f"{key}.baseline", round(r["macro_f1_baseline"], 3), path)
        for name in ("estimate", "lower", "upper"):
            _put(numbers, f"{key}.{name}", round(r["difference"][name], 3), path)
        for stream, rows in r["per_stream"].items():
            for seed, values in rows.items():
                name = seed if seed == "baseline" else f"seed{seed}"
                for metric, value in values.items():
                    _put(numbers, f"{key}.{stream}.{name}.{metric}", round(value, 3), path)


def a16(numbers: dict) -> None:
    """A16 (exploratory): sync score with moved audio. a16.{shift004|...|swap}.{auc|low|high|rising.*}"""
    report, path = _load("p001-a16", "a16.json")
    if report is None:
        return
    _put(numbers, "a16.clips", report["clips"], path)
    for condition, r in report["conditions"].items():
        key = "a16." + ("swap" if condition == "swap" else "shift" + condition.split()[1].replace(".", ""))
        _put(numbers, f"{key}.auc", round(r["mean_auc"], 3), path)
        _put(numbers, f"{key}.low", round(r["lower"], 3), path)
        _put(numbers, f"{key}.high", round(r["upper"], 3), path)
        _put_range(numbers, f"{key}.rising", [round(v["share_rising"], 2) for v in r["per_seed"].values()], path)


def a17(numbers: dict) -> None:
    """A17 (exploratory): FakeAVCeleb speakers in VoxCeleb2 dev and test. a17.sources.*, a17.vox2.*"""
    report, path = _load("p001-a17", "a17.json")
    if report is None:
        return
    for group, count in report["fakeavceleb_sources"].items():
        _put(numbers, f"a17.sources.{group}", count, path)
    for group, count in report["vox2_speakers"].items():
        _put(numbers, f"a17.vox2.{group}", count, path)
    for group, entry in report["fakeavceleb_test_partition"].items():
        _put(numbers, f"a17.test_clips.{group}", entry["clips"], path)


def a19(numbers: dict) -> None:
    """A19 (exploratory): AVH-Align's method against the sync path. a19.{set}.{auc|minus_sync.*|reading}"""
    report, path = _load("p001-a19", "a19.json")
    if report is None:
        return
    for name, r in report["sets"].items():
        key = f"a19.{name}"
        _put(numbers, f"{key}.clips", r["clips"], path)
        _put(numbers, f"{key}.auc", round(r["mean_auc"], 3), path)
        _put(numbers, f"{key}.low", round(r["lower"], 3), path)
        _put(numbers, f"{key}.high", round(r["upper"], 3), path)
        _put_range(numbers, f"{key}.sync", [round(v, 3) for v in r["auc_sync"].values()], path)
        diff = r["minus_sync"]
        for part in ("estimate", "lower", "upper"):
            _put(numbers, f"{key}.minus_sync.{part}", round(diff[part], 3), path)
        reading = "better" if diff["lower"] > 0 else "worse" if diff["upper"] < 0 else "not different"
        _put(numbers, f"{key}.reading", reading, path)


PARTS_SHOWN = (1, 2, 5, 10, 20)     # a 60 s upload, the site's longest, has at most 20 parts


def website(numbers: dict) -> None:
    """The website's flag and unsure lines per number of parts, and its measured speed per part."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from deepfake_detection.webapp.calibration import Calibration

    path = RESULTS / "web-calibration.json"
    calibration = Calibration.from_json(path.read_text(encoding="utf-8"))
    for parts in PARTS_SHOWN:
        flag, unsure = calibration.limits(parts)
        _put(numbers, f"web.limit.parts{parts}.flag", round(flag, 3), path)
        _put(numbers, f"web.limit.parts{parts}.unsure", round(unsure, 3), path)
    for corpus, sides in calibration.meta.get("outcomes", {}).items():
        name = corpus.lower().replace("-", "")
        for side, counts in sides.items():
            clips = sum(counts.values())
            _put(numbers, f"web.outcomes.{name}.{side}.clips", clips, path)
            share = round(counts.get("signs", 0) / clips, 3)
            _put(numbers, f"web.outcomes.{name}.{side}.{'flagged' if side == 'real' else 'caught'}", share, path)
    timing = RESULTS / "web-timing.json"
    if not timing.exists():
        return
    runs = json.loads(timing.read_text(encoding="utf-8"))
    per_part = sum(c["seconds"] for c in runs["clips"]) / sum(c["parts"] for c in runs["clips"])
    _put(numbers, "web.seconds_per_part", round(per_part), timing)
    _put(numbers, "web.minutes_60s", round(max(PARTS_SHOWN) * per_part / 60, 1), timing)
    _put(numbers, "web.timing.threads", runs["threads"], timing)


def file_check(numbers: dict) -> None:
    """How many HiFi-AVDF files carry an AI label the website's file check reads, by generator."""
    path = RESULTS / "web-provenance.json"
    if not path.exists():
        return
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    for group, counts in groups.items():
        _put(numbers, f"web.provenance.{group}.labelled", counts["labelled"], path)
        _put(numbers, f"web.provenance.{group}.files", counts["files"], path)
    fakes = [c for g, c in groups.items() if g != "real"]
    labelled, files = sum(c["labelled"] for c in fakes), sum(c["files"] for c in fakes)
    _put(numbers, "web.provenance.fakes.labelled", labelled, path)
    _put(numbers, "web.provenance.fakes.files", files, path)
    _put(numbers, "web.provenance.fakes.share", round(labelled / files, 3), path)


def main() -> int:
    numbers: dict = {}
    audits(numbers)
    neutralization(numbers)
    sync(numbers)
    sync_training(numbers)
    h4(numbers)
    branches(numbers)
    h2(numbers)
    h2_levels(numbers)
    h2_checks(numbers)
    a12(numbers)
    a11(numbers)
    a13(numbers)
    a14(numbers)
    a15(numbers)
    a16(numbers)
    a17(numbers)
    a19(numbers)
    website(numbers)
    file_check(numbers)
    out = RESULTS / "numbers.json"
    out.write_text(json.dumps(dict(sorted(numbers.items())), indent=1) + "\n", encoding="utf-8")
    print(f"{len(numbers)} numbers -> {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
