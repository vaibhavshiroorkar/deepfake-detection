"""What the locked system gets wrong, and whether it is wrong confidently.

Phase 5 asks for error analysis without retuning, so this reads the prediction
CSVs the frozen evaluations already wrote and changes nothing. No model is
loaded and no threshold is chosen here: the 0.5 the evaluations used is the
threshold, including where it is clearly wrong for the corpus.

Four questions, in the order they matter for a deployment.

**Is it wrong confidently?** A detector that hedges on the clips it gets wrong
can be salvaged by abstention. One that is certain and wrong cannot, because
there is no score left to threshold on. This reports the mean probability of the
errors beside the mean of the correct answers, and counts errors past 0.9.

**Do the errors concentrate on a few identities?** Every interval in this
project is clustered on the speaker, which assumes errors travel with speakers.
If a handful of identities carry most of the errors, the corpus is measuring
fewer independent things than its clip count suggests.

**Do the branches fail together?** Fusion can only repair a branch's mistake on
clips where another branch is right. Where the visual branch and the fused score
are wrong on the same clip, fusion had nothing to add, and that bounds what any
amount of head tuning could ever buy.

**Which manipulation survives?** Errors by method, so a method that is never
caught is visible rather than averaged away.

    uv run python scripts/run_error_analysis.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

# Each row is a recorded evaluation: its predictions file and what it scored.
# MNW is fake-only, so its "false alarm" column is structurally empty and its
# miss column is the whole story.
REPORTS = (
    ("fusion-in-domain", "fusion-test-predictions.csv", "FakeAVCeleb test"),
    ("fusion-dfdc", "fusion-dfdc-predictions.csv", "DFDC"),
    ("visual-in-domain", "visual-in-domain-test-predictions.csv", "FakeAVCeleb test"),
    ("visual-dfdc", "visual-dfdc-predictions.csv", "DFDC"),
    ("visual-celebdf", "visual-celebdf-predictions.csv", "Celeb-DF-v2"),
    ("visual-mnw", "visual-mnw-predictions.csv", "MNW"),
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _analyse(rows: list[dict], *, threshold: float) -> dict:
    scored = [
        {
            "clip_id": row["clip_id"],
            "label": int(row["label"]),
            "probability": float(row["probability"]),
            "source": row.get("source", "unknown"),
            "method": row.get("method", "unknown"),
        }
        for row in rows
        if row.get("probability") not in (None, "")
    ]
    for row in scored:
        row["predicted"] = int(row["probability"] >= threshold)
        row["correct"] = row["predicted"] == row["label"]

    errors = [row for row in scored if not row["correct"]]
    correct = [row for row in scored if row["correct"]]
    genuine = [row for row in scored if row["label"] == 0]
    fake = [row for row in scored if row["label"] == 1]

    def confidence(row: dict) -> float:
        """How sure the model was of the answer it gave, regardless of side."""
        return (
            row["probability"] if row["predicted"] == 1 else 1.0 - row["probability"]
        )

    by_source: collections.Counter = collections.Counter(
        row["source"] for row in errors
    )
    by_method: dict[str, dict] = {}
    for method in sorted({row["method"] for row in scored}):
        subset = [row for row in scored if row["method"] == method]
        wrong = [row for row in subset if not row["correct"]]
        by_method[method] = {
            "clips": len(subset),
            "errors": len(wrong),
            "error_rate": len(wrong) / len(subset) if subset else None,
        }

    sources = {row["source"] for row in scored}
    top = by_source.most_common(5)
    return {
        "clips": len(scored),
        "threshold": threshold,
        "genuine": len(genuine),
        "fake": len(fake),
        "errors": len(errors),
        "false_alarms": sum(1 for row in genuine if row["predicted"] == 1),
        "misses": sum(1 for row in fake if row["predicted"] == 0),
        "mean_confidence_when_right": (
            statistics.fmean(confidence(row) for row in correct) if correct else None
        ),
        "mean_confidence_when_wrong": (
            statistics.fmean(confidence(row) for row in errors) if errors else None
        ),
        "confident_errors_over_0_9": sum(
            1 for row in errors if confidence(row) > 0.9
        ),
        "identities": len(sources),
        "identities_with_an_error": len({row["source"] for row in errors}),
        "worst_identities": [
            {"source": source, "errors": count} for source, count in top
        ],
        "errors_in_worst_five_identities": sum(count for _, count in top),
        "by_method": by_method,
    }


def _paired_failure(fused: list[dict], visual: list[dict], *, threshold: float) -> dict:
    """Where fusion and its own visual branch are both wrong on the same clip.

    Fusion can only repair a mistake the other branches disagree with, so the
    clips both get wrong are the ceiling on what any head could recover.

    Two traps, both of which this project has already been caught by once.

    The two files do not hold the same clips: fusion demands every branch, so it
    scores fewer. Comparing the raw error sets would count a clip fusion never
    saw as one it repaired. Only the intersection is compared.

    The two files do not always agree on a clip's label either. Branch reports
    use the branch's own cue, where a RealVideo-FakeAudio clip is genuine for
    the visual branch; fusion uses the clip-level label, where it is fake. Those
    clips are counted and reported separately rather than folded in, because a
    repair that is really a relabelling is not a repair.
    """
    fused_by_id = {row["clip_id"]: row for row in fused}
    visual_by_id = {row["clip_id"]: row for row in visual}
    shared_ids = sorted(set(fused_by_id) & set(visual_by_id))

    disagreed = [
        clip_id
        for clip_id in shared_ids
        if int(fused_by_id[clip_id]["label"]) != int(visual_by_id[clip_id]["label"])
    ]
    comparable = [clip_id for clip_id in shared_ids if clip_id not in set(disagreed)]

    def wrong(by_id: dict, ids: list[str]) -> set[str]:
        return {
            clip_id
            for clip_id in ids
            if (float(by_id[clip_id]["probability"]) >= threshold)
            != (int(by_id[clip_id]["label"]) == 1)
        }

    fused_wrong = wrong(fused_by_id, comparable)
    visual_wrong = wrong(visual_by_id, comparable)
    return {
        "fusion_rows": len(fused),
        "visual_rows": len(visual),
        "shared_clips": len(shared_ids),
        "label_convention_disagreements": len(disagreed),
        "compared_clips": len(comparable),
        "fusion_errors": len(fused_wrong),
        "visual_errors": len(visual_wrong),
        "both_wrong": len(fused_wrong & visual_wrong),
        "fusion_repaired": len(visual_wrong - fused_wrong),
        "fusion_broke": len(fused_wrong - visual_wrong),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    evaluation = arguments.run_dir / "evaluation"

    report: dict[str, dict] = {}
    loaded: dict[str, list[dict]] = {}
    for name, filename, dataset in REPORTS:
        path = evaluation / filename
        if not path.is_file():
            print(f"{name}: no predictions at {path}, skipping")
            continue
        rows = _rows(path)
        loaded[name] = rows
        entry = _analyse(rows, threshold=arguments.threshold)
        entry["dataset"] = dataset
        entry["predictions"] = str(path).replace("\\", "/")
        report[name] = entry

    width = max(len(name) for name in report) if report else 10
    print(
        f"{'evaluation':{width}}  {'clips':>6}  {'errors':>7}  {'false alarm':>11}  "
        f"{'miss':>6}  {'sure/right':>10}  {'sure/wrong':>10}  {'>0.9 wrong':>10}"
    )
    print("-" * (width + 72))
    for name, entry in report.items():
        right = entry["mean_confidence_when_right"]
        wrong = entry["mean_confidence_when_wrong"]
        print(
            f"{name:{width}}  {entry['clips']:6,}  {entry['errors']:7,}  "
            f"{entry['false_alarms']:>4}/{entry['genuine']:<6}  "
            f"{entry['misses']:6,}  "
            f"{(f'{right:.4f}' if right is not None else 'n/a'):>10}  "
            f"{(f'{wrong:.4f}' if wrong is not None else 'n/a'):>10}  "
            f"{entry['confident_errors_over_0_9']:10,}"
        )
    print()

    for name, entry in report.items():
        if not entry["errors"]:
            continue
        share = (
            entry["errors_in_worst_five_identities"] / entry["errors"]
            if entry["errors"]
            else 0.0
        )
        print(
            f"{name}: {entry['errors']} errors over "
            f"{entry['identities_with_an_error']} of {entry['identities']} "
            f"identities; the worst five carry {share:.1%}"
        )
    print()

    paired: dict[str, dict] = {}
    for fused_name, visual_name in (
        ("fusion-in-domain", "visual-in-domain"),
        ("fusion-dfdc", "visual-dfdc"),
    ):
        if fused_name in loaded and visual_name in loaded:
            paired[fused_name] = _paired_failure(
                loaded[fused_name],
                loaded[visual_name],
                threshold=arguments.threshold,
            )
            entry = paired[fused_name]
            print(
                f"{fused_name} against {visual_name}: "
                f"{entry['compared_clips']:,} comparable clips "
                f"({entry['label_convention_disagreements']:,} excluded for "
                f"differing label conventions); both wrong on "
                f"{entry['both_wrong']:,}, fusion repaired "
                f"{entry['fusion_repaired']:,}, fusion broke "
                f"{entry['fusion_broke']:,}"
            )
    print()

    for name, entry in report.items():
        methods = {
            method: value
            for method, value in entry["by_method"].items()
            if value["errors"]
        }
        if methods:
            summary = ", ".join(
                f"{method} {value['errors']}/{value['clips']}"
                for method, value in sorted(methods.items())
            )
            print(f"{name} errors by method: {summary}")
    print()

    payload = {
        "threshold": arguments.threshold,
        "note": (
            "Read from the frozen evaluation predictions. No model was loaded "
            "and no threshold was selected here."
        ),
        "evaluations": report,
        "fusion_against_its_visual_branch": paired,
    }
    out = arguments.output or evaluation / "error-analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
