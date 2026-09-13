"""Regenerate the result registry from the artifacts it points at.

`docs/research/result-traceability.md` requires every reported number to resolve
to an analysis command, content hashes, and MLflow runs. Written by hand, that
table goes stale the first time a run is repeated, and a stale hash is worse
than none: it still looks like evidence.

So the rows are declared here, next to the command that produces each artifact,
and the hashes are computed from the files at the moment of writing. An artifact
that is missing produces a `pending` row naming what to run, rather than being
quietly dropped.

Status follows the acceptance rules in that document, and the seed rule is the
one that bites: the frozen protocol asks for three training seeds and the
Design B streams were trained on one, so their rows are `provisional` however
well the hashes resolve.

    uv run python scripts/update_result_registry.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = Path("docs/research/result-traceability.md")
BEGIN = "<!-- BEGIN GENERATED REGISTRY -->"
END = "<!-- END GENERATED REGISTRY -->"


@dataclass(frozen=True, slots=True)
class Result:
    """One reported number and everything needed to get back to it."""

    result_id: str
    paper_location: str
    command: str
    report: Path
    # The per-clip scores behind the report. Design A writes a predictions CSV;
    # the Design B streams write a feature store instead, which carries the same
    # per-clip logits alongside the embeddings fusion reads.
    predictions: Path | None
    mlflow: tuple[tuple[str, str], ...] = ()
    decision: str = ""
    seeds: int = 1
    notes: str = ""
    # Set where a result is quoted from a run whose artifact was never written
    # to `runs/`. It stays in the table as `pending` with the command that would
    # produce it, because a number in the paper with no file behind it is the
    # thing this registry exists to make visible.
    missing_reason: str = ""
    extra: tuple[str, ...] = field(default_factory=tuple)


PROGRAM = Path("runs/program-20260906")
DESIGN_B = Path("runs/design-b-20260910")

RESULTS: tuple[Result, ...] = (
    Result(
        result_id="A-visual-dfdc",
        paper_location="Results, cross-corpus baseline",
        command="ddf evaluate branch --branch visual --dataset DFDC",
        report=PROGRAM / "evaluation" / "visual-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-dfdc-predictions.csv",
        mlflow=(
            ("program-20260906", "final-visual-seed17"),
            ("program-20260906", "visual-dfdc-evaluation"),
        ),
        decision="Design A visual baseline, the number Design B is measured against",
        seeds=1,
    ),
    Result(
        result_id="A-fusion-dfdc",
        paper_location="Results, late fusion cross-corpus",
        command="ddf evaluate fusion --dataset DFDC",
        report=PROGRAM / "evaluation" / "fusion-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "fusion-dfdc-predictions.csv",
        mlflow=(("program-20260906", "fusion-dfdc-evaluation"),),
        decision="Late fusion does not beat its own visual branch cross-corpus",
        seeds=1,
    ),
    Result(
        result_id="A-fusion-in-domain",
        paper_location="Results, late fusion in-domain",
        command="ddf evaluate fusion --dataset FakeAVCeleb --partition test",
        report=PROGRAM / "evaluation" / "fusion-test-metrics.json",
        predictions=PROGRAM / "evaluation" / "fusion-test-predictions.csv",
        mlflow=(("program-20260906", "fusion-test-evaluation"),),
        decision="In-domain ceiling for the Design A pipeline",
        seeds=1,
    ),
    Result(
        result_id="A-visual-mnw",
        paper_location="Results, how generalization fails",
        command="ddf evaluate branch --branch visual --dataset MNW",
        report=PROGRAM / "evaluation" / "visual-mnw-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-mnw-predictions.csv",
        mlflow=(("program-20260906", "visual-mnw-evaluation"),),
        decision=(
            "Detection is 25 of 85 and not uniform across generators: 0 of 10 on "
            "vasa_1 against 7 of 8 in the wild"
        ),
        seeds=1,
        notes="Fake-only, so no ranking metric exists. Detection rate only.",
    ),
    Result(
        result_id="B-stream-scores",
        paper_location="Results, per-stream ROC-AUC with intervals",
        command="python scripts/score_streams.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "stream-scores.json",
        predictions=DESIGN_B / "features" / "dfdc.parquet",
        mlflow=(
            ("design-b-frozen-bn", "visual-efficientnet"),
            ("design-b-frozen-bn", "visual-dinov3"),
            ("design-b-frozen-bn", "stream-lipsync"),
            ("design-b-frozen-bn", "stream-emotion"),
        ),
        decision="Every stream except emotion reads near chance cross-corpus",
        seeds=1,
    ),
    Result(
        result_id="B-training-histories",
        paper_location="Results, attention that never moved; lip-sync collapse",
        command="pwsh scripts/train_design_b.ps1",
        report=DESIGN_B / "checkpoints" / "stream-emotion-history.json",
        predictions=DESIGN_B / "checkpoints" / "stream-lipsync-lowlr-history.json",
        mlflow=(
            ("design-b-frozen-bn", "stream-emotion"),
            ("design-b-frozen-bn", "stream-lipsync"),
        ),
        decision=(
            "Both cross-modal streams sit on chance diagonal mass for every "
            "epoch of every run"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-deep-ablation",
        paper_location="Results, does fusing streams beat the best single stream",
        command="python scripts/run_deep_ablation.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "deep-ablation.json",
        predictions=DESIGN_B / "features" / "holdout.parquet",
        mlflow=(),
        decision=(
            "A subset beats the full fusion cross-corpus, so more streams is not "
            "the lever"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-stream-correlation",
        paper_location="Results, redundancy between streams",
        command="python scripts/stream_correlation.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "stream-correlation.json",
        predictions=DESIGN_B / "features" / "in-domain.parquet",
        mlflow=(),
        decision="Redundancy is measurable and was not where it was predicted",
        seeds=1,
    ),
    Result(
        result_id="B-batchnorm-arms",
        paper_location="Results, freezing BatchNorm trades transfer for accuracy",
        command=(
            "python scripts/score_batchnorm_arms.py"
        ),
        report=DESIGN_B / "evaluation" / "batchnorm-arms.json",
        predictions=None,
        mlflow=(
            ("design-b-frozen-bn", "visual-efficientnet"),
            ("design-b-live-bn", "visual-efficientnet"),
        ),
        decision=(
            "Live BatchNorm buys 0.1276 on DFDC and costs 0.5430 in-domain. The "
            "frozen arm ships; the trade is the result"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-batchnorm-trade",
        paper_location="Results, freezing BatchNorm trades transfer for accuracy",
        command=(
            "python scripts/batchnorm_sweep.py --run-dir runs/design-b-20260910"
        ),
        report=DESIGN_B / "evaluation" / "batchnorm-sweep.json",
        predictions=None,
        mlflow=(),
        decision=(
            "Holds the epoch budget fixed, which the full-scale comparison cannot"
        ),
        seeds=1,
        missing_reason=(
            "Measured in a scratch script whose output was never written under "
            "runs/. Re-run before citing."
        ),
    ),
    Result(
        result_id="B-fusion-ablations",
        paper_location="Results, late against deep and abstention against fallback",
        command=(
            "python scripts/run_fusion_ablations.py --run-dir runs/design-b-20260910"
        ),
        report=DESIGN_B / "evaluation" / "fusion-ablations.json",
        predictions=DESIGN_B / "features" / "dfdc.parquet",
        mlflow=(),
        decision=(
            "The embedding carries nothing the logit does not, and abstention "
            "buys no accuracy"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-gate-thresholds",
        paper_location="Method, per-media-kind decision thresholds",
        command="python scripts/fit_gate_fusion.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "checkpoints" / "gate-fusion.json",
        predictions=DESIGN_B / "features" / "holdout.parquet",
        mlflow=(),
        decision="One threshold cannot serve video, image and audio",
        seeds=1,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_cell(path: Path | None) -> str:
    if path is None:
        return "not applicable"
    if not path.is_file():
        return "missing"
    return f"`{_sha256(path)[:16]}`"


def _mlflow_ids(entries, tracking_uri: str) -> dict[tuple[str, str], str]:
    """Run IDs for every (experiment, run name) the registry cites."""
    try:
        import mlflow
    except ImportError:
        return {}
    mlflow.set_tracking_uri(tracking_uri)
    found: dict[tuple[str, str], str] = {}
    for experiment in sorted({name for name, _ in entries}):
        try:
            runs = mlflow.search_runs(
                experiment_names=[experiment], output_format="list", max_results=5000
            )
        except Exception as error:
            # An experiment that is not in the store is the ordinary case on a
            # fresh checkout, and it must not stop the rest of the table.
            print(f"  cannot read experiment {experiment}: {error}")
            continue
        for run in runs:
            key = (experiment, run.info.run_name or "")
            # Newest wins: a retrained stream logs a second run under the same
            # name, and the registry should point at the current one.
            found.setdefault(key, run.info.run_id)
    return found


def _status(result: Result, resolved: bool) -> str:
    if not resolved:
        return "pending"
    if result.seeds < 3:
        # The frozen protocol asks for three training seeds. One seed is a real
        # result and not an accepted one.
        return "provisional (1 seed)"
    return "accepted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    arguments = parser.parse_args(argv)

    cited = [entry for result in RESULTS for entry in result.mlflow]
    run_ids = _mlflow_ids(cited, arguments.tracking_uri)

    lines = [
        "| Result ID | Paper location | Analysis command | Report SHA-256 | "
        "Prediction SHA-256 | MLflow run IDs | Decision | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    pending = 0
    for result in sorted(RESULTS, key=lambda item: item.result_id):
        report_cell = _hash_cell(result.report)
        prediction_cell = _hash_cell(result.predictions)
        resolved = report_cell not in {"missing"} and prediction_cell != "missing"
        if not resolved:
            pending += 1
        ids = [
            f"`{run_ids[key][:12]}`" if key in run_ids else "not logged"
            for key in result.mlflow
        ]
        decision = result.decision
        if result.missing_reason and not resolved:
            decision = f"{decision}. {result.missing_reason}"
        lines.append(
            f"| `{result.result_id}` | {result.paper_location} | "
            f"`{result.command}` | {report_cell} | {prediction_cell} | "
            f"{', '.join(ids) if ids else 'none'} | {decision} | "
            f"{_status(result, resolved)} |"
        )

    text = arguments.registry.read_text(encoding="utf-8")
    block = BEGIN + "\n" + "\n".join(lines) + "\n" + END
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + block + tail
    else:
        text = text.replace(
            "## Acceptance rules", block + "\n\n## Acceptance rules", 1
        )
    arguments.registry.write_text(text, encoding="utf-8")

    print(f"{len(RESULTS)} rows, {pending} pending -> {arguments.registry}")
    for result in RESULTS:
        for key in result.mlflow:
            if key not in run_ids:
                print(f"  no MLflow run for {key[0]} / {key[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
