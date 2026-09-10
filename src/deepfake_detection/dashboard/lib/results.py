"""Read a training run's recorded results off disk, for the pages that show them.

Every number these pages display comes from a file written by `ddf` or by
`scripts/run_ablation.py`, never from a model run in the dashboard. The
dashboard has no training path and should not acquire one: a figure that
appears here has to be traceable to a run directory and a checkpoint hash.

Nothing raises for a missing file. A run that has not produced an artifact yet
is a normal state, and the page says which file it wanted rather than showing a
stack trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from deepfake_detection.dashboard.paths import RUNS_DIR

# The Design A run that produced the trained fusion model. Named rather than
# discovered: several run directories carry a `fusion-ablation.json`, and the
# page should show the one the recorded results belong to instead of whichever
# sorted last.
PROGRAM_RUN = "program-20260906"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class FusionResults:
    """The trained fusion model, its metrics, and the stream ablation."""

    run_dir: Path
    metadata: dict | None = None
    ablation: dict | None = None
    partitions: dict[str, dict] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.metadata is not None

    @property
    def missing(self) -> list[str]:
        """Which expected files are absent, for a page to report."""
        wanted = {
            "checkpoints/fusion-logistic.json": self.metadata,
            "evaluation/fusion-ablation.json": self.ablation,
        }
        return sorted(name for name, value in wanted.items() if value is None)


def load_fusion(run: str = PROGRAM_RUN, root: Path | None = None) -> FusionResults:
    run_dir = (root or RUNS_DIR) / run
    partitions = {}
    for name, filename in (
        ("in-domain", "fusion-test-metrics.json"),
        ("dfdc", "fusion-dfdc-metrics.json"),
    ):
        found = _read(run_dir / "evaluation" / filename)
        if found is not None:
            partitions[name] = found
    return FusionResults(
        run_dir=run_dir,
        metadata=_read(run_dir / "checkpoints" / "fusion-logistic.json"),
        ablation=_read(run_dir / "evaluation" / "fusion-ablation.json"),
        partitions=partitions,
    )


def ablation_rows(ablation: dict) -> list[dict]:
    """Ablation combinations, smallest subset first, as display rows."""
    rows = []
    for row in sorted(
        ablation.get("combinations", []),
        key=lambda item: (item["size"], item["branches"]),
    ):
        rows.append(
            {
                "streams": " + ".join(row["branches"]),
                "count": row["size"],
                **{
                    name: row.get(name)
                    for name in ("in-domain", "dfdc")
                    if name in row
                },
            }
        )
    return rows


def beats_every_single_stream(ablation: dict, partition: str) -> bool | None:
    """Does the full combination beat every stream on its own?

    Objective 3's success metric, stated as a question rather than assumed. The
    answer differs by partition, and that disagreement is the result: in-domain
    the combination wins, cross-corpus it does not.
    """
    combinations = ablation.get("combinations", [])
    scored = [row for row in combinations if row.get(partition) is not None]
    singles = [row for row in scored if row["size"] == 1]
    largest = max(scored, key=lambda row: row["size"], default=None)
    if largest is None or not singles or largest["size"] == 1:
        return None
    return all(largest[partition] > single[partition] for single in singles)


@dataclass(frozen=True, slots=True)
class BranchBreakdown:
    """One evaluated branch, broken down by manipulation method."""

    branch: str
    dataset: str
    rows: int
    per_method: dict
    per_manipulation_type: dict
    checkpoint_sha256: str


def load_branch_breakdowns(
    run: str = PROGRAM_RUN, root: Path | None = None
) -> list[BranchBreakdown]:
    """Every per-method evaluation the run recorded, newest naming first.

    Read from whatever `*-metrics.json` files exist rather than a fixed list, so
    a run that evaluated more partitions shows more here without a code change.
    """
    directory = (root or RUNS_DIR) / run / "evaluation"
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*-metrics.json")):
        payload = _read(path)
        if payload is None or "per_method" not in payload:
            continue
        # Fusion metrics carry per_method too but no branch field; they are the
        # Fusion page's business, not the per-branch breakdown.
        if "branch" not in payload:
            continue
        found.append(
            BranchBreakdown(
                branch=str(payload["branch"]),
                dataset=str(payload.get("dataset", "unknown")),
                rows=int(payload.get("rows", 0)),
                per_method=payload.get("per_method", {}),
                per_manipulation_type=payload.get("per_manipulation_type", {}),
                checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            )
        )
    return found


def subgroup_rows(metrics: dict, attribute: str) -> list[dict]:
    """Per-subgroup metrics with their coverage, for a fairness table.

    Coverage travels with the metric on purpose. A subgroup the model abstained
    on half the time has a score computed from the half it answered, and a bare
    accuracy would hide that.
    """
    rows = []
    for name, payload in sorted((metrics.get(attribute) or {}).items()):
        values = payload.get("metrics", {})
        rows.append(
            {
                attribute: name,
                "coverage": payload.get("coverage"),
                "abstained": payload.get("abstained"),
                "roc_auc": values.get("roc_auc"),
                "balanced_accuracy": values.get("balanced_accuracy"),
                "eer": values.get("eer"),
            }
        )
    return rows
