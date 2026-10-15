"""How much does the branch seed move the fused number?

F21 asked this of the visual branch alone and found 0.0733 on DFDC, wider than
most effects this project reports. This asks it of the fused system, which is
the number the project actually headlines.

Two tables. The first varies only the three final branch checkpoints and
scores every seed through the head fitted on the seed-17 out-of-fold store, so
it measures seed variance in the branches through a fixed head.

The second is the full three-seed replication the protocol asks for.
`run_crossfit_seeds.ps1` cross-fits the folds per seed, and each seed gets its
own head, `fusion-logistic-seed<N>.joblib`, fitted on its own out-of-fold store.
Seed 17's own head is the original `fusion-logistic.joblib`, so its row is the
same in both tables.

The head is refitted on nothing here, so no seed's test partition ever
influences the weights that score it.

    uv run python scripts/run_fusion_seed_spread.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

BRANCHES = ("visual", "audio", "sync")


@dataclass(frozen=True, slots=True)
class Scored:
    label: int
    probability: float
    source_identity: str


def _assemble(store_path: Path):
    from deepfake_detection.fusion.store import FeatureStore

    logits: dict[str, dict[str, float]] = collections.defaultdict(dict)
    meta: dict[str, dict] = {}
    for row in FeatureStore(store_path).read():
        if not row.available:
            continue
        logits[row.clip_id][row.branch] = row.logit
        meta[row.clip_id] = {
            "label": row.label,
            "source_identity": row.source_identity,
            "quality": (
                row.face_coverage,
                row.audio_clipped,
                row.av_duration_delta_sec,
            ),
        }
    complete = sorted(k for k, v in logits.items() if len(v) == len(BRANCHES))
    return complete, logits, meta


def _samples(ids, logits, meta):
    from deepfake_detection.fusion.late import FusionSample

    return [
        FusionSample(
            branch_logits={b: logits[i][b] for b in BRANCHES},
            face_coverage=meta[i]["quality"][0],
            audio_clipped=meta[i]["quality"][1],
            av_duration_delta_sec=meta[i]["quality"][2],
        )
        for i in ids
    ]


def main(argv: list[str] | None = None) -> int:
    import joblib
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    arguments = parser.parse_args(argv)
    run = arguments.run_dir

    def load_head(path: Path):
        artifact = joblib.load(path)
        return getattr(artifact, "model", artifact)

    fixed = load_head(run / "checkpoints" / "fusion-logistic.joblib")

    def own_head(seed: int):
        name = "fusion-logistic" if seed == 17 else f"fusion-logistic-seed{seed}"
        path = run / "checkpoints" / f"{name}.joblib"
        return load_head(path) if path.is_file() else None

    # Seed 17 wrote the original stores without a suffix.
    def store_for(partition: str, seed: int) -> Path:
        stem = partition if seed == 17 else f"{partition}-seed{seed}"
        return run / "features" / f"{stem}.parquet"

    def score(model, path: Path) -> dict:
        ids, logits, meta = _assemble(path)
        probabilities = model.predict_proba(_samples(ids, logits, meta))
        items = [
            Scored(meta[i]["label"], float(p), meta[i]["source_identity"])
            for i, p in zip(ids, probabilities, strict=True)
        ]

        def statistic(sample):
            labels = [r.label for r in sample]
            if len(set(labels)) < 2:
                raise ValueError("Bootstrap sample does not contain both classes")
            return float(roc_auc_score(labels, [r.probability for r in sample]))

        interval = cluster_bootstrap_interval(
            items,
            statistic,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        return {
            "clips": len(ids),
            "roc_auc": interval.estimate,
            "roc_auc_lower": interval.lower,
            "roc_auc_upper": interval.upper,
        }

    report: dict[str, dict] = {}
    own_report: dict[str, dict] = {}
    for partition, label in (("test", "in-domain"), ("dfdc", "DFDC")):
        rows: dict[str, dict] = {}
        own_rows: dict[str, dict] = {}
        for seed in arguments.seeds:
            path = store_for(partition, seed)
            if not path.is_file():
                print(f"{label} seed {seed}: no store at {path}, skipping")
                continue
            rows[str(seed)] = score(fixed, path)
            head = own_head(seed)
            if head is None:
                print(f"{label} seed {seed}: no head of its own yet")
            else:
                own_rows[str(seed)] = score(head, path)
        report[label] = rows
        own_report[label] = own_rows

    def show(title: str, table: dict[str, dict]) -> dict[str, dict]:
        print(f"\n{title}")
        print(
            f"{'partition':10}  {'seed':>5}  {'clips':>6}  {'ROC-AUC':>8}  "
            f"{'interval':>18}"
        )
        print("-" * 56)
        for label, rows in table.items():
            for seed, entry in rows.items():
                print(
                    f"{label:10}  {seed:>5}  {entry['clips']:6,}  "
                    f"{entry['roc_auc']:8.4f}  "
                    f"[{entry['roc_auc_lower']:.4f}, {entry['roc_auc_upper']:.4f}]"
                )
        spread: dict[str, dict] = {}
        for label, rows in table.items():
            values = [entry["roc_auc"] for entry in rows.values()]
            if len(values) < 2:
                continue
            spread[label] = {
                "seeds": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "spread": max(values) - min(values),
                "mean": statistics.fmean(values),
            }
            item = spread[label]
            print(
                f"{label}: {item['minimum']:.4f} to {item['maximum']:.4f}, "
                f"spread {item['spread']:.4f}, mean {item['mean']:.4f}"
            )
        return spread

    spread = show("branches vary, seed-17 head fixed", report)
    own_spread = show("branches and head vary, each seed its own head", own_report)

    payload = {
        "scope": (
            "Only the final branch checkpoints vary. The fusion head is the one "
            "fitted on the seed-17 out-of-fold store and is reused unchanged. "
            "This is seed variance in the branches through a fixed head; the "
            "own-head tables are the full three-seed replication."
        ),
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_unit": "source_identity",
        "partitions": report,
        "spread": spread,
        "own_head_scope": (
            "Each seed scored through the head fitted on its own out-of-fold "
            "store, so the branches, the fold models and the head all vary. "
            "This is the three-seed replication the protocol asks for."
        ),
        "own_head_partitions": own_report,
        "own_head_spread": own_spread,
    }
    out = arguments.output or run / "evaluation" / "fusion-seed-spread.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
