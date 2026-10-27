"""One table over the visual recipes: ranking, and what a fixed threshold does.

Reads the per-clip predictions `run_visual_recipes.py` wrote. ROC-AUC carries an
identity-clustered interval. The operating point is chosen once per arm, on
FakeAVCeleb validation genuine clips, as the threshold that accuses at most
`--target-fpr` of them; it is then applied unchanged to every other partition.
That is the threshold rule F16 asks for: set on genuine clips the model did not
train on, never on a test corpus.

    uv run python scripts/summarize_visual_recipes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PARTITIONS = ("in-domain-test", "dfdc", "celebdf", "ffpp-test")


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            r for r in csv.DictReader(handle) if r["probability"] not in ("", "nan")
        ]
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["probability"]) for r in rows])
    sources = np.array([r["source"] for r in rows])
    return labels, scores, sources


def _auc(labels, scores, sources, samples: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.evaluation.metrics import EvaluationItem

    items = [
        EvaluationItem(label=int(y), probability=float(p), source_identity=s, method="")
        for y, p, s in zip(labels, scores, sources, strict=True)
    ]

    def statistic(chosen):
        return float(
            roc_auc_score([i.label for i in chosen], [i.probability for i in chosen])
        )

    interval = cluster_bootstrap_interval(items, statistic, samples=samples, seed=seed)
    return {
        "roc_auc": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retrain-20261022"))
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    evaluation = arguments.run_dir / "evaluation"

    arms = sorted(
        {
            p.name.removesuffix("-validation-predictions.csv")
            for p in evaluation.glob("*-validation-predictions.csv")
        }
    )
    summary = {"target_fpr": arguments.target_fpr, "arms": {}}
    for arm in arms:
        val_labels, val_scores, _ = _read(
            evaluation / f"{arm}-validation-predictions.csv"
        )
        genuine = np.sort(val_scores[val_labels == 0])
        # The smallest threshold that accuses at most target_fpr of genuine
        # validation clips.
        index = int(np.ceil((1 - arguments.target_fpr) * len(genuine))) - 1
        threshold = float(genuine[min(max(index, 0), len(genuine) - 1)])
        row = {"threshold": threshold, "partitions": {}}
        for partition in PARTITIONS:
            path = evaluation / f"{arm}-{partition}-predictions.csv"
            if not path.is_file():
                continue
            labels, scores, sources = _read(path)
            entry = _auc(labels, scores, sources, arguments.samples, arguments.seed)
            called = scores > threshold
            entry["false_alarm_rate"] = float(called[labels == 0].mean())
            entry["detection_rate"] = float(called[labels == 1].mean())
            entry["clips"] = int(len(labels))
            row["partitions"][partition] = entry
        summary["arms"][arm] = row

    output = evaluation / "recipes-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    header = f"{'arm':22} {'thr':>6}"
    for partition in PARTITIONS:
        header += f" | {partition:^30}"
    print(header)
    for arm, row in summary["arms"].items():
        line = f"{arm:22} {row['threshold']:6.3f}"
        for partition in PARTITIONS:
            e = row["partitions"].get(partition)
            line += (
                f" | {e['roc_auc']:.4f} [{e['lower']:.3f},{e['upper']:.3f}] "
                f"FA {e['false_alarm_rate']:.2f} D {e['detection_rate']:.2f}"
                if e
                else f" | {'':30}"
            )
        print(line)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
