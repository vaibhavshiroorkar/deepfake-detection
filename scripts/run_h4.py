"""P-001 H4: does a threshold set from real target clips hold the false-alarm rate?

For one target and one seed:

1. **Source threshold.** The threshold chosen on FakeAVCeleb validation, applied
   unchanged. This is what a deployed detector would do.
2. **Real-only threshold.** A random set of `n_cal` real target clips (seed 20261113,
   fixed in P-001) is set aside as calibration. A split-conformal threshold at
   `alpha`, and a Neyman-Pearson umbrella threshold at (alpha, delta), come from
   their scores alone. No fake label is read.
3. All three are scored on the remaining target clips: false-positive rate, true-positive rate.

Success (preregistered): realized FPR at most alpha + 0.02 with the conformal threshold.

    uv run python scripts/run_h4.py --predictions runs/p001/sync-targets/sync-hifiavdf-seed17-predictions.csv \
        --source-threshold runs/p001/sync/sync-seed17-threshold.json --output runs/p001/h4/sync-hifiavdf-seed17.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepfake_detection.evaluation.conformal import (
    conformal_threshold,
    minimum_calibration_size,
    neyman_pearson_threshold,
    operating_point,
)

CALIBRATION_SEED = 20261113


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-threshold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-cal", type=int, default=200)
    arguments = parser.parse_args(argv)

    with arguments.predictions.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["probability"].strip()]
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["probability"]) for r in rows])
    real = np.flatnonzero(labels == 0)
    if len(real) <= arguments.n_cal:
        raise SystemExit(
            f"only {len(real)} real clips; need more than n_cal={arguments.n_cal}"
        )
    calibration = np.random.default_rng(CALIBRATION_SEED).choice(
        real, arguments.n_cal, replace=False
    )
    held = np.setdiff1d(np.arange(len(rows)), calibration)

    source = float(
        json.loads(arguments.source_threshold.read_text(encoding="utf-8"))["threshold"]
    )
    thresholds = {
        "source_validation": source,
        "conformal_real_only": conformal_threshold(
            scores[calibration], arguments.alpha
        ),
        "umbrella_real_only": neyman_pearson_threshold(
            scores[calibration], arguments.alpha, arguments.delta
        ),
    }
    # The evaluation tools flag at >= threshold; the conformal module flags at >.
    # Scores are continuous, so the two agree except on exact ties.
    report = {
        "predictions": str(arguments.predictions).replace("\\", "/"),
        "alpha": arguments.alpha,
        "delta": arguments.delta,
        "n_cal": arguments.n_cal,
        "minimum_n_for_umbrella": minimum_calibration_size(
            arguments.alpha, arguments.delta
        ),
        "calibration_seed": CALIBRATION_SEED,
        "held_out_real": int((labels[held] == 0).sum()),
        "held_out_fake": int((labels[held] == 1).sum()),
        "results": {},
    }
    for name, value in thresholds.items():
        point = operating_point(value, labels[held], scores[held])
        report["results"][name] = {
            "threshold": value,
            "false_positive_rate": point.false_positive_rate,
            "true_positive_rate": point.true_positive_rate,
        }
    report["h4_success"] = (
        report["results"]["conformal_real_only"]["false_positive_rate"]
        <= arguments.alpha + 0.02
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, values in report["results"].items():
        print(
            f"{name:20} thr {values['threshold']:.4f}  FPR {values['false_positive_rate']:.3f}  TPR {values['true_positive_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
