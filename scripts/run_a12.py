"""P-001 amendment A12 (exploratory): does a fusion fit on other corpora transfer?

For each target T, the A8 fusion (`fit_fusion`, unchanged) is fit on the pooled clips of the
other targets and scores T (leave one corpus out). An in-target cross-fit by identity
(`cross_fit_scores`, unchanged) gives an upper bound that uses T's own labels. Both are compared
with the A8 fusion and each single path on the same clips, with the A8 identity bootstrap.

Reads `{h2_dir}/fusion-{target}-seed{S}-predictions.csv` (clip label, identity, A8 fusion, branch
logits) and writes `{output}/a12.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from deepfake_detection.fusion.h2 import SeedPair, cross_fit_scores, fit_fusion, mean_seed_difference

TARGETS = ("dfdc", "lavdf-window", "hifiavdf")
SEEDS = (17, 29, 43)
PATHS = ("visual", "audio", "sync")


def load(h2_dir: Path, target: str, seed: int) -> dict:
    with (h2_dir / f"fusion-{target}-seed{seed}-predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "labels": np.array([int(r["label"]) for r in rows]),
        "groups": np.array([r["source"] for r in rows]),
        "x": np.array([[float(r[f"{p}_logit"]) for p in PATHS] for r in rows]),
        "a8": np.array([float(r["fusion"]) for r in rows]),
    }


def scores_for_seed(data: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    """Every score A12 compares, per target, for one seed."""
    out = {}
    for target, held in data.items():
        others = [d for name, d in data.items() if name != target]
        fusion = fit_fusion(np.vstack([d["x"] for d in others]), np.concatenate([d["labels"] for d in others]))
        out[target] = {
            "leave_one_out": fusion.score(held["x"]),
            "upper_bound": cross_fit_scores(held["x"], held["labels"], held["groups"]),
            "a8": held["a8"],
            **{p: held["x"][:, i] for i, p in enumerate(PATHS)},
        }
    return out


def compare(per_seed: list[dict], target: str, labels: np.ndarray, groups: np.ndarray) -> dict:
    """AUC of each score per seed, and the paired differences A12 reports."""
    names = ("leave_one_out", "upper_bound", "a8", *PATHS)
    result = {"auc": {n: [float(roc_auc_score(labels, s[target][n])) for s in per_seed] for n in names}}
    pairs = [("leave_one_out", other) for other in ("a8", *PATHS)] + [("upper_bound", p) for p in PATHS]
    result["differences"] = {}
    for first, second in pairs:
        seed_pairs = [SeedPair(labels, groups, s[target][first], s[target][second]) for s in per_seed]
        result["differences"][f"{first}-minus-{second}"] = mean_seed_difference(seed_pairs)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {seed: {t: load(args.h2_dir, t, seed) for t in TARGETS} for seed in SEEDS}
    for target in TARGETS:
        first = loaded[SEEDS[0]][target]
        for seed in SEEDS[1:]:
            if not np.array_equal(loaded[seed][target]["groups"], first["groups"]):
                raise ValueError(f"Seeds score different clips on {target}")
    per_seed = [scores_for_seed(loaded[seed]) for seed in SEEDS]
    report = {"amendment": "A12", "exploratory": True, "seeds": list(SEEDS), "targets": {}}
    for target in TARGETS:
        labels, groups = loaded[SEEDS[0]][target]["labels"], loaded[SEEDS[0]][target]["groups"]
        report["targets"][target] = {"clips": int(len(labels)), **compare(per_seed, target, labels, groups)}
        auc = report["targets"][target]["auc"]
        print(target, " ".join(f"{n}={np.mean(v):.3f}" for n, v in auc.items()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "a12.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
