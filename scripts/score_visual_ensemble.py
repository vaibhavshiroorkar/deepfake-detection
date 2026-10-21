"""Score the six visual checkpoints as one ensemble, from saved predictions.

The ensemble averages the logits of EfficientNet-B0 and ConvNeXt-Tiny at
seeds 17, 29 and 43, all trained on the same FakeAVCeleb split. Every weight
is equal and nothing is fitted, so no test partition touches the rule that
scores it.

One caveat belongs next to the number. "Average all six" was picked after
looking at three groupings on DFDC: EfficientNet only (0.7270), ConvNeXt only
(0.7398) and all six (0.7739). The paired intervals below say how much of the
gain over the best single model survives resampling by identity.

    uv run python scripts/score_visual_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERS = (
    ("efficientnet-b0-seed17", "visual-{split}-seed17"),
    ("efficientnet-b0-seed29", "visual-{split}-seed29"),
    ("efficientnet-b0-seed43", "visual-{split}-seed43"),
    # ConvNeXt seed 17 wrote its reports without a seed suffix.
    ("convnext-tiny-seed17", "visual-convnext-{split}"),
    ("convnext-tiny-seed29", "visual-convnext-{split}-seed29"),
    ("convnext-tiny-seed43", "visual-convnext-{split}-seed43"),
)
SPLITS = (
    ("in-domain-test", "FakeAVCeleb"),
    ("dfdc", "DFDC"),
    ("celebdf", "Celeb-DF-v2"),
)
# MNW has one genuine clip, so a ranking metric means nothing there. Its own
# metrics file says to read the detection rate instead, and so does this.
DETECTION_ONLY = (("mnw", "MNW"),)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logit))


def load_members(evaluation: Path, split: str) -> pd.DataFrame:
    """One row per clip that every member scored, with one column per member."""
    frame: pd.DataFrame | None = None
    for name, stem in MEMBERS:
        path = evaluation / f"{stem.format(split=split)}-predictions.csv"
        if not path.is_file() and stem.endswith("-seed17"):
            # The program run wrote seed 17's Celeb-DF and MNW reports unsuffixed.
            path = (
                evaluation
                / f"{stem.removesuffix('-seed17').format(split=split)}-predictions.csv"
            )
        member = pd.read_csv(path)
        member = member[member["probability"].notna()].set_index("clip_id")
        if frame is None:
            frame = member[["label", "source", "manipulation_type", "method"]].copy()
        frame = frame.join(member["probability"].rename(name), how="inner")
    assert frame is not None
    return frame


def ensemble_probability(frame: pd.DataFrame) -> np.ndarray:
    names = [name for name, _ in MEMBERS]
    return _sigmoid(_logit(frame[names].to_numpy()).mean(axis=1))


def main(argv: list[str] | None = None) -> int:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import (
        PairedPrediction,
        bootstrap_binary_metrics,
        paired_auc_difference,
    )
    from deepfake_detection.evaluation.metrics import EvaluationItem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("runs/program-20260906/evaluation"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    evaluation = arguments.evaluation_dir

    for split, dataset in SPLITS:
        frame = load_members(evaluation, split)
        probability = ensemble_probability(frame)
        labels = frame["label"].astype(int).to_numpy()
        sources = frame["source"].astype(str).to_numpy()
        items = [
            EvaluationItem(
                label=int(label),
                probability=float(p),
                source_identity=str(source),
                method=str(method),
            )
            for label, p, source, method in zip(
                labels, probability, sources, frame["method"], strict=True
            )
        ]
        intervals = bootstrap_binary_metrics(
            items,
            threshold=arguments.threshold,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        metrics = {name: interval.estimate for name, interval in intervals.items()}

        members = {
            name: float(roc_auc_score(labels, frame[name])) for name, _ in MEMBERS
        }
        seed_mean = _sigmoid(
            _logit(frame[[n for n, _ in MEMBERS[:3]]].to_numpy()).mean(axis=1)
        )

        def paired(
            right: np.ndarray,
            labels: np.ndarray = labels,
            sources: np.ndarray = sources,
            probability: np.ndarray = probability,
        ) -> dict[str, float]:
            interval = paired_auc_difference(
                [
                    PairedPrediction(
                        label=int(label),
                        source_identity=str(source),
                        left_probability=float(left),
                        right_probability=float(other),
                    )
                    for label, source, left, other in zip(
                        labels, sources, probability, right, strict=True
                    )
                ],
                samples=arguments.bootstrap_samples,
                seed=arguments.seed,
            )
            return asdict(interval)

        predicted = (probability >= arguments.threshold).astype(int)
        payload = {
            "branch": "visual-ensemble",
            "dataset": dataset,
            "members": [name for name, _ in MEMBERS],
            "combination": "mean of member logits, equal weights, nothing fitted",
            "fixed_threshold": arguments.threshold,
            "rows": int(len(frame)),
            "class_balance": {
                "fake": int(labels.sum()),
                "real": int((labels == 0).sum()),
            },
            "confusion": {
                "true_positive": int(((labels == 1) & (predicted == 1)).sum()),
                "false_positive": int(((labels == 0) & (predicted == 1)).sum()),
                "true_negative": int(((labels == 0) & (predicted == 0)).sum()),
                "false_negative": int(((labels == 1) & (predicted == 0)).sum()),
            },
            "metrics": metrics,
            "intervals": {name: asdict(value) for name, value in intervals.items()},
            "bootstrap_samples": arguments.bootstrap_samples,
            "bootstrap_unit": "source_identity",
            "member_roc_auc": members,
            "paired_roc_auc_difference": {
                "versus_efficientnet_seed17": paired(
                    frame["efficientnet-b0-seed17"].to_numpy()
                ),
                "versus_efficientnet_seed_mean": paired(seed_mean),
            },
            "selection_caveat": (
                "All six was chosen after comparing three groupings on DFDC "
                "(EfficientNet only, ConvNeXt only, all six)."
            ),
        }
        stem = f"visual-ensemble-{split}"
        (evaluation / f"{stem}-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        out = frame[["label", "source", "manipulation_type", "method"]].copy()
        out.insert(1, "probability", probability)
        out.insert(2, "predicted", predicted)
        out.to_csv(evaluation / f"{stem}-predictions.csv")

        auc = intervals["roc_auc"]
        print(f"\n{dataset}: {len(frame)} clips")
        for name, value in members.items():
            print(f"  {name:24} {value:.4f}")
        print(
            f"  {'ensemble':24} {auc.estimate:.4f} [{auc.lower:.4f}, {auc.upper:.4f}]"
            f"  EER {metrics['eer']:.4f}  BalAcc {metrics['balanced_accuracy']:.4f}"
            f"  ECE {metrics['expected_calibration_error']:.4f}"
        )
        for name, value in payload["paired_roc_auc_difference"].items():
            print(
                f"  {name}: {value['estimate']:+.4f} "
                f"[{value['lower']:+.4f}, {value['upper']:+.4f}]"
            )

    for split, dataset in DETECTION_ONLY:
        frame = load_members(evaluation, split)
        probability = ensemble_probability(frame)
        fake = frame["label"].astype(int).to_numpy() == 1
        detected = {
            name: int((frame[name].to_numpy()[fake] >= arguments.threshold).sum())
            for name, _ in MEMBERS
        }
        detected["ensemble"] = int((probability[fake] >= arguments.threshold).sum())
        payload = {
            "branch": "visual-ensemble",
            "dataset": dataset,
            "members": [name for name, _ in MEMBERS],
            "fixed_threshold": arguments.threshold,
            "rows": int(len(frame)),
            "fakes": int(fake.sum()),
            "real": int((~fake).sum()),
            "fakes_detected": detected,
            "note": (
                "Only one genuine clip, so ROC-AUC is not reported. Read the "
                "detection rate on the fakes."
            ),
        }
        (evaluation / f"visual-ensemble-{split}-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"\n{dataset}: {int(fake.sum())} fakes, detected at {arguments.threshold}"
        )
        for name, count in detected.items():
            print(f"  {name:24} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
