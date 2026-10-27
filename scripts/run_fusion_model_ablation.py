"""Which fusion head, and does calibration or the quality features earn its place?

Three ablations the research design names and that were never run. All three
read the out-of-fold feature store the shipped head was fitted on, so the branch
checkpoints, the split and the preprocessing hash are held fixed and the only
thing varying is the head.

**FUS-01, logistic against the small MLP.** `train fusion --model mlp` has
existed the whole time and was never measured against the default. The roadmap
asks for the comparison and the scope controls say to prefer the cheaper model
when the two are statistically indistinguishable, so the answer needs an
interval rather than two bare numbers.

**Calibration.** The shipped head fits a per-branch Platt scaler on the
out-of-fold logits before the head sees them. Expected calibration error on this
pipeline moves from 0.0022 in-domain to 0.6366 on DFDC, so the scaler is clearly
not transporting. Whether it was buying anything in the first place is a
separate question, and this arm asks it by feeding the raw logits instead.

**Quality features.** Face coverage, audio clipping and audio-video duration
delta enter the head alongside the branch scores. They are the head's only route
to knowing a clip is hard, and dropping them says whether that route carries
anything.

A fifth arm averages the calibrated probabilities with no learned head at all.
It is the floor: a head that cannot beat an unweighted mean of its own inputs
has not learned a combination rule, and on three branches that is a real
possibility rather than a rhetorical one.

Every arm is scored on the held-out in-domain test partition and on DFDC,
because those two disagree everywhere else in this project and the disagreement
has been the finding every time. Intervals are bootstrapped over source
identity, not over clips, since FakeAVCeleb puts many clips on one speaker.

    uv run python scripts/run_fusion_model_ablation.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

BRANCHES = ("visual", "audio", "sync")


def _assemble(store_path: Path):
    """Clips readable by all three branches, with their logits and metadata.

    A clip missing any branch is dropped rather than imputed. The late head
    demands every column, so a partial clip is outside what it can score at all,
    and the dropped count is printed rather than absorbed.
    """
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
            "method": row.method,
            "race": row.race,
            "gender": row.gender,
            "quality": (
                row.face_coverage,
                row.audio_clipped,
                row.av_duration_delta_sec,
            ),
        }
    complete = sorted(k for k, v in logits.items() if len(v) == len(BRANCHES))
    return complete, logits, meta


def _samples(ids, logits, meta, branches):
    from deepfake_detection.fusion.late import FusionSample

    return [
        FusionSample(
            branch_logits={b: logits[i][b] for b in branches},
            face_coverage=meta[i]["quality"][0],
            audio_clipped=meta[i]["quality"][1],
            av_duration_delta_sec=meta[i]["quality"][2],
        )
        for i in ids
    ]


def _matrix(ids, logits, meta, *, calibrators, quality: bool):
    """Feature matrix for the ablation arms, built the way the shipped head does.

    A `calibrators` of None feeds the raw branch logit straight through, which
    is the calibration ablation. Column order matches LateFusion so the
    reimplementation can be checked against it rather than trusted.
    """
    import numpy as np

    columns = []
    for name in BRANCHES:
        values = np.asarray([[logits[i][name]] for i in ids], dtype=np.float64)
        if calibrators is None:
            columns.append(values[:, 0])
            continue
        probabilities = calibrators[name].predict_proba(values)[:, 1]
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        columns.append(np.log(probabilities / (1 - probabilities)))
    if quality:
        columns.extend(
            [
                np.asarray([meta[i]["quality"][0] for i in ids], dtype=np.float64),
                np.asarray(
                    [float(meta[i]["quality"][1]) for i in ids], dtype=np.float64
                ),
                np.asarray([meta[i]["quality"][2] for i in ids], dtype=np.float64),
            ]
        )
    return np.column_stack(columns)


def _fit_calibrators(ids, logits, labels):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    calibrators = {}
    for name in BRANCHES:
        values = np.asarray([[logits[i][name]] for i in ids], dtype=np.float64)
        calibrator = LogisticRegression(C=1.0, solver="liblinear")
        calibrator.fit(values, labels)
        calibrators[name] = calibrator
    return calibrators


def _interval(ids, meta, probabilities, *, seed: int, samples: int):
    """Identity-clustered ROC-AUC interval for one arm on one partition."""
    from dataclasses import dataclass

    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    @dataclass(frozen=True, slots=True)
    class Scored:
        label: int
        probability: float
        source_identity: str

    items = [
        Scored(meta[i]["label"], float(p), meta[i]["source_identity"])
        for i, p in zip(ids, probabilities, strict=True)
    ]

    def statistic(rows):
        labels = [r.label for r in rows]
        if len(set(labels)) < 2:
            raise ValueError("Bootstrap sample does not contain both classes")
        return float(roc_auc_score(labels, [r.probability for r in rows]))

    return cluster_bootstrap_interval(items, statistic, samples=samples, seed=seed)


def _paired(ids, meta, left, right, *, seed: int, samples: int):
    from deepfake_detection.evaluation.bootstrap import (
        PairedPrediction,
        paired_auc_difference,
    )

    return paired_auc_difference(
        [
            PairedPrediction(
                label=meta[i]["label"],
                source_identity=meta[i]["source_identity"],
                left_probability=float(a),
                right_probability=float(b),
            )
            for i, a, b in zip(ids, left, right, strict=True)
        ],
        samples=samples,
        seed=seed,
    )


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from deepfake_detection.evaluation.metrics import binary_metrics
    from deepfake_detection.fusion.late import LateFusion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    run = arguments.run_dir

    train_ids, train_logits, train_meta = _assemble(run / "features" / "oof.parquet")
    train_labels = [train_meta[i]["label"] for i in train_ids]
    partitions = {}
    for name, store in (("in-domain", "test.parquet"), ("dfdc", "dfdc.parquet")):
        path = run / "features" / store
        if path.is_file():
            partitions[name] = _assemble(path)

    print(f"out-of-fold training clips: {len(train_ids):,}")
    for name, (ids, _, meta) in partitions.items():
        y = [meta[i]["label"] for i in ids]
        identities = len({meta[i]["source_identity"] for i in ids})
        print(
            f"{name}: {len(ids):,} clips "
            f"(fake {sum(y)}, real {len(y) - sum(y)}, identities {identities})"
        )
    print()

    calibrators = _fit_calibrators(train_ids, train_logits, train_labels)

    def logistic():
        return LogisticRegression(C=1.0, solver="liblinear")

    arms: dict[str, dict] = {}

    # The two shipped arms go through LateFusion itself, so the default is
    # measured exactly as it ships rather than as this script rebuilds it.
    for kind in ("logistic", "mlp"):
        model = LateFusion(branch_names=BRANCHES, classifier_kind=kind).fit(
            _samples(train_ids, train_logits, train_meta, BRANCHES), train_labels
        )
        arms[kind] = {
            "description": f"shipped head, {kind}: calibrated logits plus quality",
            "scores": {
                name: model.predict_proba(_samples(ids, lg, mt, BRANCHES))
                for name, (ids, lg, mt) in partitions.items()
            },
        }

    for label, use_calibration, use_quality, description in (
        (
            "logistic-raw-logits",
            False,
            True,
            "calibration ablation: raw branch logits plus quality",
        ),
        (
            "logistic-no-quality",
            True,
            False,
            "quality ablation: calibrated logits only",
        ),
    ):
        cal = calibrators if use_calibration else None
        head = logistic().fit(
            _matrix(
                train_ids,
                train_logits,
                train_meta,
                calibrators=cal,
                quality=use_quality,
            ),
            train_labels,
        )
        arms[label] = {
            "description": description,
            "scores": {
                name: head.predict_proba(
                    _matrix(ids, lg, mt, calibrators=cal, quality=use_quality)
                )[:, 1]
                for name, (ids, lg, mt) in partitions.items()
            },
        }

    def mean_calibrated(ids, lg):
        columns = []
        for name in BRANCHES:
            values = np.asarray([[lg[i][name]] for i in ids], dtype=np.float64)
            columns.append(calibrators[name].predict_proba(values)[:, 1])
        return np.column_stack(columns).mean(axis=1)

    arms["mean-calibrated"] = {
        "description": "no learned head: unweighted mean of calibrated probabilities",
        "scores": {
            name: mean_calibrated(ids, lg) for name, (ids, lg, _) in partitions.items()
        },
    }

    # The rebuilt feature matrix must reproduce the shipped head, or the
    # ablation arms measure the rebuild rather than the ablation.
    check = logistic().fit(
        _matrix(
            train_ids, train_logits, train_meta, calibrators=calibrators, quality=True
        ),
        train_labels,
    )
    for name, (ids, lg, mt) in partitions.items():
        mine = check.predict_proba(
            _matrix(ids, lg, mt, calibrators=calibrators, quality=True)
        )[:, 1]
        theirs = arms["logistic"]["scores"][name]
        delta = float(np.abs(mine - theirs).max())
        if delta > 1e-9:
            print(
                f"rebuild check FAILED on {name}: max probability delta {delta:.3e}",
                file=sys.stderr,
            )
            return 1
    print("rebuild check: the ablation arms reproduce the shipped head exactly\n")

    report: dict[str, dict] = {}
    for label, arm in arms.items():
        report[label] = {"description": arm["description"], "partitions": {}}
        for name, (ids, _, meta) in partitions.items():
            probabilities = arm["scores"][name]
            labels = [meta[i]["label"] for i in ids]
            interval = _interval(
                ids,
                meta,
                probabilities,
                seed=arguments.seed,
                samples=arguments.bootstrap_samples,
            )
            metrics = binary_metrics(
                labels=labels,
                probabilities=[float(p) for p in probabilities],
                threshold=0.5,
            )
            entry = {
                "roc_auc": interval.estimate,
                "roc_auc_lower": interval.lower,
                "roc_auc_upper": interval.upper,
                "expected_calibration_error": metrics.expected_calibration_error,
                "balanced_accuracy": metrics.balanced_accuracy,
                "brier": metrics.brier,
                "clips": len(ids),
            }
            if label != "logistic":
                paired = _paired(
                    ids,
                    meta,
                    probabilities,
                    arms["logistic"]["scores"][name],
                    seed=arguments.seed,
                    samples=arguments.bootstrap_samples,
                )
                entry["paired_difference_vs_logistic"] = {
                    "estimate": paired.estimate,
                    "lower": paired.lower,
                    "upper": paired.upper,
                    "separated_from_zero": paired.lower > 0 or paired.upper < 0,
                }
            report[label]["partitions"][name] = entry

    order = [
        "logistic",
        "mlp",
        "logistic-raw-logits",
        "logistic-no-quality",
        "mean-calibrated",
    ]
    width = max(len(a) for a in order)
    for name in partitions:
        print(f"{name}:")
        print(
            f"  {'arm':{width}}  {'ROC-AUC':>8}  {'95% interval':>18}  "
            f"{'ECE':>7}  {'vs logistic':>26}"
        )
        print("  " + "-" * (width + 66))
        for label in order:
            entry = report[label]["partitions"][name]
            paired = entry.get("paired_difference_vs_logistic")
            if paired is None:
                verdict = "reference"
            else:
                mark = "*" if paired["separated_from_zero"] else " "
                verdict = (
                    f"{paired['estimate']:+.4f} "
                    f"[{paired['lower']:+.4f}, {paired['upper']:+.4f}]{mark}"
                )
            print(
                f"  {label:{width}}  {entry['roc_auc']:8.4f}  "
                f"[{entry['roc_auc_lower']:.4f}, {entry['roc_auc_upper']:.4f}]  "
                f"{entry['expected_calibration_error']:7.4f}  {verdict:>26}"
            )
        print()
    print("* interval excludes zero\n")

    payload = {
        "out_of_fold_clips": len(train_ids),
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_seed": arguments.seed,
        "bootstrap_unit": "source_identity",
        "branches": list(BRANCHES),
        "arms": report,
    }
    out = arguments.output or run / "evaluation" / "fusion-model-ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
