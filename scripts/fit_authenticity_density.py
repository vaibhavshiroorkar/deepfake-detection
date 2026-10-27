"""A density model of authentic capture, for abstaining instead of accusing.

The problem this exists for is measured and severe. At the fixed 0.50 threshold
the visual branch calls 0.00 percent of genuine FakeAVCeleb clips fake, 4.14
percent of genuine DFDC clips, and 90.97 percent of genuine Celeb-DF clips:
141 of 155 authentic videos accused. Celeb-DF's real clips are shot differently,
and that alone moves the false-alarm rate by ninety-one points.

Raising the threshold does not fix it. Forcing zero false alarms on Celeb-DF
needs a 0.9998 cut-off, at which nothing is detected at all, because the two
score distributions overlap almost completely.

The motion gate in `inference/distribution_gate.py` does not fix it either. It
intercepts 5.7 percent of those false alarms, because Celeb-DF's genuine clips
sit at motion 0.2964, inside the training range. They are unfamiliar in
appearance, not in motion, and one hand-picked statistic cannot see that.

So this fits a Gaussian over frozen encoder features of *authentic clips only*,
and scores a new clip by Mahalanobis distance from it. No fake example is
involved, which is what makes it work on capture conditions nobody anticipated:
the question is "is this unlike the footage I know", not "is this like a fake I
have seen".

The output is an abstention, not a verdict. A clip the model has no business
judging should produce "outside what I was trained on", which is defensible,
rather than "manipulated", which is a false accusation.

    uv run python scripts/fit_authenticity_density.py \\
        --features runs/probe-20260914/ood-dinov3.npz --reference real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--reference",
        default="real",
        help="Group holding the authentic clips the model is fitted on.",
    )
    parser.add_argument(
        "--flag-rate",
        type=float,
        default=0.05,
        help="Share of the reference distribution to flag. The threshold is set "
        "so this fraction of known-good footage is called unfamiliar, which is "
        "the price paid for catching the unfamiliar footage that matters.",
    )
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--components",
        type=int,
        default=48,
        help="PCA width before fitting. The encoder emits 768 dimensions and "
        "there are a few hundred reference clips, so a full covariance is "
        "badly estimated and in-sample distances come back optimistically "
        "small.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--splits",
        type=int,
        default=20,
        help="Also refit on this many reference splits, seeds counting up from "
        "--seed, and report the spread. One split is one draw: at a 0.10 flag "
        "rate, seed 17 intercepts 74.4 percent where twenty splits range from "
        "49.4 to 83.1.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("runs/probe-20260914/authenticity-density.npz"),
        help="Where the served gate loads the fitted model from.",
    )
    parser.add_argument("--encoder", default="dinov3")
    arguments = parser.parse_args(argv)

    payload = np.load(arguments.features, allow_pickle=False)
    features = payload["features"].astype(np.float64)
    groups = payload["groups"]
    reference = groups == arguments.reference
    if reference.sum() < 20:
        print(f"Only {int(reference.sum())} clips in group {arguments.reference!r}.")
        return 1

    from sklearn.decomposition import PCA

    def fit(seed: int):
        # The reference splits in two. One half fits the model, the other sets
        # the threshold. Setting it on the fitted half is what produced a first
        # run where held-out clips from the *same* corpus flagged at 100
        # percent: the distances the model was fitted on are not the distances
        # it will see.
        rng = np.random.default_rng(seed)
        index = rng.permutation(int(reference.sum()))
        pool = features[reference]
        split = len(index) // 2
        fitted, calibration = pool[index[:split]], pool[index[split:]]

        components = min(arguments.components, len(fitted) - 1, features.shape[1])
        # The exact solver. With 768 dimensions and 125 rows, "auto" picks the
        # randomized one, and with no random state the same seed then gave
        # 75.0 and 77.5 percent interception on consecutive runs.
        projection = PCA(n_components=components, svd_solver="full").fit(fitted)
        reduced = projection.transform(fitted)
        mean = reduced.mean(axis=0)
        centred = reduced - mean
        covariance = centred.T @ centred / (len(reduced) - 1)
        covariance += (
            arguments.shrinkage * np.trace(covariance) / components * np.eye(components)
        )
        inverse = np.linalg.inv(covariance)

        def distance(values):
            delta = projection.transform(values) - mean
            return np.sqrt(np.einsum("ij,jk,ik->i", delta, inverse, delta))

        threshold = float(np.quantile(distance(calibration), 1 - arguments.flag_rate))
        return projection, mean, inverse, threshold, distance

    projection, mean, inverse, threshold, distance = fit(arguments.seed)
    print(
        f"fitted on {int(reference.sum())} authentic clips from "
        f"{arguments.reference!r}, {features.shape[1]} dimensions"
    )
    print(
        f"threshold {threshold:.2f} flags {arguments.flag_rate:.0%} of the "
        "reference itself\n"
    )

    print(f"{'group':32} {'clips':>6} {'median distance':>16} {'flagged':>9}")
    print("-" * 68)
    results: dict[str, dict] = {}
    for group in sorted(set(groups.tolist())):
        values = distance(features[groups == group])
        flagged = float((values > threshold).mean())
        results[group] = {
            "clips": int(len(values)),
            "median_distance": float(np.median(values)),
            "flagged": flagged,
        }
        print(f"{group:32} {len(values):6} {np.median(values):16.2f} {flagged:8.1%}")

    spread: dict[str, dict] = {}
    if arguments.splits > 1:
        draws: dict[str, list[float]] = {g: [] for g in results}
        for seed in range(arguments.seed, arguments.seed + arguments.splits):
            *_, split_threshold, split_distance = fit(seed)
            for group in results:
                values = split_distance(features[groups == group])
                draws[group].append(float((values > split_threshold).mean()))
        print(f"\nflagged over {arguments.splits} reference splits")
        for group, values in draws.items():
            spread[group] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            print(
                f"{group:32} mean {np.mean(values):6.1%}, "
                f"{np.min(values):.1%} to {np.max(values):.1%}"
            )

    from deepfake_detection.inference.authenticity import AuthenticityDensity, save

    arguments.save_model.parent.mkdir(parents=True, exist_ok=True)
    save(
        arguments.save_model,
        AuthenticityDensity(
            mean=projection.mean_,
            components=projection.components_,
            centre=mean,
            inverse=inverse,
            threshold=threshold,
            encoder=arguments.encoder,
            reference=arguments.reference,
            flag_rate=arguments.flag_rate,
        ),
    )
    print(f"model saved to {arguments.save_model}")

    output = arguments.output or arguments.features.with_name(
        arguments.features.stem + "-density.json"
    )
    output.write_text(
        json.dumps(
            {
                "reference": arguments.reference,
                "flag_rate": arguments.flag_rate,
                "threshold": threshold,
                "groups": results,
                "seed": arguments.seed,
                "splits": arguments.splits,
                "split_spread": spread,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
