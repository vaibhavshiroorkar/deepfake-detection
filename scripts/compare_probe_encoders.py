"""Paired comparison of frozen encoders under leave-one-generator-out.

Each encoder's macro ROC-AUC already exists as its own `-logo.json`, but two
unpaired intervals cannot say whether one encoder beats another: the arms share
their test clips, so the encoders' errors are correlated and the intervals
overlap more than the difference does. This scores every encoder on the same
split, then bootstraps the difference.

The resampling unit is the source video, not the clip. Every arm scores the same
held-out real clips, and a fake shares its source with the real clip it came
from, so resampling clips would treat correlated rows as independent. Drawing
source videos keeps them together. The interval is therefore wider than the
per-arm clip intervals in the `-logo.json` files.

    uv run python scripts/compare_probe_encoders.py \\
        --features runs/probe-20260914/df26-norm-dinov3.npz \\
        runs/probe-20260914/df26-norm-pe.npz runs/probe-20260914/df26-norm-clip.npz \\
        --output runs/probe-20260914/df26-norm-encoder-comparison.json

The first file is the reference every other encoder is compared against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train_logo_probe import held_out_scores


def _encoder_name(path: Path) -> str:
    # df26-norm-pe.npz -> pe
    return path.stem.rsplit("-", 1)[-1]


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if len(arguments.features) < 2:
        parser.error("A comparison needs at least two feature files.")

    payloads = [np.load(path, allow_pickle=False) for path in arguments.features]
    names = [_encoder_name(path) for path in arguments.features]
    reference = payloads[0]
    for name, payload in zip(names[1:], payloads[1:], strict=True):
        # A pairing is only valid if row i is the same clip in every file.
        if not (
            np.array_equal(payload["files"], reference["files"])
            and np.array_equal(payload["labels"], reference["labels"])
        ):
            print(f"{name} does not list the same clips as {names[0]}.")
            return 1

    labels = reference["labels"]
    arms_by_encoder: dict[str, dict] = {}
    for name, payload in zip(names, payloads, strict=True):
        sources, _groups, test_sources, arms = held_out_scores(
            payload["features"],
            payload["labels"],
            payload["groups"],
            payload["files"],
            seed=arguments.seed,
        )
        arms_by_encoder[name] = arms
    generators = list(arms_by_encoder[names[0]])
    for name in names[1:]:
        for generator in generators:
            if not np.array_equal(
                arms_by_encoder[name][generator][0],
                arms_by_encoder[names[0]][generator][0],
            ):
                print(f"{name} scored a different test set for {generator}.")
                return 1

    def macro(name: str, rows) -> tuple[float, dict[str, float]]:
        per_arm = {}
        for generator in generators:
            test_index, scores, _train = arms_by_encoder[name][generator]
            chosen = rows[generator]
            truth = labels[test_index][chosen]
            if len(set(truth.tolist())) != 2:
                return float("nan"), {}
            per_arm[generator] = float(roc_auc_score(truth, scores[chosen]))
        return float(np.mean(list(per_arm.values()))), per_arm

    everything = {
        generator: np.arange(len(arms_by_encoder[names[0]][generator][0]))
        for generator in generators
    }
    point = {name: macro(name, everything) for name in names}

    # Position of each clip inside each arm's score array, -1 when the clip is
    # not in that arm's test set.
    position = {}
    for generator in generators:
        test_index = arms_by_encoder[names[0]][generator][0]
        lookup = np.full(len(labels), -1)
        lookup[test_index] = np.arange(len(test_index))
        position[generator] = lookup
    held_sources = sorted(test_sources)
    clips_of = {source: np.flatnonzero(sources == source) for source in held_sources}

    rng = np.random.default_rng(arguments.seed)
    draws: dict[str, list[float]] = {name: [] for name in names[1:]}
    arm_draws: dict[str, dict[str, list[float]]] = {
        name: {generator: [] for generator in generators} for name in names[1:]
    }
    skipped = 0
    for _ in range(arguments.bootstrap_samples):
        drawn = rng.choice(len(held_sources), len(held_sources), replace=True)
        clips = np.concatenate([clips_of[held_sources[i]] for i in drawn])
        rows = {}
        for generator in generators:
            chosen = position[generator][clips]
            rows[generator] = chosen[chosen >= 0]
        base, base_arms = macro(names[0], rows)
        if np.isnan(base):
            skipped += 1
            continue
        for name in names[1:]:
            value, arms = macro(name, rows)
            draws[name].append(value - base)
            for generator in generators:
                arm_draws[name][generator].append(
                    arms[generator] - base_arms[generator]
                )

    def interval(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    comparisons = {}
    print(f"reference {names[0]}: macro {point[names[0]][0]:.4f}")
    for name in names[1:]:
        difference = point[name][0] - point[names[0]][0]
        low, high = interval(draws[name])
        separated = low > 0 or high < 0
        comparisons[name] = {
            "macro_mean": point[name][0],
            "difference": difference,
            "ci": [low, high],
            "separated": separated,
            "per_generator": {
                generator: {
                    "difference": point[name][1][generator]
                    - point[names[0]][1][generator],
                    "ci": interval(arm_draws[name][generator]),
                }
                for generator in generators
            },
        }
        print(
            f"{name:>8}: macro {point[name][0]:.4f}, difference "
            f"{difference:+.4f} [{low:+.4f}, {high:+.4f}]"
            f"{'' if separated else ', not separated'}"
        )

    arguments.output.write_text(
        json.dumps(
            {
                "protocol": "leave-one-generator-out",
                "resampling_unit": "source video",
                "bootstrap_samples": arguments.bootstrap_samples,
                "skipped_draws": skipped,
                "seed": arguments.seed,
                "reference": {
                    "encoder": names[0],
                    "macro_mean": point[names[0]][0],
                    "features": str(arguments.features[0]).replace("\\", "/"),
                },
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
