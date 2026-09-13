"""Does camera motion make the visual model call a clip fake?

The claim has been in the documents as a mechanism with no measurement behind
it: `visual_view` samples 16 frames spread across a whole clip, so a hand-held
shot produces large frame-to-frame differences, and the model was trained to
read frame-to-frame difference as evidence of manipulation. If that is right,
motion should predict the score on clips the model has never seen, including on
the authentic ones where any such effect is pure false alarm.

Motion here is the mean absolute difference between consecutive frames of the
cached view, which is what the model is actually handed. It is not optical flow
and does not separate camera motion from subject motion. That is a limitation of
the measure, not a reason to skip it: both are motion the model was not trained
to expect, and the question is whether either drives the score.

    uv run python scripts/motion_vs_error.py --dataset dfdc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DATASETS = {
    "dfdc": ("dfdc-visual.csv", "DFDC", "visual-dfdc-predictions.csv"),
    "in-domain": ("test-usable.csv", "FakeAVCeleb", "visual-in-domain-test-predictions.csv"),
    "celebdf": ("celebdf-visual.csv", "Celeb-DF-v2", "visual-celebdf-predictions.csv"),
}


def _spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation, so a heavy-tailed motion distribution cannot dominate."""
    import numpy as np

    def ranks(values: list[float]):
        order = np.argsort(np.argsort(np.asarray(values)))
        return order.astype(float)

    a, b = ranks(left), ranks(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
    return float(a @ b / denominator) if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/program-20260906"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dfdc")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    manifest_name, dataset, predictions_name = DATASETS[arguments.dataset]
    predictions_path = arguments.run_dir / "evaluation" / predictions_name
    if not predictions_path.is_file():
        print(f"No predictions at {predictions_path}")
        return 1

    scores: dict[str, tuple[float, int]] = {}
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scores[row["clip_id"]] = (float(row["probability"]), int(row["label"]))

    index: dict[str, Path] = {}
    with (arguments.run_dir / "cache-index.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index[row["clip_id"]] = arguments.run_dir / row["cache_path"]
    store = CacheStore(arguments.run_dir / "cache")

    motion: list[float] = []
    probability: list[float] = []
    labels: list[int] = []
    missing = 0
    for clip_id, (score, label) in scores.items():
        path = index.get(clip_id)
        if path is None:
            missing += 1
            continue
        try:
            prepared = store.load(path, views=("visual_view",))
        except (OSError, ValueError):
            missing += 1
            continue
        view = prepared.visual_view
        if view is None or len(view) < 2:
            missing += 1
            continue
        frames = np.asarray(view, dtype=np.float32)
        motion.append(float(np.abs(np.diff(frames, axis=0)).mean()))
        probability.append(score)
        labels.append(label)

    if not motion:
        print("No clip had a cached visual view.")
        return 1

    motion_array = np.asarray(motion)
    probability_array = np.asarray(probability)
    label_array = np.asarray(labels)

    overall = _spearman(motion, probability)
    # The distribution, not just the correlation. A correlation says motion
    # moves the score; comparing the distributions across corpora says whether
    # the target corpus actually has more of it, which is what turns a
    # correlation into an explanation of the transfer gap.
    quantiles = [float(np.quantile(motion_array, q)) for q in (0.25, 0.5, 0.75)]
    # The tails as well as the middle. The distribution gate needs a lower
    # bound, and an interquartile fence puts it below zero, where no clip can
    # ever fall: motion is an absolute difference. Generated video being
    # unnaturally smooth is the case that bound exists to catch.
    tails = {
        f"p{int(q * 100):02d}": float(np.quantile(motion_array, q))
        for q in (0.01, 0.05, 0.95, 0.99)
    }
    result = {
        "dataset": dataset,
        "clips": len(motion),
        "missing": missing,
        "spearman_motion_vs_score": overall,
        "motion_mean": float(motion_array.mean()),
        "motion_quartiles": quantiles,
        "motion_tails": tails,
        "motion_min": float(motion_array.min()),
        "motion_max": float(motion_array.max()),
    }
    print(f"{dataset}: {len(motion):,} clips scored, {missing:,} without a view")
    print(
        f"motion: mean {motion_array.mean():.4f}, quartiles "
        + ", ".join(f"{value:.4f}" for value in quantiles)
    )
    print(
        "        tails " + ", ".join(f"{k} {v:.4f}" for k, v in tails.items())
        + f", min {motion_array.min():.4f}, max {motion_array.max():.4f}"
    )
    print(f"motion against fake probability, all clips : {overall:+.3f}")

    for name, mask in (
        ("authentic only", label_array == 0),
        ("manipulated only", label_array == 1),
    ):
        if mask.sum() < 10:
            continue
        value = _spearman(
            motion_array[mask].tolist(), probability_array[mask].tolist()
        )
        result[f"spearman_{name.split()[0]}"] = value
        print(f"motion against fake probability, {name:15}: {value:+.3f}")

    # The deployment-relevant number: among genuine clips, how much more often
    # does the model cry fake in the top motion quartile than the bottom.
    authentic = label_array == 0
    if authentic.sum() >= 20:
        values = motion_array[authentic]
        called = probability_array[authentic] >= 0.5
        low = values <= np.quantile(values, 0.25)
        high = values >= np.quantile(values, 0.75)
        low_rate = float(called[low].mean())
        high_rate = float(called[high].mean())
        result["false_alarm_low_motion"] = low_rate
        result["false_alarm_high_motion"] = high_rate
        print(
            f"\nfalse alarms on genuine clips: {low_rate:.1%} in the calmest "
            f"quartile, {high_rate:.1%} in the most moving quartile"
        )

    output = (
        arguments.output
        or arguments.run_dir / "evaluation" / f"motion-{arguments.dataset}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
