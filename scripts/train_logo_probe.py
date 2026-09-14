"""Leave-one-generator-out probe over cached frozen features.

The protocol, and the only headline this project reports: every number comes
from a probe that never saw the generator it is scored on. There is no in-domain
column. Detecting a generator you trained on is not the problem, and a strong
in-domain number has three times here turned out to be a shortcut rather than
skill: 0.9990 against 0.4920 cross-corpus, a 0.9969 matched-pair stream reading
0.5059 once capture conditions changed, and a 0.9997 on Veo 3 that was entirely
content confound.

One arm per generator, and the split is on the **source video** as well as the
generator. Both parts matter and the second one was added after a smoke test
returned 1.0000 on every arm.

A generator-only split leaks twice. The real clips end up in training and test
together, so the probe memorises them. And in corpora where a fake is derived
from a specific real clip, which is true of FaceForensics++ by construction and
of DF26 by prompt, the held-out generator's fake can be matched to its own
source's content sitting on the training side. Splitting source ids first closes
both: a source video appears in training or in test, never in both, whichever
generator produced the clip.

Intervals are bootstrap over clips. Clustering by identity is not available here
because generated clips have no speaker identity to cluster on, so these are
narrower than the identity-clustered intervals elsewhere in the project and
should not be compared against them directly.

    uv run python scripts/train_logo_probe.py --features data/df26-scored/features-dinov3.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap(scores, labels, samples: int, seed: int):
    """Percentile interval over clip resamples."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    count = len(labels)
    for _ in range(samples):
        index = rng.integers(0, count, count)
        drawn = labels[index]
        if len(set(drawn.tolist())) != 2:
            continue
        values.append(roc_auc_score(drawn, scores[index]))
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--source-pattern",
        choices=("prefix", "stem"),
        default="prefix",
        help="How a filename names its source video. 'prefix' takes everything "
        "before the first underscore, which is FaceForensics++'s target_source "
        "convention. 'stem' takes the whole filename.",
    )
    parser.add_argument(
        "--modes-are-one-generator",
        action="store_true",
        default=True,
        help="Treat a model's image-to-video and text-to-video outputs as one "
        "generator. They share weights, so holding out one while training on "
        "the other is not an unseen generator.",
    )
    parser.add_argument(
        "--split-modes",
        dest="modes_are_one_generator",
        action="store_false",
        help="Score each generation mode as its own generator.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.4,
        help="Share of source videos held out for testing.",
    )
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    payload = np.load(arguments.features, allow_pickle=False)
    features = payload["features"]
    labels = payload["labels"]
    groups = payload["groups"]
    names = payload["files"]
    if arguments.modes_are_one_generator:
        # LTX_2.3_distilled_i2v and LTX_2.3_distilled_t2v are one model in two
        # modes. Holding out a mode while training on its sibling leaks the
        # model, which is the same class of leak as sharing a source video.
        groups = np.asarray(
            [
                g[:-4] if g.endswith(("_i2v", "_t2v")) else g
                for g in groups.tolist()
            ]
        )
    generators = sorted({g for g in groups.tolist() if g != "real"})
    real = groups == "real"

    def source_of(name: str) -> str:
        stem = Path(str(name)).stem
        return stem.split("_", 1)[0] if arguments.source_pattern == "prefix" else stem

    sources = np.asarray([source_of(n) for n in names.tolist()])
    unique = sorted(set(sources.tolist()))
    rng = np.random.default_rng(arguments.seed)
    order = rng.permutation(len(unique))
    cut = max(1, int(len(unique) * arguments.test_fraction))
    test_sources = {unique[i] for i in order[:cut]}
    in_test_source = np.asarray([s in test_sources for s in sources.tolist()])
    print(
        f"{len(labels)} clips, {int(real.sum())} real, "
        f"{len(generators)} generators, {features.shape[1]} dimensions"
    )
    print(f"{len(unique)} source videos, {len(test_sources)} held out\n")
    if len(generators) < 2:
        print("Leave-one-generator-out needs at least two generators.")
        return 1

    rows: dict[str, dict] = {}
    print(f"{'held-out generator':28} {'ROC-AUC':>8}  {'95% interval':>18}  clips")
    print("-" * 70)
    for generator in generators:
        held = groups == generator
        # Two exclusions, not one. The held-out generator never appears in
        # training, and no source video appears on both sides, so neither the
        # real clips nor a fake's own source can be memorised.
        train_mask = (~held) & (~in_test_source)
        test_mask = (held | real) & in_test_source

        scaler = StandardScaler().fit(features[train_mask])
        probe = LogisticRegression(
            C=arguments.regularization, max_iter=2000, class_weight="balanced"
        )
        probe.fit(scaler.transform(features[train_mask]), labels[train_mask])
        scores = probe.predict_proba(scaler.transform(features[test_mask]))[:, 1]
        truth = labels[test_mask]
        if len(set(truth.tolist())) != 2:
            print(f"{generator:28} one class, skipped")
            continue
        auc = float(roc_auc_score(truth, scores))
        low, high = _bootstrap(
            scores, truth, arguments.bootstrap_samples, arguments.seed
        )
        rows[generator] = {
            "roc_auc": auc,
            "ci": [low, high],
            "test_clips": int(test_mask.sum()),
            "train_clips": int(train_mask.sum()),
        }
        print(
            f"{generator:28} {auc:8.4f}  [{low:.4f}, {high:.4f}]  "
            f"{int(test_mask.sum()):5}"
        )

    if rows:
        values = [row["roc_auc"] for row in rows.values()]
        macro = float(np.mean(values))
        spread = float(max(values) - min(values))
        print("-" * 70)
        print(f"{'macro mean':28} {macro:8.4f}")
        print(f"{'spread, best minus worst':28} {spread:8.4f}")
        if max(values) > 0.99:
            print(
                "\nA generator above 0.99 under this protocol usually means a "
                "leak rather than a success. Check that the held-out generator "
                "shares no source video with the training side."
            )

    output = arguments.output or arguments.features.with_name(
        arguments.features.stem + "-logo.json"
    )
    output.write_text(
        json.dumps(
            {
                "protocol": "leave-one-generator-out",
                "features": str(arguments.features).replace("\\", "/"),
                "generators": rows,
                "macro_mean": float(np.mean([r["roc_auc"] for r in rows.values()]))
                if rows
                else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
