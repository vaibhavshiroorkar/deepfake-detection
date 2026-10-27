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
import hashlib
import json
import sys
import uuid
from pathlib import Path


def split_masks(
    groups,
    names,
    *,
    source_pattern="prefix",
    modes_are_one_generator=True,
    test_fraction=0.4,
    seed=17,
    identity_protocol="target-only",
):
    """Split targets first, then optionally exclude shared FF++ donor identities."""
    import numpy as np

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if identity_protocol not in {"target-only", "ffpp-strict"}:
        raise ValueError("Unknown identity protocol")
    groups = np.asarray(groups)
    if modes_are_one_generator:
        groups = np.asarray(
            [g[:-4] if g.endswith(("_i2v", "_t2v")) else g for g in groups.tolist()]
        )
    stems = [Path(str(name)).stem for name in names]
    sources = np.asarray(
        [
            stem.split("_", 1)[0] if source_pattern == "prefix" else stem
            for stem in stems
        ]
    )
    identities = [set(stem.split("_")) for stem in stems]
    if identity_protocol == "ffpp-strict" and any(
        len(stem.split("_")) != (1 if group == "real" else 2)
        or not all(part.isdigit() for part in stem.split("_"))
        for stem, group in zip(stems, groups, strict=True)
    ):
        raise ValueError("ffpp-strict requires real ID and fake target_donor filenames")
    unique = sorted(set(sources.tolist()))
    if len(unique) < 2:
        raise ValueError("At least two source videos are required")
    order = np.random.default_rng(seed).permutation(len(unique))
    cut = min(len(unique) - 1, max(1, int(len(unique) * test_fraction)))
    test_sources = {unique[i] for i in order[:cut]}
    in_test_source = np.asarray([source in test_sources for source in sources])
    masks = {}
    for generator in sorted(set(groups.tolist()) - {"real"}):
        held = groups == generator
        train = (~held) & (~in_test_source)
        test = (held | (groups == "real")) & in_test_source
        if identity_protocol == "ffpp-strict":
            test_ids = set().union(*(identities[i] for i in np.flatnonzero(test)))
            train &= np.asarray([not ids & test_ids for ids in identities])
        masks[generator] = train, test
    return sources, groups, test_sources, masks


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


def held_out_scores(
    features,
    labels,
    groups,
    names,
    *,
    source_pattern: str = "prefix",
    modes_are_one_generator: bool = True,
    test_fraction: float = 0.4,
    regularization: float = 1.0,
    seed: int = 17,
    identity_protocol: str = "target-only",
):
    """Score every generator with a probe that never saw it.

    Returns the per-clip source ids, the merged generator labels, and one
    ``(test_index, scores, train_count)`` triple per generator. Comparisons
    between encoders call this so that both arms use the same split.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sources, groups, test_sources, masks = split_masks(
        groups,
        names,
        source_pattern=source_pattern,
        modes_are_one_generator=modes_are_one_generator,
        test_fraction=test_fraction,
        seed=seed,
        identity_protocol=identity_protocol,
    )

    arms: dict[str, tuple] = {}
    for generator, (train_mask, test_mask) in masks.items():
        if len(set(labels[train_mask].tolist())) != 2:
            raise ValueError(f"{generator}: training split needs both classes")
        if len(set(labels[test_mask].tolist())) != 2:
            raise ValueError(f"{generator}: test split needs both classes")

        scaler = StandardScaler().fit(features[train_mask])
        probe = LogisticRegression(
            C=regularization, max_iter=2000, class_weight="balanced"
        )
        probe.fit(scaler.transform(features[train_mask]), labels[train_mask])
        scores = probe.predict_proba(scaler.transform(features[test_mask]))[:, 1]
        arms[generator] = (np.flatnonzero(test_mask), scores, int(train_mask.sum()))
    return sources, groups, test_sources, arms


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

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
    parser.add_argument(
        "--identity-protocol",
        choices=("target-only", "ffpp-strict"),
        default="target-only",
        help="Target-only separation is the historical protocol. "
        "ffpp-strict also excludes donors shared with test clips.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.features.with_name(
        f"{arguments.features.stem}-logo-seed{arguments.seed}-{uuid.uuid4().hex[:12]}.json"
    )
    predictions = output.with_name(output.stem + "-predictions.json")
    if output.exists() or predictions.exists():
        raise FileExistsError(
            f"Evidence already exists at {output}; choose a new output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = np.load(arguments.features, allow_pickle=False)
    features = payload["features"]
    labels = payload["labels"]
    groups = payload["groups"]
    names = payload["files"]
    sources, groups, test_sources, arms = held_out_scores(
        features,
        labels,
        groups,
        names,
        source_pattern=arguments.source_pattern,
        modes_are_one_generator=arguments.modes_are_one_generator,
        test_fraction=arguments.test_fraction,
        regularization=arguments.regularization,
        seed=arguments.seed,
        identity_protocol=arguments.identity_protocol,
    )
    generators = list(arms)
    print(
        f"{len(labels)} clips, {int((groups == 'real').sum())} real, "
        f"{len(generators)} generators, {features.shape[1]} dimensions"
    )
    print(f"{len(set(sources.tolist()))} source videos, {len(test_sources)} held out\n")
    if len(generators) < 2:
        print("Leave-one-generator-out needs at least two generators.")
        return 1

    rows: dict[str, dict] = {}
    print(f"{'held-out generator':28} {'ROC-AUC':>8}  {'95% interval':>18}  clips")
    print("-" * 70)
    for generator, (test_index, scores, train_count) in arms.items():
        truth = labels[test_index]
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
            "test_clips": len(test_index),
            "train_clips": train_count,
        }
        print(
            f"{generator:28} {auc:8.4f}  [{low:.4f}, {high:.4f}]  {len(test_index):5}"
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

    _, _, _, masks = split_masks(
        groups,
        names,
        source_pattern=arguments.source_pattern,
        modes_are_one_generator=arguments.modes_are_one_generator,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
        identity_protocol=arguments.identity_protocol,
    )
    prediction_rows = [
        {
            "generator": generator,
            "file": str(names[index]),
            "source": str(sources[index]),
            "label": int(labels[index]),
            "probability_fake": float(score),
        }
        for generator, (indices, scores, _) in arms.items()
        for index, score in zip(indices, scores, strict=True)
    ]
    with predictions.open("x", encoding="utf-8") as handle:
        json.dump(prediction_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report = (
        json.dumps(
            {
                "protocol": "leave-one-generator-out",
                "features": str(arguments.features).replace("\\", "/"),
                "feature_sha256": hashlib.sha256(
                    arguments.features.read_bytes()
                ).hexdigest(),
                "predictions": str(predictions.resolve()).replace("\\", "/"),
                "predictions_sha256": hashlib.sha256(
                    predictions.read_bytes()
                ).hexdigest(),
                "seed": arguments.seed,
                "regularization": arguments.regularization,
                "test_fraction": arguments.test_fraction,
                "bootstrap_samples": arguments.bootstrap_samples,
                "source_pattern": arguments.source_pattern,
                "modes_are_one_generator": arguments.modes_are_one_generator,
                "identity_protocol": arguments.identity_protocol,
                "source_assignments": {
                    str(source): "test" if source in test_sources else "train"
                    for source in sorted(set(sources.tolist()))
                },
                "arm_assignments": {
                    generator: {
                        "train_files": names[train].tolist(),
                        "test_files": names[test].tolist(),
                    }
                    for generator, (train, test) in masks.items()
                },
                "generators": rows,
                "macro_mean": float(np.mean([r["roc_auc"] for r in rows.values()]))
                if rows
                else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
