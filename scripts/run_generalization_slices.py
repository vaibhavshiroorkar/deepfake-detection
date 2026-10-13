"""Where the fused system fails: by manipulation method, and by subgroup.

Three slices of the shipped late-fusion head, all read off the feature stores
the head was already fitted and scored on. No training, no GPU, and no new
decisions: this reports the breakdown of a result that already exists.

**Per-method.** One ROC-AUC per manipulation method, each method's fakes against
the same pool of real clips. A single headline AUC over a corpus whose fakes are
70 percent one method is mostly a measurement of that method, and the per-method
column says how much of the number is carried by the rest.

**Method holdout for the head.** The head is refitted with one method dropped
from the out-of-fold set, then scored on exactly that method. This bounds one
narrow question: are the head's weights method-specific? It is **not** a
leave-one-method-out generalization result, and must never be reported as one.
The branch checkpoints were trained on every method, so the held-out method is
still seen by everything upstream of the head. A real unseen-method number needs
the branches retrained per fold, which is GPU work this script does not do.

**Subgroups.** Race and gender, in-domain only. DFDC carries neither annotation,
so every DFDC row there would read `unknown`, and a subgroup table that silently
covers one corpus is worse than no table.

Real clips are the scarce resource in every slice here: 75 of them in the
in-domain test partition, against 1,573 fakes. Every interval below is driven by
that denominator, and the counts are printed beside each row for that reason.

    uv run python scripts/run_generalization_slices.py --run-dir runs/program-20260906
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path

BRANCHES = ("visual", "audio", "sync")


@dataclass(frozen=True, slots=True)
class Scored:
    label: int
    probability: float
    source_identity: str


def _assemble(store_path: Path):
    """Clips readable by all three branches, with their logits and metadata."""
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


def _samples(ids, logits, meta):
    from deepfake_detection.fusion.late import FusionSample

    return [
        FusionSample(
            branch_logits={b: logits[i][b] for b in BRANCHES},
            face_coverage=meta[i]["quality"][0],
            audio_clipped=meta[i]["quality"][1],
            av_duration_delta_sec=meta[i]["quality"][2],
        )
        for i in ids
    ]


def _auc_interval(items, *, seed: int, samples: int):
    """Identity-clustered ROC-AUC interval, or None when a slice has one class."""
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    if len({item.label for item in items}) < 2:
        return None

    def statistic(rows):
        labels = [r.label for r in rows]
        if len(set(labels)) < 2:
            raise ValueError("Bootstrap sample does not contain both classes")
        return float(roc_auc_score(labels, [r.probability for r in rows]))

    try:
        return cluster_bootstrap_interval(items, statistic, samples=samples, seed=seed)
    except ValueError:
        return None


def _row(items, interval, *, threshold: float = 0.5) -> dict:
    fakes = sum(1 for item in items if item.label == 1)
    # In-domain AUC is at the ceiling in nearly every slice here, so it cannot
    # show a gap even where one exists. Counts at the frozen threshold can: a
    # genuine clip called fake is the error this project cares about most.
    genuine = [item for item in items if item.label == 0]
    accused = sum(1 for item in genuine if item.probability >= threshold)
    missed = sum(
        1 for item in items if item.label == 1 and item.probability < threshold
    )
    entry = {
        "clips": len(items),
        "fake": fakes,
        "real": len(items) - fakes,
        "identities": len({item.source_identity for item in items}),
        "threshold": threshold,
        "genuine_called_fake": accused,
        "false_alarm_rate": accused / len(genuine) if genuine else None,
        "fake_called_genuine": missed,
        "miss_rate": missed / fakes if fakes else None,
    }
    if interval is None:
        entry["roc_auc"] = None
        entry["note"] = "single class in slice, no ranking metric exists"
    else:
        entry["roc_auc"] = interval.estimate
        entry["roc_auc_lower"] = interval.lower
        entry["roc_auc_upper"] = interval.upper
        entry["contains_chance"] = interval.lower <= 0.5 <= interval.upper
    return entry


def _print_table(title: str, rows: dict[str, dict]) -> None:
    print(title)
    width = max(len(name) for name in rows)
    print(
        f"  {'slice':{width}}  {'fake':>6}  {'real':>5}  {'ids':>4}  "
        f"{'ROC-AUC':>8}  {'95% interval':>18}  {'accused':>9}  {'missed':>8}"
    )
    print("  " + "-" * (width + 70))
    for name, entry in rows.items():
        if entry["roc_auc"] is None:
            cells = f"{'n/a':>8}  {'single class':>18}"
        else:
            mark = "?" if entry["contains_chance"] else " "
            cells = (
                f"{entry['roc_auc']:8.4f}  "
                f"[{entry['roc_auc_lower']:.4f}, {entry['roc_auc_upper']:.4f}]{mark}"
            )
        accused = f"{entry['genuine_called_fake']:>4}/{entry['real']:<4}"
        missed = f"{entry['fake_called_genuine']:>3}/{entry['fake']:<4}"
        print(
            f"  {name:{width}}  {entry['fake']:6,}  {entry['real']:5,}  "
            f"{entry['identities']:4,}  {cells}  {accused:>9}  {missed:>8}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.fusion.late import LateFusion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    run = arguments.run_dir
    bootstrap = {"seed": arguments.seed, "samples": arguments.bootstrap_samples}

    train_ids, train_logits, train_meta = _assemble(run / "features" / "oof.parquet")
    train_labels = [train_meta[i]["label"] for i in train_ids]
    test_ids, test_logits, test_meta = _assemble(run / "features" / "test.parquet")

    print(f"out-of-fold clips: {len(train_ids):,}")
    print(
        f"in-domain test clips: {len(test_ids):,} "
        f"(fake {sum(test_meta[i]['label'] for i in test_ids)}, "
        f"real {sum(1 for i in test_ids if not test_meta[i]['label'])})\n"
    )

    full = LateFusion(branch_names=BRANCHES).fit(
        _samples(train_ids, train_logits, train_meta), train_labels
    )
    scores = dict(
        zip(
            test_ids,
            full.predict_proba(_samples(test_ids, test_logits, test_meta)),
            strict=True,
        )
    )

    def scored(ids):
        return [
            Scored(
                test_meta[i]["label"],
                float(scores[i]),
                test_meta[i]["source_identity"],
            )
            for i in ids
        ]

    reals = [i for i in test_ids if test_meta[i]["label"] == 0]
    methods = sorted(
        {test_meta[i]["method"] for i in test_ids if test_meta[i]["label"] == 1}
    )

    # Per-method: that method's fakes against the whole real pool, so the rows
    # differ only in which manipulation is on the positive side.
    per_method = {}
    for method in methods:
        fakes = [
            i
            for i in test_ids
            if test_meta[i]["label"] == 1 and test_meta[i]["method"] == method
        ]
        items = scored(fakes + reals)
        per_method[method] = _row(items, _auc_interval(items, **bootstrap))
    _print_table("per-method, shipped fusion head on in-domain test:", per_method)

    # Head-only method holdout. See the module docstring: the branches saw every
    # method, so this measures the head's weights and nothing wider.
    holdout = {}
    for method in methods:
        kept = [i for i in train_ids if train_meta[i]["method"] != method]
        kept_labels = [train_meta[i]["label"] for i in kept]
        if len(set(kept_labels)) < 2:
            continue
        head = LateFusion(branch_names=BRANCHES).fit(
            _samples(kept, train_logits, train_meta), kept_labels
        )
        fakes = [
            i
            for i in test_ids
            if test_meta[i]["label"] == 1 and test_meta[i]["method"] == method
        ]
        subset = fakes + reals
        held = dict(
            zip(
                subset,
                head.predict_proba(_samples(subset, test_logits, test_meta)),
                strict=True,
            )
        )
        items = [
            Scored(
                test_meta[i]["label"],
                float(held[i]),
                test_meta[i]["source_identity"],
            )
            for i in subset
        ]
        entry = _row(items, _auc_interval(items, **bootstrap))
        entry["out_of_fold_clips_used"] = len(kept)
        entry["roc_auc_with_method_in_fit"] = per_method[method]["roc_auc"]
        if entry["roc_auc"] is not None and entry["roc_auc_with_method_in_fit"]:
            entry["delta"] = entry["roc_auc"] - entry["roc_auc_with_method_in_fit"]
        holdout[method] = entry
    _print_table(
        "head-only method holdout (branches still saw every method):", holdout
    )
    print("  delta against the same method with it in the fit:")
    for method, entry in holdout.items():
        if "delta" in entry:
            print(f"    {method:20} {entry['delta']:+.4f}")
    print()

    # Subgroups, in-domain only: DFDC annotates neither race nor gender.
    subgroups = {}
    for field in ("race", "gender"):
        rows = {}
        for value in sorted({test_meta[i][field] for i in test_ids}):
            ids = [i for i in test_ids if test_meta[i][field] == value]
            items = scored(ids)
            rows[value] = _row(items, _auc_interval(items, **bootstrap))
        subgroups[field] = rows
        _print_table(f"{field}, shipped fusion head on in-domain test:", rows)

    resolved = [
        entry["roc_auc"] for entry in per_method.values() if entry["roc_auc"]
    ]
    if resolved:
        print(
            f"per-method spread: {min(resolved):.4f} to {max(resolved):.4f}, "
            f"range {max(resolved) - min(resolved):.4f}"
        )
    print("? interval contains 0.5\n")

    payload = {
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_seed": arguments.seed,
        "bootstrap_unit": "source_identity",
        "partition": "FakeAVCeleb in-domain test",
        "out_of_fold_clips": len(train_ids),
        "test_clips": len(test_ids),
        "real_clips": len(reals),
        "per_method": per_method,
        "head_only_method_holdout": holdout,
        "holdout_caveat": (
            "The branch checkpoints were trained on every method. This measures "
            "whether the fusion head's weights are method-specific, not whether "
            "the system transfers to an unseen manipulation."
        ),
        "subgroups": subgroups,
    }
    out = arguments.output or run / "evaluation" / "generalization-slices.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
