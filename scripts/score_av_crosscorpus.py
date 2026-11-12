"""Every branch and the fused system on every corpus that carries audio.

F10 compares corpora for the visual stream alone, because FF++ has no audio.
This is the audiovisual counterpart: the three Design A branches and the fused
head, each seed with its own head, on FakeAVCeleb test, DFDC and LAV-DF test.

LAV-DF is scored blind, one row per clip with no annotation used, which is how
DFDC is scored. It is built on VoxCeleb2 like FakeAVCeleb, so it tests a new
generator rather than new capture conditions. It is also the only one of the
three that says which modality each forgery touched, so its rows are broken down
by manipulation type: each fake type against the same genuine clips.

The matched store asks the sync branch one narrower question. Pointed at the
manipulated span, does it score that span above an untouched span of the same
file? `pairs_ranked` is the share of files where it does. The visual and audio
views of the two rows are identical, so only sync is reported there.

    uv run python scripts/score_av_crosscorpus.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

BRANCHES = ("visual", "audio", "sync")
SEEDS = (17, 29, 43)
PROGRAM = Path("runs/program-20260906")
LAVDF = Path("runs/lavdf-eval-20261031")
FAKE_KINDS = ("FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio")


def store_path(directory: Path, stem: str, seed: int) -> Path:
    # Seed 17 wrote the original stores without a suffix.
    return directory / "features" / (f"{stem}.parquet" if seed == 17 else f"{stem}-seed{seed}.parquet")


def head_path(seed: int) -> Path:
    name = "fusion-logistic" if seed == 17 else f"fusion-logistic-seed{seed}"
    return PROGRAM / "checkpoints" / f"{name}.joblib"


def assemble(path: Path, model) -> list[dict]:
    """One dict per clip with all three branch logits and the fused probability."""
    from deepfake_detection.fusion.late import FusionSample
    from deepfake_detection.fusion.store import FeatureStore

    logits: dict[str, dict[str, float]] = collections.defaultdict(dict)
    meta: dict[str, dict] = {}
    for row in FeatureStore(path).read():
        if not row.available:
            continue
        logits[row.clip_id][row.branch] = row.logit
        meta[row.clip_id] = {
            "label": row.label,
            "source": row.source_identity,
            "quality": (row.face_coverage, row.audio_clipped, row.av_duration_delta_sec),
        }
    ids = sorted(k for k, v in logits.items() if len(v) == len(BRANCHES))
    samples = [
        FusionSample(
            branch_logits={b: logits[i][b] for b in BRANCHES},
            face_coverage=meta[i]["quality"][0],
            audio_clipped=meta[i]["quality"][1],
            av_duration_delta_sec=meta[i]["quality"][2],
        )
        for i in ids
    ]
    fused = model.predict_proba(samples)
    return [
        {"clip_id": i, "label": meta[i]["label"], "source": meta[i]["source"],
         "fused": float(p), **{b: logits[i][b] for b in BRANCHES}}
        for i, p in zip(ids, fused, strict=True)
    ]


def auc(clips: list[dict], score: str, *, samples: int, seed: int) -> dict:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

    class Item:
        __slots__ = ("label", "value", "source_identity")

        def __init__(self, clip):
            self.label, self.value, self.source_identity = clip["label"], clip[score], clip["source"]

    def statistic(sample):
        labels = [item.label for item in sample]
        if len(set(labels)) < 2:
            raise ValueError("Bootstrap sample does not contain both classes")
        return float(roc_auc_score(labels, [item.value for item in sample]))

    interval = cluster_bootstrap_interval(
        [Item(clip) for clip in clips], statistic, samples=samples, seed=seed
    )
    return {"roc_auc": interval.estimate, "lower": interval.lower, "upper": interval.upper}


def summarise(per_seed: dict[int, dict]) -> dict:
    values = [entry["roc_auc"] for entry in per_seed.values()]
    return {
        "seeds": {str(seed): entry for seed, entry in per_seed.items()},
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=LAVDF / "evaluation" / "av-crosscorpus.json")
    arguments = parser.parse_args(argv)
    boot = {"samples": arguments.bootstrap_samples, "seed": arguments.seed}

    heads = {}
    for seed in SEEDS:
        artifact = joblib.load(head_path(seed))
        heads[seed] = getattr(artifact, "model", artifact)

    with (LAVDF / "blind.csv").open(newline="", encoding="utf-8") as handle:
        kind_of = {row["clip_id"]: row["manipulation_type"] for row in csv.DictReader(handle)}

    corpora = {
        "FakeAVCeleb test": (PROGRAM, "test"),
        "DFDC": (PROGRAM, "dfdc"),
        "LAV-DF test": (LAVDF, "lavdf"),
    }
    report: dict[str, dict] = {"corpora": {}, "lavdf_by_kind": {}, "matched_sync": {}}
    scored: dict[str, dict[int, list[dict]]] = {}
    for name, (directory, stem) in corpora.items():
        scored[name] = {seed: assemble(store_path(directory, stem, seed), heads[seed]) for seed in SEEDS}
        clips = scored[name][SEEDS[0]]
        report["corpora"][name] = {
            "clips": len(clips),
            "fake_share": sum(c["label"] for c in clips) / len(clips),
            **{
                score: summarise({s: auc(scored[name][s], score, **boot) for s in SEEDS})
                for score in (*BRANCHES, "fused")
            },
        }

    for kind in FAKE_KINDS:
        keep = {"RealVideo-RealAudio", kind}
        subsets = {
            seed: [c for c in clips if kind_of[c["clip_id"]] in keep]
            for seed, clips in scored["LAV-DF test"].items()
        }
        report["lavdf_by_kind"][kind] = {
            "clips": len(subsets[SEEDS[0]]),
            **{
                score: summarise({s: auc(subsets[s], score, **boot) for s in SEEDS})
                for score in (*BRANCHES, "fused")
            },
        }

    with (LAVDF / "matched.csv").open(newline="", encoding="utf-8") as handle:
        matched_rows = list(csv.DictReader(handle))
    file_of = {row["clip_id"]: row["video_path"] for row in matched_rows}
    kind_of_file = {
        row["video_path"]: row["manipulation_type"]
        for row in matched_rows
        if row["method"] == "lavdf-forged"
    }
    for seed in SEEDS:
        clips = assemble(store_path(LAVDF, "matched", seed), heads[seed])
        by_file: dict[str, dict[int, float]] = collections.defaultdict(dict)
        for clip in clips:
            by_file[file_of[clip["clip_id"]]][clip["label"]] = clip["sync"]
        pairs = {f: v for f, v in by_file.items() if len(v) == 2}
        entry = {"clips": len(clips), "pairs": len(pairs), **auc(clips, "sync", **boot)}
        entry["pairs_ranked"] = sum(v[1] > v[0] for v in pairs.values()) / len(pairs)
        for kind in FAKE_KINDS:
            chosen = [v for f, v in pairs.items() if kind_of_file[f] == kind]
            entry[f"pairs_ranked_{kind}"] = sum(v[1] > v[0] for v in chosen) / len(chosen)
        report["matched_sync"][str(seed)] = entry

    report["bootstrap"] = {"samples": arguments.bootstrap_samples, "unit": "source_identity"}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def cell(entry: dict) -> str:
        return f"{entry['mean']:.4f} ({entry['minimum']:.4f} to {entry['maximum']:.4f})"

    print("| System | " + " | ".join(corpora) + " |")
    print("| --- | " + " | ".join("---" for _ in corpora) + " |")
    for score in (*BRANCHES, "fused"):
        print(f"| {score} | " + " | ".join(cell(report["corpora"][n][score]) for n in corpora) + " |")
    print()
    print("| System | " + " | ".join(FAKE_KINDS) + " |")
    print("| --- | " + " | ".join("---" for _ in FAKE_KINDS) + " |")
    for score in (*BRANCHES, "fused"):
        print(f"| {score} | " + " | ".join(cell(report["lavdf_by_kind"][k][score]) for k in FAKE_KINDS) + " |")
    print()
    for seed, entry in report["matched_sync"].items():
        print(f"matched sync seed {seed}: {entry['roc_auc']:.4f} [{entry['lower']:.4f}, {entry['upper']:.4f}], "
              f"pairs ranked {entry['pairs_ranked']:.3f} over {entry['pairs']}")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
