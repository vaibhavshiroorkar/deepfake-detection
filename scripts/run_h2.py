"""Run H2 of P-001 (amendment A8): fusion of visual, audio and sync against visual-only.

    .venv/Scripts/python.exe scripts/run_h2.py --output results/p001-h2

Reads, per seed s and part p (val, test, test-identity-strict, or a target):
  visual and audio: runs/p001/evaluation/{branch}-seed{s}-{p}-predictions.csv
  sync, FakeAVCeleb: results/p001-sync/sync-{p}-seed{s}-predictions.csv
  sync, targets:     results/p001-sync-targets/sync-{p}-seed{s}-predictions.csv
Labels (clip label) and identities come from the sync files. A clip counts only
if all three paths scored it. Writes h2.json, fused predictions and the weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detection.fusion.h2 import (  # noqa: E402
    BRANCHES,
    SeedPair,
    cross_fit_scores,
    fit_fusion,
    holm,
    logit,
    mean_seed_difference,
)

SEEDS = (17, 29, 43)
H2_TARGETS = ("test-identity-strict", "dfdc", "lavdf-window")    # the Holm family (A8; LAV-DF by window label, A9)
REPORTED = ("test", "hifiavdf", "lavdf-test")                     # same statistic, outside the rule
# The window label (A9) changes labels only; the visual and audio scores are those of the same clips.
BRANCH_PART = {"lavdf-window": "lavdf-test"}


def read(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle) if row.get("probability", "").strip()}


def load(part: str, seed: int, args) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
    sync_dir = args.sync_dir if part in ("val", "test", "test-identity-strict") else args.sync_targets
    sync = read(sync_dir / f"sync-{part}-seed{seed}-predictions.csv")
    branch_part = BRANCH_PART.get(part, part)
    visual = read(args.branch_dir / f"visual-seed{seed}-{branch_part}-predictions.csv")
    audio = read(args.branch_dir / f"audio-seed{seed}-{branch_part}-predictions.csv")
    clips = sorted(set(sync) & set(visual) & set(audio))
    x = np.column_stack([logit([float(src[c]["probability"]) for c in clips]) for src in (visual, audio, sync)])
    labels = np.array([int(sync[c]["label"]) for c in clips])
    groups = np.array([sync[c]["source"] for c in clips])
    return clips, labels, groups, x, len(sync)


def balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best, best_t = -1.0, 0.5
    for t in np.unique(scores):
        flagged = scores >= t
        ba = 0.5 * (flagged[labels == 1].mean() + (~flagged[labels == 0]).mean())
        if ba > best:
            best, best_t = ba, float(t)
    return best_t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=ROOT / "runs/p001/evaluation")
    parser.add_argument("--sync-dir", type=Path, default=ROOT / "results/p001-sync")
    parser.add_argument("--sync-targets", type=Path, default=ROOT / "results/p001-sync-targets")
    parser.add_argument("--output", type=Path, default=ROOT / "results/p001-h2")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    fusions, weights = {}, {}
    for seed in SEEDS:
        clips, labels, groups, x, _ = load("val", seed, args)
        oof = cross_fit_scores(x, labels, groups)
        fusions[seed] = fit_fusion(x, labels)
        weights[seed] = {
            "branches": list(BRANCHES),
            "coef": fusions[seed].model.coef_[0].tolist(),
            "intercept": float(fusions[seed].model.intercept_[0]),
            "mean": fusions[seed].mean.tolist(), "scale": fusions[seed].scale.tolist(),
            "threshold_from_oof_validation": balanced_threshold(labels, oof),
            "validation_clips": len(clips),
        }

    report = {"amendment": "A8", "seeds": list(SEEDS), "weights": weights, "targets": {}}
    for part in H2_TARGETS + REPORTED:
        loaded = {seed: load(part, seed, args) for seed in SEEDS}
        # The shared bootstrap needs every seed on the same clips, so keep the clips all seeds scored.
        common = sorted(set.intersection(*(set(loaded[s][0]) for s in SEEDS)))
        if not common:
            raise SystemExit(f"{part}: no clip is scored by all three paths in every seed")
        pairs, coverage = [], []
        for seed in SEEDS:
            clips, labels, groups, x, n_sync = loaded[seed]
            position = {c: i for i, c in enumerate(clips)}
            keep = np.array([position[c] for c in common])
            labels, groups, x = labels[keep], groups[keep], x[keep]
            fused = fusions[seed].score(x)
            pairs.append(SeedPair(labels, groups, fused, x[:, 0]))
            coverage.append({"seed": seed, "scored_by_all": len(clips), "scored_by_sync": n_sync})
            with (args.output / f"fusion-{part}-seed{seed}-predictions.csv").open("w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["clip_id", "label", "source", "fusion", "visual_logit", "audio_logit", "sync_logit"])
                for i, c in enumerate(common):
                    w.writerow([c, labels[i], groups[i], f"{fused[i]:.6f}", *(f"{v:.6f}" for v in x[i])])
        result = mean_seed_difference(pairs, samples=args.samples)
        result["coverage"] = coverage
        result["clips_scored_by_every_seed"] = len(common)
        report["targets"][part] = result

    family = {t: report["targets"][t]["p_one_sided"] for t in H2_TARGETS}
    rejected = holm(family)
    report["holm"] = {t: {"p_one_sided": family[t], "rejected_at_0.05": rejected[t]} for t in H2_TARGETS}
    report["h2_success"] = sum(rejected.values()) >= 2
    (args.output / "h2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for t in H2_TARGETS + REPORTED:
        r = report["targets"][t]
        print(f"{t:22} diff {r['estimate']:+.4f} [{r['lower']:+.4f}, {r['upper']:+.4f}] p={r['p_one_sided']:.3f} "
              f"clips={r['clips']}")
    print("H2 holds" if report["h2_success"] else "H2 does not hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
