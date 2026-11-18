"""P-001 amendment A11 (exploratory): each upgraded path against its P-001 counterpart.

CLIP against the P-001 visual path, XLS-R against the P-001 audio path, each on its own label,
on the identity-strict FakeAVCeleb test and each locked target. Clips are paired by `clip_id`,
and the difference AUC(new) - AUC(P-001) is averaged over the seeds that finished, with the
A8 identity bootstrap shared by the seeds. Writes `{output}/compare.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from deepfake_detection.fusion.h2 import SeedPair, mean_seed_difference

SEEDS = (17, 29, 43)
PARTS = ("test-identity-strict", "dfdc", "lavdf-test", "hifiavdf")
MODELS = (("clip", "visual"), ("xlsr", "audio"))


def _read(path: Path) -> dict[str, tuple[int, float, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            r["clip_id"]: (int(r["label"]), float(r["probability"]), r["source"])
            for r in csv.DictReader(handle)
        }


def compare_part(
    new_dir: Path, old_dir: Path, new: str, old: str, part: str, *, samples: int = 1000
) -> dict:
    pairs, seeds = [], []
    for seed in SEEDS:
        new_file = new_dir / f"{new}-seed{seed}-{part}-predictions.csv"
        old_file = old_dir / f"{old}-seed{seed}-{part}-predictions.csv"
        if not (new_file.exists() and old_file.exists()):
            continue
        new_rows, old_rows = _read(new_file), _read(old_file)
        clips = sorted(set(new_rows) & set(old_rows))
        labels = np.array([new_rows[c][0] for c in clips])
        if any(new_rows[c][0] != old_rows[c][0] for c in clips):
            raise ValueError(f"Labels differ between {new_file} and {old_file}")
        if labels.min() == labels.max():
            return {"skipped": "one class in the set"}
        groups = np.array([new_rows[c][2] for c in clips])
        pairs.append(
            SeedPair(
                labels,
                groups,
                np.array([new_rows[c][1] for c in clips]),
                np.array([old_rows[c][1] for c in clips]),
            )
        )
        seeds.append(seed)
    if not pairs:
        return {"skipped": "no finished seed"}
    return {
        "seeds": seeds,
        "clips": int(len(pairs[0].labels)),
        "auc_new": [float(roc_auc_score(p.labels, p.fusion)) for p in pairs],
        "auc_old": [float(roc_auc_score(p.labels, p.visual)) for p in pairs],
        "difference": mean_seed_difference(pairs, samples=samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a11-dir", type=Path, required=True)
    parser.add_argument("--p001-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"amendment": "A11", "exploratory": True, "results": {}}
    for new, old in MODELS:
        for part in PARTS:
            result = compare_part(args.a11_dir, args.p001_dir, new, old, part)
            report["results"][f"{new}-vs-{old}.{part}"] = result
            if "difference" in result:
                d = result["difference"]
                print(
                    f"{new} vs {old} {part}: seeds {result['seeds']} new {np.mean(result['auc_new']):.3f} "
                    f"old {np.mean(result['auc_old']):.3f} diff {d['estimate']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]"
                )
            else:
                print(f"{new} vs {old} {part}: {result['skipped']}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compare.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
