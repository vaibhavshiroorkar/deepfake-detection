"""P-001 amendment A17 (exploratory): FakeAVCeleb speakers against AV-HuBERT's pretraining speakers.

    .venv/Scripts/python.exe scripts/run_a17.py --vox2-dir DOWNLOADS

DOWNLOADS holds train.csv and test.csv of the public `yangwang825/vox2-veri-3s` dataset (VoxCeleb2's
verification split; the official metadata needs a login). The first run writes
results/p001-a17/vox2-speakers.csv (speaker id, split) with the source files' hashes; later runs
read that file. AV-HuBERT Base pretrained on LRS3 and the English part of VoxCeleb2 dev (assumed,
from its paper), so a VoxCeleb2 test speaker was never in pretraining. Writes results/p001-a17/a17.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "p001-a17"
SPEAKERS = OUTPUT / "vox2-speakers.csv"
SOURCE = "https://huggingface.co/datasets/yangwang825/vox2-veri-3s"
SEEDS = (17, 29, 43)
MIN_IDENTITIES = 5


def speakers_from(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    split, hashes = {}, {}
    for name, label in (("train.csv", "dev"), ("test.csv", "test")):
        path = directory / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                split[row["spk_id"].strip()] = label
    return split, hashes


def group_of(identity: str, split: dict[str, str]) -> str:
    return split.get(identity, "neither")


class _Row:
    def __init__(self, source: str, label: int, probability: float) -> None:
        self.source_identity, self.label, self.probability = source, label, probability


def _auc(rows) -> float:
    return float(roc_auc_score([r.label for r in rows], [r.probability for r in rows]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vox2-dir", type=Path)
    arguments = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    if arguments.vox2_dir is not None:
        split, hashes = speakers_from(arguments.vox2_dir)
        with open(SPEAKERS, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["speaker_id", "split"])
            writer.writerows(sorted(split.items()))
        meta = {"source": SOURCE, "sha256": hashes}
    with open(SPEAKERS, encoding="utf-8", newline="") as f:
        split = {r["speaker_id"]: r["split"] for r in csv.DictReader(f)}

    with open(ROOT / "data/manifests/fakeavceleb.csv", encoding="utf-8", newline="") as f:
        sources = sorted({r["source"] for r in csv.DictReader(f)})
    counts = {g: sum(group_of(s, split) == g for s in sources) for g in ("dev", "test", "neither")}

    report = {"amendment": "A17", "exploratory": True, "vox2": meta or "from vox2-speakers.csv",
              "vox2_speakers": {g: sum(v == g for v in split.values()) for g in ("dev", "test")},
              "fakeavceleb_sources": counts, "fakeavceleb_test_partition": {}, "lavdf": "unknown: no speaker ids"}
    tables = []
    for seed in SEEDS:
        with open(ROOT / f"results/p001-sync/sync-test-seed{seed}-predictions.csv", encoding="utf-8", newline="") as f:
            tables.append(list(csv.DictReader(f)))
    test_rows = tables[0]
    for g in ("dev", "test", "neither"):
        members = [r for r in test_rows if group_of(r["source"], split) == g]
        real_ids = {r["source"] for r in members if r["label"] == "0"}
        report["fakeavceleb_test_partition"][g] = {"clips": len(members), "identities": len({r["source"] for r in members}),
                                                    "identities_with_real_clips": len(real_ids)}
    enough = all(report["fakeavceleb_test_partition"][g]["identities_with_real_clips"] >= MIN_IDENTITIES
                 for g in ("dev", "test"))
    report["auc_by_group"] = "not reported: fewer than 5 identities with real clips in dev or test"
    if enough:
        report["auc_by_group"] = {}
        for g in ("dev", "test"):
            per_seed = {}
            for seed, table in zip(SEEDS, tables, strict=True):
                rows = [_Row(r["source"], int(r["label"]), float(r["probability"])) for r in table
                        if group_of(r["source"], split) == g and r["probability"] not in ("", "nan")]
                interval = cluster_bootstrap_interval(rows, _auc, samples=1000, seed=17)
                per_seed[str(seed)] = {"auc": interval.estimate, "lower": interval.lower, "upper": interval.upper}
            report["auc_by_group"][g] = per_seed | {"mean": float(np.mean([v["auc"] for v in per_seed.values()]))}
    (OUTPUT / "a17.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("vox2_speakers", "fakeavceleb_sources", "fakeavceleb_test_partition",
                                              "auc_by_group")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
