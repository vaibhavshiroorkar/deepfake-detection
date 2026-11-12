"""Lip-sync scored with no training: do AV-HuBERT's lips and audio agree?

AV-HuBERT was pretrained with modality dropout, so a lips-only pass and an
audio-only pass land in one feature space. On genuine speech the two should
agree frame by frame. The score is their mean cosine similarity over the 2 s
sync window, negated so that a higher score means more likely fake.

Nothing is fitted, so there is no training set to overfit and no dataset
shortcut to learn. That makes it the cleanest test of whether lip-sync carries
signal across corpora (F26, F30 found the trained sync models did not).

Each partition is reported whole, with an identity-clustered interval, and
per manipulation type, because the types make opposite predictions. A cloned
voice over genuine video should disagree with the lips. Wav2Lip regenerates the
lips from the fake audio, so its two streams should agree, and a sync score
should miss it.

AV-HuBERT was pretrained on VoxCeleb2 and LRS3. FakeAVCeleb and LAV-DF are
built from VoxCeleb2, so their genuine speakers may be in its pretraining data.
It never saw a label, but DFDC is the only partition here it cannot have seen.

`lavdf-blind` cuts its window where it would for an unlabelled upload, so it
usually misses LAV-DF's short forgery. `lavdf-matched` places the window on the
forgery and scores it against an untouched window of the same file.

    uv run python scripts/avhubert_sync_score.py
    uv run python scripts/avhubert_sync_score.py --partitions lavdf-blind lavdf-matched \
        --output runs/lavdf-eval-20261031/evaluation/avhubert-sync-score.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROGRAM = Path("runs/program-20260906")
LAVDF = Path("runs/lavdf-eval-20261031")
# Name: manifest, dataset, cache index, cache root.
PARTITIONS = {
    "in-domain-test": (
        PROGRAM / "split" / "test-sync-usable.csv", "FakeAVCeleb",
        PROGRAM / "cache-index.csv", PROGRAM / "cache",
    ),
    "dfdc": (
        PROGRAM / "split" / "dfdc-sync.csv", "DFDC",
        PROGRAM / "cache-index.csv", PROGRAM / "cache",
    ),
    "lavdf-blind": (
        LAVDF / "blind.csv", "LAV-DF", LAVDF / "cache-index-blind.csv", LAVDF / "cache",
    ),
    "lavdf-matched": (
        LAVDF / "matched.csv", "LAV-DF", LAVDF / "cache-index-matched.csv", LAVDF / "cache",
    ),
}


def _scores(
    manifest: Path, dataset: str, cache_index: Path, cache_root: Path, device: str, batch: int
):
    import torch
    import torch.nn.functional as F

    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.paths import AVHUBERT_CHECKPOINT
    from deepfake_detection.streams.encoders import AVHubertPairEncoder
    from deepfake_detection.views.cache_store import CacheStore

    records = load_manifest(manifest, dataset=dataset).records
    index = read_cache_index(cache_index)
    store = CacheStore(cache_root)
    encoder = AVHubertPairEncoder(checkpoint=str(AVHUBERT_CHECKPOINT)).to(device)
    rows, videos, audios = [], [], []

    def flush():
        if not videos:
            return
        video = torch.from_numpy(np.stack(videos)).to(device)
        audio = torch.from_numpy(np.stack(audios)).to(device)
        lips, sound = encoder(video, audio)
        agreement = F.cosine_similarity(lips, sound, dim=-1).mean(dim=1)
        for row, value in zip(rows[-len(videos) :], agreement.tolist(), strict=True):
            row["score"] = -value
        videos.clear()
        audios.clear()

    for count, record in enumerate(records, start=1):
        path = index.get(record.clip_id)
        if path is None:
            continue
        prepared = store.load(path, views=("sync_video_view", "sync_audio_view"))
        if prepared.sync_video_view is None or prepared.sync_audio_view is None:
            continue
        rows.append(
            {
                "clip_id": record.clip_id,
                "label": int(record.clip_fake),
                "source": record.source,
                "kind": record.manipulation_type or "",
            }
        )
        videos.append(prepared.sync_video_view)
        audios.append(prepared.sync_audio_view)
        if len(videos) == batch:
            flush()
        if count % 500 == 0:
            print(f"  {count:,} of {len(records):,}", flush=True)
    flush()
    return rows


def _interval(rows, samples: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.evaluation.bootstrap import cluster_bootstrap_interval
    from deepfake_detection.evaluation.metrics import EvaluationItem

    items = [
        EvaluationItem(
            label=r["label"],
            probability=r["score"],
            source_identity=r["source"],
            method="",
        )
        for r in rows
    ]

    def auc(chosen):
        labels = [i.label for i in chosen]
        if len(set(labels)) < 2:
            raise ValueError("one class")
        return float(roc_auc_score(labels, [i.probability for i in chosen]))

    interval = cluster_bootstrap_interval(items, auc, samples=samples, seed=seed)
    return {
        "roc_auc": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "clips": len(items),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/encoders-20261022/evaluation/avhubert-sync-score.json"),
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=tuple(PARTITIONS),
        default=("in-domain-test", "dfdc"),
    )
    arguments = parser.parse_args(argv)

    report: dict[str, dict] = {}
    for name in arguments.partitions:
        manifest, dataset, cache_index, cache_root = PARTITIONS[name]
        print(f"{name}: {manifest}", flush=True)
        rows = _scores(
            manifest, dataset, cache_index, cache_root, arguments.device, arguments.batch
        )
        entry = _interval(rows, arguments.samples, arguments.seed)
        print(
            f"  {name:30} {entry['roc_auc']:.4f} "
            f"[{entry['lower']:.4f}, {entry['upper']:.4f}]  n={entry['clips']}"
        )
        # Each manipulation type against the genuine clips of the partition.
        genuine = [r for r in rows if r["label"] == 0]
        kinds = {}
        for kind in sorted({r["kind"] for r in rows if r["label"] == 1 and r["kind"]}):
            subset = genuine + [
                r for r in rows if r["kind"] == kind and r["label"] == 1
            ]
            kinds[kind] = _interval(subset, arguments.samples, arguments.seed)
            k = kinds[kind]
            print(
                f"    {kind:28} {k['roc_auc']:.4f} "
                f"[{k['lower']:.4f}, {k['upper']:.4f}]  n={k['clips']}"
            )
        entry["by_kind"] = kinds
        report[name] = entry
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
