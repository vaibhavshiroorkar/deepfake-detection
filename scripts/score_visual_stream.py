"""Score a trained visual-stream checkpoint on cached clips (P-001 amendment A11).

    .venv/Scripts/python.exe scripts/score_visual_stream.py --checkpoint runs/p001/checkpoints/clip-seed17.pt \
        --manifest runs/p001/usable/test-identity-strict-visual.csv --cache-index runs/p001/cache-index.csv \
        --cache-root runs/p001/cache --dataset FakeAVCeleb --output runs/p001/evaluation/clip-seed17-test-identity-strict-predictions.csv

`ddf evaluate branch` scores only the EfficientNet branch. This rebuilds a visual stream from
its own history file (next to the checkpoint) and writes predictions in the same format, so
`ddf threshold` and `ddf evaluate predictions` handle it like any branch. The label is the
video label, the one the stream was trained on.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepfake_detection.data.cache_index import read_cache_index
from deepfake_detection.data.datasets import CachedBranchDataset, collate_branch_items
from deepfake_detection.data.manifest import load_manifest
from deepfake_detection.streams.config import StreamConfig
from deepfake_detection.streams.visual_stream import build_visual_stream
from deepfake_detection.training.checkpoints import load_checkpoint
from deepfake_detection.views.cache_store import CacheStore


def stream_from_history(history: dict):
    """Rebuild the stream exactly as trained; weights come from the checkpoint, not a download."""
    model = history["config"]["model"]
    config = StreamConfig(
        stream_name=history["config"].get("stream", "visual"),
        backbone_name=model["backbone"],
        global_pool=model["global_pool"],
        pretrained=False,
        temporal_type=model["temporal"],
        temporal_hidden=model["temporal_hidden"],
        common_dim=model["common_dim"],
        freeze_backbone=True,
        grad_checkpointing=False,
    )
    return build_visual_stream(config)


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preprocessing-hash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    history_path = args.checkpoint.with_name(args.checkpoint.stem + "-history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    model = stream_from_history(history)
    load_checkpoint(args.checkpoint, model=model)
    model.to(args.device).eval()

    records = load_manifest(args.manifest, dataset=args.dataset).records
    by_clip = {r.clip_id: r for r in records}
    dataset = CachedBranchDataset(records=records, cache_index=read_cache_index(args.cache_index),
                                  cache_store=CacheStore(args.cache_root), branch="visual",
                                  preprocessing_hash=args.preprocessing_hash)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_branch_items)

    rows = []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            logits, _ = model(batch.values.to(args.device))
        for clip_id, label, probability in zip(batch.clip_ids, batch.labels.int().tolist(),
                                               torch.sigmoid(logits.float()).cpu().tolist(), strict=True):
            record = by_clip[clip_id]
            rows.append({"clip_id": clip_id, "label": label, "probability": probability,
                         "source": record.source, "manipulation_type": record.manipulation_type,
                         "method": record.method})
    if not rows:
        raise SystemExit("no clip was scored")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored {len(rows)} clips -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
