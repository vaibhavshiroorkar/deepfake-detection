"""Score both BatchNorm arms of both visual streams, on both partitions.

The 1,400-clip sweep in `scripts/batchnorm_sweep.py` was a cheap proxy for this.
This is the full-scale measurement: the checkpoints trained by
`scripts/train_design_b.ps1` with and without `--live-batchnorm`, kept side by
side under `checkpoints-frozen-bn/` and `checkpoints-live-bn/`.

Labels are `video_fake` here, because that is what the visual stream was trained
on. `scripts/score_streams.py` scores the same checkpoints against `clip_fake`,
which the fusion store carries, and on FakeAVCeleb those differ: a clip with a
real video track and spoofed audio is `clip_fake` and not `video_fake`. So the
same checkpoint has two legitimate in-domain numbers, 0.9987 against its own
objective and 0.9719 against the clip label. They coincide on DFDC, where every
manipulated clip has a manipulated video track, which is why the DFDC column
here matches `stream-scores.json` exactly.

DINOv3 is the control. It is a frozen ViT with no running statistics, so the two
arms must agree to the last decimal, and they do.
"""

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

from deepfake_detection.data.datasets import (  # noqa: E402
    CachedBranchDataset,
    collate_branch_items,
)
from deepfake_detection.data.manifest import load_manifest  # noqa: E402
from deepfake_detection.fusion.stream_loading import build_model  # noqa: E402
from deepfake_detection.training.checkpoints import load_checkpoint  # noqa: E402
from deepfake_detection.training.ranking import roc_auc  # noqa: E402
from deepfake_detection.views.cache_store import CacheStore  # noqa: E402

SOURCE = ROOT / "runs" / "program-20260906"
RUN = ROOT / "runs" / "design-b-20260910"

index = {}
with (SOURCE / "cache-index.csv").open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        index[row["clip_id"]] = SOURCE / row["cache_path"]
store = CacheStore(SOURCE / "cache")


def loader(manifest: Path, dataset: str):
    records = load_manifest(manifest, dataset=dataset).records
    data = CachedBranchDataset(
        records=records, cache_index=index, cache_store=store, branch="visual"
    )
    return DataLoader(data, batch_size=8, collate_fn=collate_branch_items)


partitions = {
    "in-domain": loader(SOURCE / "split" / "test-usable.csv", "FakeAVCeleb"),
    "dfdc": loader(SOURCE / "split" / "dfdc-visual.csv", "DFDC"),
}
for name, batches in partitions.items():
    print(f"{name}: {len(batches.dataset):,} clips", flush=True)


def score(model, batches) -> float:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in batches:
            logit, _ = model(batch.values.to("cuda"))
            logits.append(logit.cpu())
            labels.append(batch.labels)
    return roc_auc(torch.cat(logits), torch.cat(labels))


rows = []
for arm, directory in (
    ("frozen", RUN / "checkpoints-frozen-bn"),
    ("live", RUN / "checkpoints-live-bn"),
):
    for stream in ("visual-efficientnet", "visual-dinov3"):
        checkpoint = directory / f"{stream}.pt"
        history = json.loads(
            (directory / f"{stream}-history.json").read_text(encoding="utf-8")
        )
        model, _ = build_model(stream, history)
        load_checkpoint(checkpoint, model=model)
        model = model.to("cuda").eval()
        best = history["epochs"][history["best_epoch"] - 1]["validation_auc"]
        row = {
            "arm": arm,
            "stream": stream,
            "validation": best,
            "best_epoch": history["best_epoch"],
            "epochs": len(history["epochs"]),
        }
        for name, batches in partitions.items():
            row[name] = score(model, batches)
        rows.append(row)
        print(
            f"{arm:7} {stream:22} val {best:.4f}  "
            f"in-domain {row['in-domain']:.4f}  dfdc {row['dfdc']:.4f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

out = RUN / "evaluation" / "batchnorm-arms.json"
out.write_text(json.dumps({"arms": rows}, indent=2, sort_keys=True) + "\n", "utf-8")
print(f"\nwrote {out}")
