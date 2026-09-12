"""Does freezing BatchNorm while fine-tuning cost cross-corpus transfer?

Design A's EfficientNet reads 0.7611 on DFDC and Design B's reads 0.5067,
through an evaluation path verified identical by scoring the Design A checkpoint
through the Design B code. Two changes could account for it, and they point the
same way:

  - ten epochs to a 0.9997 validation fit, against three epochs to 0.9742
  - BatchNorm frozen while fine-tuning, which was a fix for a stream that sat at
    chance, and which also removed an accidental domain adapter: BatchNorm
    statistics adapting to the data they see is what AdaBN does deliberately

One variable at a time, the same subset, the same seed, each arm scored on the
full DFDC set. The absolute numbers sit below a full-scale run because training
is a few thousand clips; what this measures is the difference between arms.

The subset is drawn here rather than passed in, from the run's own split
manifests with a fixed seed, so the comparison is reproducible from the
repository alone.

    uv run python scripts/batchnorm_sweep.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def _subset(source: Path, destination: Path, count: int, seed: int) -> Path:
    """A fixed random subset of a manifest, written beside the run.

    Written out rather than filtered in memory: the file is the record of what
    the arms were trained on, and without it a rerun a week later is a different
    experiment with the same name.
    """
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    random.Random(seed).shuffle(rows)  # noqa: S311 - sampling, not secrets
    chosen = rows[:count]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    return destination


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedBranchDataset,
        collate_branch_items,
    )
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.streams.config import efficientnet_config
    from deepfake_detection.streams.visual_stream import build_visual_stream
    from deepfake_detection.training.ranking import roc_auc
    from deepfake_detection.training.streams import (
        StreamTrainingConfig,
        fit_visual_stream,
        parameter_groups,
    )
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path("runs/program-20260906"))
    parser.add_argument("--train-clips", type=int, default=1400)
    parser.add_argument("--validation-clips", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    work = arguments.run_dir / "sweeps"
    train_manifest = _subset(
        arguments.source_run / "split" / "train-usable.csv",
        work / "batchnorm-train.csv",
        arguments.train_clips,
        arguments.seed,
    )
    validation_manifest = _subset(
        arguments.source_run / "split" / "val-usable.csv",
        work / "batchnorm-val.csv",
        arguments.validation_clips,
        arguments.seed,
    )

    index: dict[str, Path] = {}
    with (arguments.source_run / "cache-index.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index[row["clip_id"]] = arguments.source_run / row["cache_path"]
    store = CacheStore(arguments.source_run / "cache")

    def loader(manifest: Path, dataset: str, *, shuffle: bool = False):
        records = load_manifest(manifest, dataset=dataset).records
        data = CachedBranchDataset(
            records=records, cache_index=index, cache_store=store, branch="visual"
        )
        return DataLoader(
            data, batch_size=8, shuffle=shuffle, collate_fn=collate_branch_items
        )

    train_batches = loader(train_manifest, "FakeAVCeleb", shuffle=True)
    validation_batches = loader(validation_manifest, "FakeAVCeleb")
    dfdc_batches = loader(
        arguments.source_run / "split" / "dfdc-visual.csv", "DFDC"
    )
    print(
        f"train {len(train_batches.dataset):,}  "
        f"validation {len(validation_batches.dataset):,}  "
        f"dfdc {len(dfdc_batches.dataset):,}",
        flush=True,
    )

    def score(model) -> float:
        model.eval()
        logits, labels = [], []
        with torch.inference_mode():
            for batch in dfdc_batches:
                logit, _ = model(batch.values.to(arguments.device))
                logits.append(logit.cpu())
                labels.append(batch.labels)
        return roc_auc(torch.cat(logits), torch.cat(labels))

    def run(label: str, *, freeze_batchnorm: bool, epochs: int) -> dict:
        torch.manual_seed(arguments.seed)
        config = efficientnet_config(
            pretrained=True,
            common_dim=256,
            frame_chunk_size=8,
            grad_checkpointing=False,
            freeze_batchnorm_on_finetune=freeze_batchnorm,
        )
        model = build_visual_stream(config)
        optimizer = torch.optim.AdamW(
            parameter_groups(
                model, head_lr=1e-4, encoder_lr=5e-6, encoder_names=("backbone",)
            ),
            lr=1e-4,
            weight_decay=1e-4,
        )
        print(f"\n=== {label} ===", flush=True)
        history = fit_visual_stream(
            model=model,
            train_batches=train_batches,
            validation_batches=validation_batches,
            optimizer=optimizer,
            config=StreamTrainingConfig(
                epochs=epochs,
                accumulation_steps=2,
                freeze_epochs=1,
                early_stopping_patience=epochs,
            ),
            device=arguments.device,
        )
        validation = max(epoch.validation_auc for epoch in history.epochs)
        dfdc = score(model)
        print(f"{label}: validation {validation:.4f}   DFDC {dfdc:.4f}", flush=True)
        return {
            "arm": label,
            "freeze_batchnorm": freeze_batchnorm,
            "epochs": epochs,
            "validation_auc": validation,
            "dfdc_auc": dfdc,
        }

    results = [
        run("frozen BatchNorm, 10 epochs", freeze_batchnorm=True, epochs=10),
        run("live BatchNorm, 10 epochs", freeze_batchnorm=False, epochs=10),
        run("frozen BatchNorm, 3 epochs", freeze_batchnorm=True, epochs=3),
    ]

    print(f"\n{'arm':28} {'validation':>11} {'DFDC':>8}")
    print("-" * 50)
    for row in results:
        print(f"{row['arm']:28} {row['validation_auc']:11.4f} {row['dfdc_auc']:8.4f}")

    output = (
        arguments.output
        or arguments.run_dir / "evaluation" / "batchnorm-sweep.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "arms": results,
                "train_clips": len(train_batches.dataset),
                "validation_clips": len(validation_batches.dataset),
                "dfdc_clips": len(dfdc_batches.dataset),
                "seed": arguments.seed,
                "train_manifest": train_manifest.as_posix(),
                "validation_manifest": validation_manifest.as_posix(),
                "reference": {
                    "design_a_validation": 0.9742,
                    "design_a_dfdc": 0.7611,
                    "design_b_validation": 0.9997,
                    "design_b_dfdc": 0.5067,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
