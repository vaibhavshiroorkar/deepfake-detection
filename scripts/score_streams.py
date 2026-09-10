"""Score every trained Design B stream, and leave behind what fusion needs.

Two jobs in one pass, because they read the same forward passes.

The first is the number that actually answers "is this stream any good".
Training records loss only, and a validation loss of 0.27 says nothing about
ranking: the visual branch reached 0.9742 ROC-AUC in-domain and 0.7583 on DFDC
from loss curves that looked the same. Loss is what training optimises; AUC is
what the objective is stated in.

The second is the feature store `ddf train fusion --model deep` reads. Exporting
the embeddings is the same forward pass as scoring the logits, so doing them
separately would double the GPU time for nothing.

Partitions are the held-out in-domain test set and DFDC, which is the only
audio-bearing corpus here not built on VoxCeleb2. They disagree, and the
disagreement is the finding rather than a nuisance.

    uv run python scripts/score_streams.py --run-dir runs/design-b-20260910
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Which cached views each stream reads, and so which trainer built it. Keyed by
# the checkpoint stem the training script writes.
STREAM_KINDS = {
    "visual-dinov3": ("visual", "dinov3"),
    "visual-efficientnet": ("visual", "efficientnet"),
    "visual-xception": ("visual", "xception"),
    "stream-lipsync": ("lipsync", None),
    "stream-emotion": ("emotion", None),
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(name: str, history: dict):
    """Rebuild a stream exactly as it was trained, from its own history file.

    Reading the architecture back from the history rather than passing flags is
    deliberate: a stream rebuilt with the wrong temporal model still loads most
    of its tensors and then computes a different vector, which is the failure
    the dashboard's checkpoint picker had to be taught to report.
    """
    kind, backbone = STREAM_KINDS[name]
    model_config = history["config"]["model"]

    if kind == "visual":
        from deepfake_detection.streams.config import (
            dinov3_config,
            efficientnet_config,
            xception_config,
        )
        from deepfake_detection.streams.visual_stream import build_visual_stream

        presets = {
            "dinov3": dinov3_config,
            "efficientnet": efficientnet_config,
            "xception": xception_config,
        }
        config = presets[backbone](
            pretrained=False,
            common_dim=model_config["common_dim"],
            temporal_type=model_config["temporal"],
            temporal_hidden=model_config["temporal_hidden"],
            freeze_backbone=model_config["frozen_backbone"],
            frame_chunk_size=8,
        )
        return build_visual_stream(config), kind

    from deepfake_detection.streams.cross_modal_stream import (
        build_emotion_stream,
        build_lipsync_stream,
    )

    builder = build_lipsync_stream if kind == "lipsync" else build_emotion_stream
    return (
        builder(
            pretrained=False,
            common_dim=model_config.get("common_dim", 256),
        ),
        kind,
    )


def load_streams(run_dir: Path, device: str):
    """Every trained checkpoint in the run, rebuilt and loaded."""
    from deepfake_detection.fusion.stream_export import StreamSpec
    from deepfake_detection.training.checkpoints import load_checkpoint

    specs = []
    for name in STREAM_KINDS:
        checkpoint = run_dir / "checkpoints" / f"{name}.pt"
        history_path = run_dir / "checkpoints" / f"{name}-history.json"
        if not checkpoint.is_file() or not history_path.is_file():
            print(f"  {name}: not trained yet, skipping")
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        model, kind = build_model(name, history)
        load_checkpoint(checkpoint, model=model)
        specs.append(
            StreamSpec(
                name=name,
                model=model.to(device).eval(),
                checkpoint_hash=_sha256(checkpoint),
                kind=kind,
            )
        )
        best = history["epochs"][history["best_epoch"] - 1]
        print(
            f"  {name}: epoch {history['best_epoch']} of {len(history['epochs'])}, "
            f"val loss {best['validation_loss']:.4f}"
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.fusion.store import FeatureStore
    from deepfake_detection.fusion.stream_export import export_stream_features
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path("runs/program-20260906"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    audit = json.loads(
        (arguments.source_run / "cache-audit.json").read_text(encoding="utf-8")
    )
    preprocessing_hash = audit["preprocessing_hash"]

    print("trained streams:")
    specs = load_streams(arguments.run_dir, arguments.device)
    if not specs:
        print("nothing trained yet")
        return 1

    cache_index = {}
    with (arguments.source_run / "cache-index.csv").open(encoding="utf-8") as handle:
        import csv

        for row in csv.DictReader(handle):
            cache_index[row["clip_id"]] = arguments.source_run / row["cache_path"]
    cache_store = CacheStore(arguments.source_run / "cache")

    partitions = {
        "in-domain": (arguments.source_run / "split" / "test-usable.csv", "FakeAVCeleb"),
        "dfdc": (arguments.source_run / "split" / "dfdc-visual.csv", "DFDC"),
    }

    summary: dict[str, dict[str, float | None]] = {}
    features = arguments.run_dir / "features"
    features.mkdir(parents=True, exist_ok=True)

    for partition, (manifest, dataset) in partitions.items():
        if not manifest.is_file():
            print(f"\n{partition}: no manifest at {manifest}, skipping")
            continue
        records = load_manifest(manifest, dataset=dataset).records
        store_path = features / f"{partition}.parquet"
        store = FeatureStore(store_path)
        print(f"\n{partition}: {len(records):,} clips -> {store_path}")
        report = export_stream_features(
            records=records,
            cache_index=cache_index,
            cache_store=cache_store,
            feature_store=store,
            streams=specs,
            split_hash=arguments.run_dir.name,
            preprocessing_hash=preprocessing_hash,
            partition_role="external" if partition == "dfdc" else "test",
            run_id=f"score-{partition}",
            device=arguments.device,
        )
        print(
            f"  {report.exported_rows:,} rows, "
            f"{report.unavailable_rows:,} abstained"
        )

        by_stream: dict[str, list[tuple[float, int]]] = {}
        for row in store.read():
            if row.available:
                by_stream.setdefault(row.branch, []).append((row.logit, row.label))
        for name, pairs in sorted(by_stream.items()):
            labels = np.array([label for _logit, label in pairs])
            scores = np.array([logit for logit, _label in pairs])
            # One class means a ranking metric is undefined, which MNW's
            # fake-only lab set is the standing example of. Say so rather than
            # inventing a number.
            auc = (
                float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
            )
            summary.setdefault(name, {})[partition] = auc
            shown = f"{auc:.4f}" if auc is not None else "n/a (one class)"
            print(f"  {name:24} {shown}   ({len(pairs):,} clips)")

    print("\nROC-AUC")
    columns = [p for p in partitions if any(p in row for row in summary.values())]
    header = f"{'stream':24} " + "  ".join(f"{p:>12}" for p in columns)
    print(header)
    print("-" * len(header))
    for name, scores in sorted(summary.items()):
        cells = "  ".join(
            f"{scores.get(p):12.4f}"
            if isinstance(scores.get(p), float)
            else f"{'n/a':>12}"
            for p in columns
        )
        print(f"{name:24} {cells}")

    output = arguments.run_dir / "evaluation" / "stream-scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
