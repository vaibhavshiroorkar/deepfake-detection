"""Does a lip-sync stream trained on matched pairs learn correspondence?

Both cross-modal streams trained on FakeAVCeleb sit on chance `diagonal_mass`
for every epoch of every run, so the cue they are named for is not operating.
Two explanations fit that: the architecture cannot learn correspondence, or
FakeAVCeleb does not require it. They call for different work, and this
separates them.

LAV-DF is the test case because its adapter cuts two windows out of the same
file: one containing the manipulated span, one avoiding every manipulated span.
Same speaker, same lighting, same microphone, same codec. A model cannot
separate them by recognising the generator, so the correspondence cue is the one
thing left to read. If the architecture can learn it at all, it should learn it
here.

The stream, its cache and its split all come from `runs/streams-20260905`, which
used a different preprocessing hash from the main program run. That is why this
scores only in-domain: the model and the cache must agree, and no FakeAVCeleb or
DFDC clip is cached under that hash.

    uv run python scripts/score_lavdf_stream.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    from deepfake_detection.data.datasets import (
        CachedAVPairDataset,
        collate_av_pair_items,
    )
    from deepfake_detection.data.manifest import load_manifest
    from deepfake_detection.streams.cross_attention import diagonal_mass
    from deepfake_detection.streams.cross_modal_stream import build_lipsync_stream
    from deepfake_detection.training.checkpoints import load_checkpoint
    from deepfake_detection.training.ranking import roc_auc
    from deepfake_detection.views.cache_store import CacheStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/streams-20260905"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/streams-20260905/checkpoints/lavdf-lipsync-stream-seed17.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/streams-20260905/lavdf-val-usable.csv"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    history_path = arguments.checkpoint.with_name(
        arguments.checkpoint.stem + "-history.json"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    model = build_lipsync_stream(
        pretrained=False,
        common_dim=history["config"]["model"].get("common_dim", 256),
    )
    load_checkpoint(arguments.checkpoint, model=model)
    model = model.to(arguments.device).eval()

    index: dict[str, Path] = {}
    with (arguments.run_dir / "cache-index.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index[row["clip_id"]] = arguments.run_dir / row["cache_path"]
    store = CacheStore(arguments.run_dir / "cache")

    records = load_manifest(arguments.manifest, dataset="LAV-DF").records
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )
    batches = DataLoader(
        dataset, batch_size=arguments.batch_size, collate_fn=collate_av_pair_items
    )
    print(f"{len(dataset):,} clips from {arguments.manifest.name}", flush=True)

    logits, labels, masses = [], [], []
    with torch.inference_mode():
        for batch in batches:
            output = model(
                video=batch.video.to(arguments.device),
                audio=batch.audio.to(arguments.device),
            )
            logits.append(output.logit.cpu())
            labels.append(batch.labels)
            if getattr(output, "attention", None) is not None:
                masses.append(diagonal_mass(output.attention).cpu())

    scores = torch.cat(logits)
    truth = torch.cat(labels)
    auc = roc_auc(scores, truth)
    result = {
        "clips": len(dataset),
        "roc_auc": auc,
        "positive_rate": float(truth.mean()),
        "best_epoch": history["best_epoch"],
        "preprocessing_hash": history["metadata"]["preprocessing_hash"],
    }
    print(f"ROC-AUC {auc:.4f}   fake rate {float(truth.mean()):.3f}")

    if masses:
        mass = torch.cat(masses)
        queries = 50
        chance = (2 * 1 + 1) / queries
        result["diagonal_mass"] = float(mass.mean())
        result["diagonal_mass_chance"] = chance
        print(
            f"diagonal mass {float(mass.mean()):.4f} against a chance of "
            f"{chance:.4f} at {queries} query steps"
        )
        print(
            "  above chance" if float(mass.mean()) > chance * 1.05 else "  at chance"
        )

    output = arguments.output or arguments.run_dir / "lavdf-lipsync-score.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
