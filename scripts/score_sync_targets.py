"""Score a locked target with the frozen sync-path models.

Nothing is trained or chosen here. The three checkpoints from
`scripts/train_sync_path.py` (seeds 17, 29 and 43) score every clip of a target
manifest whose AV-HuBERT features have been cached. Clips with no usable window
are written blank: abstentions, counted in coverage, never dropped.

    uv run python scripts/score_sync_targets.py --checkpoints runs/p001/sync \
        --features runs/avhubert-hifiavdf-neutral --manifest data/manifests/hifiavdf-neutral.csv \
        --name hifiavdf --output runs/p001/sync-targets
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def _trainer():
    spec = importlib.util.spec_from_file_location("train_sync_path", Path(__file__).with_name("train_sync_path.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sync_path"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[17, 29, 43])
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    from deepfake_detection.avsync.heads import AlignmentHead

    trainer = _trainer()
    clips = trainer.load_split(arguments.manifest)
    # Older manifests (DFDC) carry only manipulation_type.
    for row in clips.values():
        kind = row.get("manipulation_type", "")
        row.setdefault("video_fake", str(int(kind.startswith("FakeVideo-"))))
        row.setdefault("audio_fake", str(int(kind.endswith("-FakeAudio"))))
        row.setdefault("clip_fake", str(int(row["video_fake"] == "1" or row["audio_fake"] == "1")))
    owners, lips, audio = trainer.load_windows(arguments.features, clips)
    print(f"{arguments.name}: {len(clips)} clips, {len(owners)} usable windows", flush=True)
    for seed in arguments.seeds:
        state = torch.load(arguments.checkpoints / f"sync-seed{seed}.pt", map_location="cpu", weights_only=True)
        head = AlignmentHead(width=lips.shape[-1]).to(arguments.device)
        head.load_state_dict(state["state_dict"])
        head.eval()
        scores = trainer.score_clips(head, owners, lips, audio, arguments.device)
        path = arguments.output / f"sync-{arguments.name}-seed{seed}-predictions.csv"
        scored = trainer.write_predictions(path, clips, scores)
        print(f"  seed {seed}: scored {scored} of {len(clips)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
