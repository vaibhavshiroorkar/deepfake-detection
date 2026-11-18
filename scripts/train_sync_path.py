"""Train the sync path on real clips only, and score every split.

The sync path asks whether the lips and the voice belong together. It must not
learn what a particular generator's fakes look like, so it never sees a fake:

- Training pairs come from real clips only. Positives are lip and audio frames
  at the same instant. Negatives are the same lips with audio shifted 5 to 20
  frames (200 to 800 ms), and the same lips with another clip's audio.
- The epoch is chosen by the same loss on real validation clips. No fake label
  is read until scoring.

A window's score is its mean misfit over aligned frames. A clip's score is the
mean over its windows, mapped to [0, 1) by 1 - exp(-score), which keeps the
order. Ported from the team project's E-036 (I-8b).

    uv run python scripts/train_sync_path.py --features runs/avhubert-fakeavceleb-neutral \
        --split-dir runs/p001/split --output runs/p001/sync
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MIN_FACE = 0.8
WINDOW_FRAMES = 50  # 2 s at 25 fps
SEEDS = (17, 29, 43)


def load_split(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def load_windows(
    features: Path, clips: dict[str, dict]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    from deepfake_detection.avsync.features import cache_path

    owners, lips, audio = [], [], []
    short = 0
    with (features / "index.csv").open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            if (
                entry["clip_id"] not in clips
                or entry["status"] != "ok"
                or float(entry["face"]) < MIN_FACE
            ):
                continue
            stored = np.load(cache_path(features, entry["window_id"]))
            # A window whose video ended early has fewer than 50 frames. It cannot be
            # compared frame for frame, so it is left out and counted, like a no-face window.
            if (
                stored["lips"].shape[0] != WINDOW_FRAMES
                or stored["audio"].shape[0] != WINDOW_FRAMES
            ):
                short += 1
                continue
            owners.append(entry["clip_id"])
            lips.append(stored["lips"])
            audio.append(stored["audio"])
    if short:
        print(
            f"left out {short} windows shorter than {WINDOW_FRAMES} frames", flush=True
        )
    if not owners:
        return (
            [],
            np.zeros((0, 50, 768), np.float16),
            np.zeros((0, 50, 768), np.float16),
        )
    return owners, np.stack(lips), np.stack(audio)


def alignment_loss(
    head, lips: torch.Tensor, audio: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    batch = lips.shape[0]
    shift = int(torch.randint(5, 21, (1,), generator=generator))
    shifted = torch.roll(audio, shifts=shift, dims=1)
    other = audio[torch.roll(torch.arange(batch, device=audio.device), 1)]
    positive = head(lips, audio)
    negative = torch.cat([head(lips, shifted), head(lips, other)], dim=1)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
        + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
    )


def _tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def train(
    seed: int,
    lips: np.ndarray,
    audio: np.ndarray,
    val_lips: np.ndarray,
    val_audio: np.ndarray,
    device: str,
    epochs: int,
) -> tuple[torch.nn.Module, dict]:
    from deepfake_detection.avsync.heads import AlignmentHead

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = AlignmentHead(width=lips.shape[-1]).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (float("inf"), None, -1)
    history = []
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(lips), generator=generator).numpy()
        for start in range(0, len(order), 64):
            pick = order[start : start + 64]
            if len(pick) < 2:
                continue
            loss = alignment_loss(
                head,
                _tensor(lips[pick], device),
                _tensor(audio[pick], device),
                generator,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            fixed = torch.Generator().manual_seed(20261113)
            losses = [
                alignment_loss(
                    head,
                    _tensor(val_lips[i : i + 256], device),
                    _tensor(val_audio[i : i + 256], device),
                    fixed,
                ).item()
                for i in range(0, len(val_lips), 256)
                if len(val_lips[i : i + 256]) > 1
            ]
        value = float(np.mean(losses)) if losses else float("inf")
        history.append(value)
        if value < best[0]:
            best = (
                value,
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                epoch,
            )
    head.load_state_dict(best[1])
    return head, {
        "selection": "lowest alignment loss on real validation windows",
        "val_loss": best[0],
        "epoch": best[2],
        "history": history,
    }


def score_clips(
    head, owners: list[str], lips: np.ndarray, audio: np.ndarray, device: str
) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    with torch.no_grad():
        for i in range(0, len(owners), 512):
            window = head.window_score(
                _tensor(lips[i : i + 512], device), _tensor(audio[i : i + 512], device)
            )
            for owner, value in zip(
                owners[i : i + 512], window.cpu().numpy(), strict=True
            ):
                sums.setdefault(owner, []).append(float(value))
    return {clip: float(np.mean(values)) for clip, values in sums.items()}


def write_predictions(
    path: Path, clips: dict[str, dict], scores: dict[str, float]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "clip_id",
        "label",
        "probability",
        "score",
        "source",
        "method",
        "manipulation_type",
        "video_fake",
        "audio_fake",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip_id, row in clips.items():
            score = scores.get(clip_id)
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "label": row["clip_fake"],
                    # Clips without a usable window are abstentions, written blank, never dropped.
                    "probability": "" if score is None else f"{1 - np.exp(-score):.6f}",
                    "score": "" if score is None else f"{score:.6f}",
                    "source": row["source"],
                    "method": row["method"],
                    "manipulation_type": row["manipulation_type"],
                    "video_fake": row["video_fake"],
                    "audio_fake": row["audio_fake"],
                }
            )
    return sum(1 for c in clips if c in scores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score",
        nargs="*",
        default=["val", "test", "test-identity-strict"],
        help="split CSV stems in --split-dir to score",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    train_clips = load_split(arguments.split_dir / "train.csv")
    val_clips = load_split(arguments.split_dir / "val.csv")
    real_train = {k: v for k, v in train_clips.items() if v["clip_fake"] == "0"}
    real_val = {k: v for k, v in val_clips.items() if v["clip_fake"] == "0"}
    _, lips, audio = load_windows(arguments.features, real_train)
    _, val_lips, val_audio = load_windows(arguments.features, real_val)
    print(
        f"real training windows {len(lips)}, real validation windows {len(val_lips)}",
        flush=True,
    )
    if len(lips) < 2 or len(val_lips) < 2:
        raise SystemExit("Too few real windows to train the sync path")

    targets = {}
    for stem in arguments.score:
        path = arguments.split_dir / f"{stem}.csv"
        if path.exists():
            clips = load_split(path)
            targets[stem] = (clips, *load_windows(arguments.features, clips))

    arguments.output.mkdir(parents=True, exist_ok=True)
    log = {}
    for seed in arguments.seeds:
        head, info = train(
            seed, lips, audio, val_lips, val_audio, arguments.device, arguments.epochs
        )
        torch.save(
            {"state_dict": head.state_dict(), "metadata": {"seed": seed, **info}},
            arguments.output / f"sync-seed{seed}.pt",
        )
        scored = {}
        for stem, (clips, owners, target_lips, target_audio) in targets.items():
            scores = score_clips(
                head, owners, target_lips, target_audio, arguments.device
            )
            scored[stem] = write_predictions(
                arguments.output / f"sync-{stem}-seed{seed}-predictions.csv",
                clips,
                scores,
            )
        log[f"seed{seed}"] = {
            **info,
            "real_train_windows": len(lips),
            "real_val_windows": len(val_lips),
            "scored_clips": scored,
        }
        print(
            seed, {k: v for k, v in info.items() if k != "history"}, scored, flush=True
        )
    (arguments.output / "training.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
