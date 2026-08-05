"""
Evaluate a trained visual stream on a held-out split.

The trainer only ever reports validation numbers, and validation is what
selects the checkpoint -- so it cannot be an unbiased estimate of anything.
This runs a saved checkpoint over a split it never influenced (test.csv by
default: 300 clips, identity-disjoint from train and val).

The architecture is rebuilt from the config stored *inside* the checkpoint,
not from the preset, so a checkpoint trained under different settings still
loads correctly instead of silently mismatching shapes.

Run:  python -m evaluation.evaluate_stream --stream xception
      python -m evaluation.evaluate_stream --stream efficientnet --split val
      python -m evaluation.evaluate_stream --stream xception --checkpoint models/streams/xception/last.pt
"""
import sys
import argparse
import dataclasses
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

from models.streams.common.config import StreamConfig
from models.streams.common.visual_stream import VisualStream
from preprocessing.dataset import ClipDataset
from evaluation.metrics import compute_metrics, format_metrics

CKPT_DIR = _REPO_ROOT / "models" / "streams"


def _collate(batch):
    face_seqs, audios, labels, clip_ids = zip(*batch)
    return torch.stack(face_seqs), torch.stack(audios), torch.stack(labels), list(clip_ids)


def config_from_checkpoint(state: dict, stream_name: str) -> StreamConfig:
    """Rebuild the StreamConfig the checkpoint was trained with.

    Checkpoints store the config as a plain dict (the dashboard loads with
    weights_only=True, which refuses to unpickle a dataclass instance). Unknown
    keys are dropped rather than raising: a checkpoint written before a field
    existed should still load, and StreamConfig's default fills the gap.
    """
    saved = state.get("config")
    if not isinstance(saved, dict):
        print("  checkpoint carries no config; falling back to defaults")
        return StreamConfig(stream_name=stream_name)
    known = {f.name for f in dataclasses.fields(StreamConfig)}
    unknown = set(saved) - known
    if unknown:
        print(f"  ignoring unknown config keys: {sorted(unknown)}")
    return StreamConfig(**{k: v for k, v in saved.items() if k in known})


@torch.no_grad()
def collect_logits(model, loader, device, use_amp):
    """Run the split once, returning raw logits, labels and clip ids."""
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for frames, _audio, labels, clip_ids in loader:
        frames = frames.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logit, _emb = model(frames)
        all_logits.append(logit.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_ids.extend(clip_ids)
    return np.concatenate(all_logits), np.concatenate(all_labels), all_ids


def calibration_report(logits: np.ndarray, labels: np.ndarray) -> str:
    """How much of the [0,1] range the model actually uses.

    A saturated model -- every fake at 0.9998, every real at 0.0001 -- can score
    a near-perfect AUC while carrying almost no usable confidence. That matters
    at fusion, where a stream pinned to two values contributes no gradient of
    information, and it means the 0.5 threshold is nowhere near the real
    decision boundary.
    """
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    real, fake = probs[labels == 0], probs[labels == 1]
    lines = []
    for name, arr in (("real", real), ("fake", fake)):
        if arr.size:
            # Percentiles, not min/max: the extremes of each class are set by
            # its misclassified clips, so a model with errors on both sides
            # looks like it uses the full range when its bulk is still pinned
            # to two values.
            lo, med, hi = np.percentile(arr, [5, 50, 95])
            lines.append(f"  {name:4s} n={arr.size:<4d} "
                         f"p5={lo:.4f}  median={med:.4f}  p95={hi:.4f}  "
                         f"(min {arr.min():.4f}, max {arr.max():.4f})")

    # The honest saturation measure: how much of the output lands in the top and
    # bottom 1% of the range. A stream near 100% here is effectively emitting a
    # hard label, and contributes almost no gradient of information to fusion,
    # whatever its AUC says.
    saturated = float(np.mean((probs > 0.99) | (probs < 0.01)))
    lines.append(f"  saturated (p>0.99 or p<0.01): {saturated:6.1%} of all clips")
    mid_band = float(np.mean((probs >= 0.1) & (probs <= 0.9)))
    lines.append(f"  in the middle band (0.1-0.9):  {mid_band:6.1%}")
    return "\n".join(lines)


def evaluate_stream(stream: str, split: str = "test", checkpoint: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint) if checkpoint else CKPT_DIR / stream / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    print(f"Stream: {stream} | split: {split} | device: {device}")
    print(f"Checkpoint: {ckpt_path.relative_to(_REPO_ROOT)}")

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "epoch" in state:
        print(f"  trained through epoch {int(state['epoch']) + 1}")
    config = config_from_checkpoint(state, stream)

    model = VisualStream(config).to(device)
    model.load_state_dict(state["model_state"])

    dataset = ClipDataset(split, label_mode="visual")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, collate_fn=_collate)
    print(f"  {len(dataset)} clips\n")

    logits, labels, _ids = collect_logits(model, loader, device, config.use_amp)
    metrics = compute_metrics(logits, labels)

    print(format_metrics(metrics))
    # format_metrics rounds to 3 places, which turns 0.99983 into "1.000" -- a
    # far stronger claim than the data supports. Print the raw values too.
    print(f"\nfull precision: auc={metrics['auc_roc']:.5f} "
          f"acc={metrics['accuracy']:.5f} eer={metrics['eer']:.5f} "
          f"logloss={metrics['log_loss']:.5f}")
    print("\nprobability distribution:")
    print(calibration_report(logits, labels))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="xception",
                        help="Stream name; also picks models/streams/<stream>/best.pt.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test).")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path, overriding the default best.pt.")
    args = parser.parse_args()
    evaluate_stream(args.stream, split=args.split, checkpoint=args.checkpoint)
