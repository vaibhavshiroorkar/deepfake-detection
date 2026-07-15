"""
Evaluate a saved visual-stream checkpoint on a split (default: the held-out
test set, which training never touches). Reports all metrics + confusion
matrix, and a per-manipulation-type breakdown so we can see WHICH fakes the
stream catches -- useful context for the "AUC looks suspiciously high" check
(docs/stage-1-plan.md) and later for the Stage 7 ablation story.

Usage:
    python -m evaluation.eval_checkpoint                       # efficientnet best.pt on test
    python -m evaluation.eval_checkpoint --split val
    python -m evaluation.eval_checkpoint --stream efficientnet --split test
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from models.streams.common.config import StreamConfig
from models.streams.common.visual_stream import VisualStream
from preprocessing.dataset import ClipDataset, VISUAL_FAKE_TYPES
from evaluation.metrics import compute_metrics, format_metrics
from training.train_visual_stream import _collate

CKPT_DIR = _REPO_ROOT / "models" / "streams"


def load_model(stream_name: str, device):
    ckpt_path = CKPT_DIR / stream_name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- train the stream first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = StreamConfig(**ckpt["config"])
    model = VisualStream(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path} (saved at epoch {ckpt.get('epoch')}, "
          f"val AUC {ckpt['val_metrics']['auc_roc']:.3f})")
    return model, config


@torch.no_grad()
def run(stream_name: str, split: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(stream_name, device)

    ds = ClipDataset(split, label_mode="visual")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=_collate,
    )

    all_logits, all_labels, all_types = [], [], []
    manip = ds.manifest["manipulation_type"].to_numpy()
    idx = 0
    for frames, _audio, labels, clip_ids in loader:
        frames = frames.to(device)
        with torch.autocast(device_type=device.type, enabled=config.use_amp):
            logit, _emb = model(frames)
        n = len(clip_ids)
        all_logits.append(logit.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_types.append(manip[idx:idx + n])
        idx += n

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    types = np.concatenate(all_types)

    print(f"\n=== {stream_name} on '{split}' split ({len(labels)} clips, "
          f"real={int((labels==0).sum())}, fake={int((labels==1).sum())}) ===")
    print(format_metrics(compute_metrics(logits, labels)))

    # Per-manipulation-type recall: of each fake method's clips, how many did we
    # catch? (Real types show accuracy on reals.) Reveals blind spots.
    print("\nPer-manipulation-type breakdown (visual label; RealVideo-* are 'real'):")
    probs = 1.0 / (1.0 + np.exp(-logits))
    for mt in sorted(set(types)):
        mask = types == mt
        visual_label = 1 if mt in VISUAL_FAKE_TYPES else 0
        pred_fake_rate = float((probs[mask] >= 0.5).mean())
        # for a fake type this is recall; for a real type it's the false-positive rate
        metric_name = "recall" if visual_label == 1 else "false-positive rate"
        print(f"  {mt:<22} n={int(mask.sum()):>3}  {metric_name}={pred_fake_rate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="efficientnet")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run(args.stream, args.split)
