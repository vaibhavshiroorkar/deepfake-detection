"""
Train one visual stream (EfficientNet now; Xception/DINOv2 later) with its
temporary classifier head, and report all required metrics. This is the
Stage 2 template every later visual stream reuses -- only the StreamConfig
changes.

6GB-VRAM strategy (see StreamConfig): small batch_size, gradient accumulation
to reach a useful effective batch, mixed precision (AMP), gradient
checkpointing in the backbone, and running the backbone on frame sub-chunks.

Two-phase freezing (Stage 2 plan):
  Phase 1 (first `freeze_backbone_epochs`): backbone frozen + in eval() so its
    BatchNorm stats don't drift; train only temporal + projection + head.
  Phase 2: unfreeze backbone, fine-tune end-to-end at a much smaller LR.

Best checkpoint is chosen by validation AUC (threshold-free), not loss.

Run:  python -m training.train_visual_stream            # full run (asks nothing)
      python -m training.train_visual_stream --smoke    # tiny run to prove the loop
"""
import sys
import argparse
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
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


def make_loaders(config: StreamConfig):
    """Train loader uses a WeightedRandomSampler for class balance; val loader doesn't."""
    train_ds = ClipDataset("train", label_mode="visual")
    val_ds = ClipDataset("val", label_mode="visual")

    # Balanced sampling: weight each sample by 1 / (count of its class) so the
    # fake-heavy training set presents ~50/50 real/fake to the model.
    labels = train_ds.labels()
    class_counts = np.bincount(labels, minlength=2)
    per_class_w = 1.0 / np.maximum(class_counts, 1)
    sample_weights = per_class_w[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    print(f"Train class counts (visual labels): real={class_counts[0]}, fake={class_counts[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=_collate,
    )
    return train_loader, val_loader


def _augment_flip(frames: torch.Tensor) -> torch.Tensor:
    """
    Random horizontal flip of the whole clip (same flip for all 16 frames).
    Safe for deepfake artifacts (mirroring doesn't erase blending seams);
    heavier augmentations that smear pixels are deliberately avoided.
    """
    if torch.rand(1).item() < 0.5:
        return torch.flip(frames, dims=[-1])
    return frames


def _set_batchnorm_eval(module):
    """Put every BatchNorm layer into eval() so it uses its (ImageNet) running
    stats instead of recomputing them from tiny batches. Standard trick for
    stable small-batch fine-tuning; without it the unfreeze corrupts features."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def set_backbone_phase(model: VisualStream, trainable: bool, freeze_bn_on_finetune: bool):
    """Toggle backbone trainability AND its train/eval mode together."""
    model.set_backbone_trainable(trainable)
    model.backbone.train(trainable)  # eval() when frozen so BatchNorm stats are fixed
    # Even when the backbone is trainable, keep its BatchNorm layers frozen
    # (eval) during fine-tuning -- conv weights still update, BN stats don't.
    if trainable and freeze_bn_on_finetune:
        _set_batchnorm_eval(model.backbone)


@torch.no_grad()
def evaluate(model, loader, device, use_amp) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for frames, _audio, labels, _clip_ids in loader:
        frames = frames.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logit, _emb = model(frames)
        all_logits.append(logit.float().cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))


def train_stream(config: StreamConfig, smoke: bool = False):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | stream: {config.stream_name} | backbone: {config.backbone_name}")
    if device.type == "cpu":
        print("WARNING: training on CPU will be very slow.")

    model = VisualStream(config).to(device)
    train_loader, val_loader = make_loaders(config)

    criterion = nn.BCEWithLogitsLoss()
    # Two param groups so the backbone can fine-tune at a smaller LR in phase 2.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.lr_backbone},
            {"params": list(model.temporal.parameters())
                       + list(model.projection.parameters())
                       + list(model.temp_head.parameters()), "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    best_auc = -1.0
    ckpt_dir = CKPT_DIR / config.stream_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    max_train_batches = 6 if smoke else None
    max_val_batches = 6 if smoke else None
    epochs = 1 if smoke else config.epochs

    for epoch in range(epochs):
        # --- flip freeze phase at the boundary ---
        frozen = epoch < config.freeze_backbone_epochs and not smoke
        set_backbone_phase(model, trainable=not frozen,
                           freeze_bn_on_finetune=config.freeze_batchnorm_on_finetune)
        model.temporal.train(); model.projection.train(); model.temp_head.train()
        phase = "FROZEN backbone" if frozen else "fine-tuning end-to-end"
        print(f"\nEpoch {epoch+1}/{epochs} ({phase})")

        running_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (frames, _audio, labels, _clip_ids) in enumerate(train_loader):
            if max_train_batches and i >= max_train_batches:
                break
            frames = _augment_flip(frames).to(device, non_blocking=True)
            targets = labels.float().to(device)

            with torch.autocast(device_type=device.type, enabled=config.use_amp):
                logit, _emb = model(frames)
                loss = criterion(logit, targets) / config.grad_accum_steps

            scaler.scale(loss).backward()
            # Step the optimizer every grad_accum_steps mini-batches (effective
            # batch = batch_size * grad_accum_steps).
            if (i + 1) % config.grad_accum_steps == 0:
                # Gradient clipping: unscale first (AMP), then clip the global
                # norm. This is what prevents the destructive gradient spike
                # the moment the backbone unfreezes.
                if config.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * config.grad_accum_steps
            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  batch {n_batches}: loss={running_loss / n_batches:.4f}")

        avg_loss = running_loss / max(n_batches, 1)

        # --- validation ---
        if smoke:  # cap val batches too, just to prove the loop end to end
            val_metrics = _evaluate_capped(model, val_loader, device, config.use_amp, max_val_batches)
        else:
            val_metrics = evaluate(model, val_loader, device, config.use_amp)
        print(f"  train_loss={avg_loss:.4f}")
        print("  val: " + format_metrics(val_metrics))

        auc = val_metrics["auc_roc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            # smoke saves to a throwaway name so a quick test can't clobber a
            # real checkpoint.
            ckpt_path = ckpt_dir / ("best_smoke.pt" if smoke else "best.pt")
            torch.save({"model_state": model.state_dict(), "config": config.__dict__,
                        "val_metrics": val_metrics, "epoch": epoch}, ckpt_path)
            print(f"  saved new best (AUC={auc:.3f}) -> {ckpt_path}")

    print(f"\nDone. Best val AUC: {best_auc:.3f}")
    return best_auc


@torch.no_grad()
def _evaluate_capped(model, loader, device, use_amp, max_batches):
    model.eval()
    all_logits, all_labels = [], []
    for i, (frames, _audio, labels, _clip_ids) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        frames = frames.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logit, _emb = model(frames)
        all_logits.append(logit.float().cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (few batches) to prove the loop works, not real training.")
    args = parser.parse_args()
    train_stream(StreamConfig(), smoke=args.smoke)
