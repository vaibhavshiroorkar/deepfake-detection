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

Best checkpoint is chosen by validation AUC (threshold-free), not loss, with
logloss breaking AUC ties that fall inside AUC_TIE_TOL -- see
is_better_checkpoint for why calibration decides when ranking cannot.

Run:  python -m training.train_visual_stream --stream xception
      python -m training.train_visual_stream --smoke    # tiny run to prove the loop
      python -m training.train_visual_stream --resume   # continue an interrupted run

A resume point (last.pt) is written after every epoch, holding model, optimizer,
GradScaler and RNG state. Without the optimizer state a resumed run restarts
AdamW's moment estimates from zero, which shows up as a loss spike.
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

from models.streams.common.config import (
    StreamConfig, efficientnet_config, xception_config, dinov2_config,
)
from models.streams.common.visual_stream import VisualStream
from preprocessing.dataset import ClipDataset
from evaluation.metrics import compute_metrics, format_metrics

CKPT_DIR = _REPO_ROOT / "models" / "streams"

# Written after every epoch so an interrupted run can pick up where it stopped.
# Deliberately not best.pt: that holds whichever epoch scored the highest AUC,
# which is usually not the last one trained, and it carries no optimizer state.
RESUME_NAME = "last.pt"

# Two epochs whose val AUC differ by less than this are tied as far as a 300-clip
# validation set can tell, so calibration decides instead. Without it, selection
# on AUC alone kept EfficientNet's epoch 7 (AUC 0.9988, logloss 0.143) over its
# epoch 8 (AUC 0.9982, logloss 0.047) -- a 0.0006 AUC edge bought with 3x the
# logloss and 1.3 points of accuracy. A stream that only ever emits 0.9998 or
# 0.0001 carries almost no information into fusion.
AUC_TIE_TOL = 1e-3


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


def is_better_checkpoint(auc, logloss, best_auc, best_logloss) -> bool:
    """Select on AUC, but break within-noise AUC ties on calibration.

    A clear AUC win always takes it. Inside AUC_TIE_TOL the two epochs rank
    equally well as far as 300 validation clips can resolve, so the better
    calibrated one (lower logloss) is the more useful checkpoint downstream.
    """
    if np.isnan(auc):
        return False
    if auc > best_auc + AUC_TIE_TOL:
        return True
    if auc < best_auc - AUC_TIE_TOL:
        return False
    # Tied on AUC, so calibration decides. NaN needs handling on both sides:
    # every comparison against it is False, so a NaN incumbent would otherwise
    # lock the tiebreak out permanently rather than losing to a real value.
    if np.isnan(logloss):
        return False
    if np.isnan(best_logloss):
        return True
    return logloss < best_logloss


def save_resume_point(path, model, optimizer, scaler, config, epoch, best_auc,
                      best_logloss=float("inf")):
    """Everything needed to continue this run, and nothing that isn't.

    Only tensors and primitives go in, so the file still loads under
    weights_only=True like every other checkpoint here. numpy's RNG state is
    left out on purpose: it would need pickling to survive, and nothing in the
    training loop draws from it -- the augmentation flip and the sampler both
    use torch's generator.
    """
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config.__dict__,
        "epoch": epoch,               # the epoch just finished, 0-based
        "best_auc": best_auc,
        "best_logloss": best_logloss,   # the logloss of the epoch best.pt holds
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(state, path)


def load_resume_point(path, model, optimizer, scaler, device) -> tuple[int, float, float]:
    """Restore a run from `path`; returns (next epoch, best AUC, its logloss).

    Restoring the optimizer matters more than it might look: AdamW carries
    per-parameter moment estimates, and dropping them restarts the moment
    accumulation from zero, which produces a visible loss spike on the first
    resumed step. The GradScaler's scale factor is the same story for AMP.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scaler.load_state_dict(state["scaler_state"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    start_epoch = int(state["epoch"]) + 1
    best_auc = float(state["best_auc"])
    # Resume points written before the logloss tiebreak existed have no such key.
    # inf is the right fallback: any real logloss beats it, so the tiebreak stays
    # available rather than being locked out by a missing value.
    best_logloss = float(state.get("best_logloss", float("inf")))
    print(f"Resumed from {path.name}: {start_epoch} epoch(s) done, best AUC so far {best_auc:.3f}")
    return start_epoch, best_auc, best_logloss


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


def train_stream(config: StreamConfig, smoke: bool = False, resume: bool = False):
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
    best_logloss = float("inf")
    ckpt_dir = CKPT_DIR / config.stream_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    max_train_batches = 6 if smoke else None
    max_val_batches = 6 if smoke else None
    epochs = 1 if smoke else config.epochs

    start_epoch = 0
    resume_path = ckpt_dir / RESUME_NAME
    if resume and not smoke:
        if resume_path.exists():
            start_epoch, best_auc, best_logloss = load_resume_point(
                resume_path, model, optimizer, scaler, device)
            if start_epoch >= epochs:
                print(f"Nothing to do: {start_epoch} epoch(s) already done of {epochs}.")
                return best_auc
        else:
            print(f"--resume given but {resume_path} does not exist; starting from scratch.")

    for epoch in range(start_epoch, epochs):
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
        logloss = val_metrics["log_loss"]
        if is_better_checkpoint(auc, logloss, best_auc, best_logloss):
            tie = abs(auc - best_auc) <= AUC_TIE_TOL and best_auc >= 0
            best_auc, best_logloss = auc, logloss
            # smoke saves to a throwaway name so a quick test can't clobber a
            # real checkpoint.
            ckpt_path = ckpt_dir / ("best_smoke.pt" if smoke else "best.pt")
            torch.save({"model_state": model.state_dict(), "config": config.__dict__,
                        "val_metrics": val_metrics, "epoch": epoch}, ckpt_path)
            why = "better calibrated at tied AUC" if tie else f"AUC={auc:.4f}"
            print(f"  saved new best ({why}, logloss={logloss:.3f}) -> {ckpt_path}")

        # After the best-checkpoint decision, so best_auc is current: a resume
        # must not re-save a checkpoint for an epoch it has already beaten.
        # Smoke runs are excluded so a quick test cannot leave a resume point
        # that a later real run would pick up.
        if not smoke:
            save_resume_point(resume_path, model, optimizer, scaler, config, epoch,
                              best_auc, best_logloss)

    print(f"\nDone. Best val AUC: {best_auc:.3f} (logloss {best_logloss:.3f})")
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


# Which preset each --stream name builds. Adding a visual stream means adding a
# preset in config.py and one line here, not touching the training loop.
STREAM_PRESETS = {
    "efficientnet": efficientnet_config,
    "xception": xception_config,
    "dinov2": dinov2_config,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="efficientnet", choices=sorted(STREAM_PRESETS),
                        help="Which visual stream to train (default: efficientnet).")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (few batches) to prove the loop works, not real training.")
    parser.add_argument("--resume", action="store_true",
                        help=f"Continue from models/streams/<stream>/{RESUME_NAME} if it exists.")
    # Overrides for the knobs that differ per run. freeze-epochs matters most:
    # the default equals `epochs`, which keeps the backbone frozen the whole run.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=None,
                        help="Epochs to keep the backbone frozen before fine-tuning.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers. Keep 0 on Windows: spawned workers "
                             "have died mid-run here and take the run with them.")
    args = parser.parse_args()

    overrides = {}
    for cli_name, field in (("epochs", "epochs"), ("batch_size", "batch_size"),
                            ("grad_accum_steps", "grad_accum_steps"),
                            ("freeze_epochs", "freeze_backbone_epochs"),
                            ("num_workers", "num_workers")):
        value = getattr(args, cli_name)
        if value is not None:
            overrides[field] = value

    config = STREAM_PRESETS[args.stream](**overrides)
    train_stream(config, smoke=args.smoke, resume=args.resume)
