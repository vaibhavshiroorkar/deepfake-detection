"""
Train the Temporal Transformer for video deepfake detection.

Usage:
    python -m training.train_video \
        --data-real /path/to/real_videos/ \
        --data-fake /path/to/fake_videos/ \
        --output ./weights/temporal_transformer.pt \
        --epochs 20 \
        --n-frames 16

Two-step process:
  1. Pre-extract DINOv2 CLS embeddings per frame (cached to disk).
  2. Train the transformer on sequences of embeddings.

Caching makes training fast — DINOv2 runs once per video, then training
iterations only touch the small transformer (4 layers, ~50M params).

Datasets:
    - FaceForensics++ (video split)
    - Celeb-DF (video split)
    - Custom: any two directories of video files (real vs fake)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sample_video_frames(
    video_path: str, n_frames: int = 16
) -> list[np.ndarray] | None:
    """Extract N evenly-spaced RGB frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < n_frames:
        # Read as many as we can
        indices = list(range(total)) if total > 0 else []
    else:
        indices = np.linspace(0, total - 1, num=n_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    cap.release()
    return frames if len(frames) >= 2 else None


class VideoEmbeddingDataset(Dataset):
    """Loads pre-extracted DINOv2 embeddings for video clips."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        cache_dir: str,
        n_frames: int,
        processor,
        backbone,
        device,
    ):
        self.samples: list[tuple[str, int]] = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_frames = n_frames

        for path in Path(real_dir).rglob("*"):
            if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                self.samples.append((str(path), 0))
        for path in Path(fake_dir).rglob("*"):
            if path.suffix.lower() in self.VIDEO_EXTENSIONS:
                self.samples.append((str(path), 1))

        print(f"Dataset: {len(self.samples)} videos "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

        # Pre-extract embeddings
        self._extract_all(processor, backbone, device)

    def _cache_path(self, video_path: str) -> Path:
        h = hashlib.md5(video_path.encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def _extract_all(self, processor, backbone, device):
        """Extract DINOv2 embeddings for all videos (cached)."""
        to_extract = [
            (i, p) for i, (p, _) in enumerate(self.samples)
            if not self._cache_path(p).exists()
        ]
        if not to_extract:
            print("All embeddings cached.")
            return

        print(f"Extracting embeddings for {len(to_extract)} videos...")
        for step, (i, video_path) in enumerate(to_extract):
            frames = _sample_video_frames(video_path, self.n_frames)
            if frames is None:
                # Save empty array as marker
                np.save(self._cache_path(video_path), np.zeros((0, 1024)))
                continue

            embeddings = []
            for rgb in frames:
                pil = Image.fromarray(rgb)
                inputs = processor(images=pil, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    cls_emb = backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
                embeddings.append(cls_emb[0].cpu().numpy())

            emb_array = np.stack(embeddings, axis=0)  # (N, 1024)
            np.save(self._cache_path(video_path), emb_array)

            if (step + 1) % 50 == 0:
                print(f"  [{step+1}/{len(to_extract)}]")

        print("Extraction complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        emb = np.load(self._cache_path(path))

        if emb.shape[0] == 0:
            # Unreadable video — return zeros
            emb = np.zeros((self.n_frames, 1024), dtype=np.float32)

        # Pad or truncate to fixed length
        if emb.shape[0] < self.n_frames:
            pad = np.zeros((self.n_frames - emb.shape[0], 1024), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)
        elif emb.shape[0] > self.n_frames:
            indices = np.linspace(0, emb.shape[0] - 1, self.n_frames, dtype=int)
            emb = emb[indices]

        return torch.from_numpy(emb.astype(np.float32)), label


def train(args):
    from transformers import AutoImageProcessor, Dinov2Model
    from app.detectors.temporal_transformer import TemporalTransformer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Load DINOv2 for embedding extraction
    print("Loading DINOv2-large for frame embedding extraction...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    backbone = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device).eval()

    # Dataset
    dataset = VideoEmbeddingDataset(
        args.data_real, args.data_fake,
        cache_dir=args.cache_dir,
        n_frames=args.n_frames,
        processor=processor,
        backbone=backbone,
        device=device,
    )

    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model
    model = TemporalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    os.makedirs(Path(args.output).parent, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                logits = model(embeddings)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.numpy())

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={total_loss/max(total,1):.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.output)
            print(f"  → Saved best model (AUC={auc:.4f}) to {args.output}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Weights saved to: {args.output}")
    print(f"Set VERITAS_TEMPORAL_WEIGHTS={os.path.abspath(args.output)} in your .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal transformer for video deepfake detection")
    parser.add_argument("--data-real", required=True, help="Directory of real videos")
    parser.add_argument("--data-fake", required=True, help="Directory of fake videos")
    parser.add_argument("--output", default="./weights/temporal_transformer.pt", help="Output weights path")
    parser.add_argument("--cache-dir", default="./cache/video_embeddings", help="Dir for cached DINOv2 embeddings")
    parser.add_argument("--n-frames", type=int, default=16, help="Frames per video")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    train(parser.parse_args())
