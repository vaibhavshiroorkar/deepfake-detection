"""
Train the DINOv2 classification head for image deepfake detection.

Usage:
    python -m training.train_image \
        --data-real /path/to/real_images/ \
        --data-fake /path/to/fake_images/ \
        --output ./weights/dinov2_head.pt \
        --epochs 15 \
        --batch-size 32 \
        --lr 1e-3

The DINOv2 backbone is frozen — only the MLP head is trained.
Supports mixed-precision and cosine LR scheduling with warmup.
Validates on a held-out split and saves the best checkpoint by AUC.

Datasets:
    - FaceForensics++ : extract faces from videos, separate real/fake folders
    - Celeb-DF       : same structure
    - Custom          : any two directories of images (real vs fake)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Add the project root so we can import from app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BinaryImageDataset(Dataset):
    """Loads images from two directories: real (label=0) and fake (label=1)."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, real_dir: str, fake_dir: str, processor):
        self.samples: list[tuple[str, int]] = []
        for path in Path(real_dir).rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 0))
        for path in Path(fake_dir).rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 1))
        self.processor = processor
        print(f"Dataset: {len(self.samples)} images "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label


def train(args):
    from transformers import AutoImageProcessor, Dinov2Model
    from app.detectors.dinov2_head import DINOv2ClassifierHead, DINOv2Classifier

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Load DINOv2 backbone + processor
    print("Loading DINOv2-large backbone...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    backbone = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device).eval()

    # Dataset
    dataset = BinaryImageDataset(args.data_real, args.data_fake, processor)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Head
    head = DINOv2ClassifierHead(in_dim=1024).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = 0.0
    os.makedirs(Path(args.output).parent, exist_ok=True)

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Extract frozen embeddings
            with torch.no_grad():
                features = backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = head(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validation
        head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                features = backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
                logits = head(features)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.numpy())

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={total_loss/max(total,1):.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(head.state_dict(), args.output)
            print(f"  → Saved best model (AUC={auc:.4f}) to {args.output}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Weights saved to: {args.output}")
    print(f"Set VERITAS_DINOV2_WEIGHTS={os.path.abspath(args.output)} in your .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv2 deepfake image classifier head")
    parser.add_argument("--data-real", required=True, help="Directory of real images")
    parser.add_argument("--data-fake", required=True, help="Directory of fake/AI images")
    parser.add_argument("--output", default="./weights/dinov2_head.pt", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    train(parser.parse_args())
