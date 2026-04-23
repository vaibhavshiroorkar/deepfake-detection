"""
Train the Whisper classifier head for audio deepfake detection.

Usage:
    python -m training.train_audio \
        --data-real /path/to/real_audio/ \
        --data-fake /path/to/fake_audio/ \
        --output ./weights/whisper_head.pt \
        --epochs 20 \
        --batch-size 16

The Whisper encoder is frozen — only the attention pooling + MLP head
are trained.  Audio files are loaded, resampled to 16kHz, and the
Whisper encoder produces a (T, 512) hidden state per clip.

Datasets:
    - ASVspoof 2019 LA
    - In-the-Wild TTS dataset
    - Custom: any two directories of audio files (real vs fake)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy import signal as sp_signal
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class AudioDataset(Dataset):
    """Loads audio from two directories: real (label=0) and fake (label=1)."""

    EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    MAX_SECONDS = 15  # cap per clip

    def __init__(self, real_dir: str, fake_dir: str, processor):
        self.samples: list[tuple[str, int]] = []
        for path in Path(real_dir).rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 0))
        for path in Path(fake_dir).rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                self.samples.append((str(path), 1))
        self.processor = processor
        print(f"Dataset: {len(self.samples)} clips "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz
        if sr != 16000:
            new_len = int(round(len(audio) * 16000 / sr))
            audio = sp_signal.resample(audio, new_len).astype(np.float32)
            sr = 16000

        # Cap length
        max_samples = sr * self.MAX_SECONDS
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        feats = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        return feats, label


def train(args):
    from transformers import WhisperFeatureExtractor, WhisperModel
    from app.detectors.whisper_classifier import WhisperClassifierHead

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Load encoder
    print("Loading Whisper-base encoder...")
    processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    full_model = WhisperModel.from_pretrained("openai/whisper-base")
    encoder = full_model.encoder.to(device).eval()

    # Dataset
    dataset = AudioDataset(args.data_real, args.data_fake, processor)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Head
    head = WhisperClassifierHead(in_dim=512).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    os.makedirs(Path(args.output).parent, exist_ok=True)

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                hidden = encoder(feats).last_hidden_state  # (B, T, 512)

            logits = head(hidden)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validation
        head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                hidden = encoder(feats).last_hidden_state
                logits = head(hidden)
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
            torch.save(head.state_dict(), args.output)
            print(f"  → Saved best model (AUC={auc:.4f}) to {args.output}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Weights saved to: {args.output}")
    print(f"Set VERITAS_WHISPER_WEIGHTS={os.path.abspath(args.output)} in your .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper audio deepfake classifier head")
    parser.add_argument("--data-real", required=True, help="Directory of real audio clips")
    parser.add_argument("--data-fake", required=True, help="Directory of fake/TTS audio clips")
    parser.add_argument("--output", default="./weights/whisper_head.pt", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    train(parser.parse_args())
