"""DINOv2-large image classifier for deepfake detection.

Architecture:
  facebook/dinov2-large (frozen) → [CLS] token (1024-d)
  → LayerNorm → Linear(1024, 512) → GELU → Dropout(0.1)
  → Linear(512, 2)   [real, fake]

The backbone is always frozen; only the classification head is trained.
When fine-tuned weights are not available, the head runs with random init
and the system falls back to the existing Swin-v2 classifier.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# DINOv2-large output dimension
DINOV2_DIM = 1024


class DINOv2ClassifierHead(nn.Module):
    """Small MLP head for binary classification on DINOv2 CLS embeddings."""

    def __init__(self, in_dim: int = DINOV2_DIM, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """cls_token: (B, in_dim) → logits: (B, 2)  [real, fake]"""
        x = self.norm(cls_token)
        return self.head(x)


class DINOv2Classifier(nn.Module):
    """Full model: frozen DINOv2 backbone + trainable head."""

    def __init__(self, backbone: nn.Module, head: DINOv2ClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] embeddings without classification.

        Returns: (B, 1024)
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
        # DINOv2 returns last_hidden_state; [CLS] is index 0
        return outputs.last_hidden_state[:, 0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Full forward: image → logits (B, 2)."""
        cls_emb = self.extract_features(pixel_values)
        return self.head(cls_emb)


def load_dinov2_classifier(
    weights_path: str | Path | None,
    device: torch.device,
) -> tuple[DINOv2Classifier, Any]:
    """Load the full DINOv2 classifier (backbone + head).

    Returns (model, processor).  If *weights_path* is None or doesn't exist,
    the head will have random weights and a warning is logged.
    """
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    backbone = Dinov2Model.from_pretrained("facebook/dinov2-large")

    head = DINOv2ClassifierHead(in_dim=DINOV2_DIM)

    if weights_path:
        wp = Path(weights_path)
        if wp.exists():
            state = torch.load(wp, map_location="cpu", weights_only=True)
            head.load_state_dict(state)
            logger.info("Loaded DINOv2 head weights from %s", wp)
        else:
            logger.warning(
                "DINOv2 head weights not found at %s — running with untrained head. "
                "Results will be unreliable until you train and provide weights.",
                wp,
            )
    else:
        logger.warning(
            "No VERITAS_DINOV2_WEIGHTS path set — DINOv2 head is untrained. "
            "Set the env var or run training/train_image.py first."
        )

    model = DINOv2Classifier(backbone, head).to(device).eval()
    return model, processor


def predict_image(
    pil_img: Image.Image,
    model: DINOv2Classifier,
    processor: Any,
    device: torch.device,
) -> tuple[float, float]:
    """Classify a single image.

    Returns (p_fake, p_real).
    """
    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values)[0]  # (2,)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p_real = float(probs[0])
    p_fake = float(probs[1])
    return p_fake, p_real


def extract_cls_embedding(
    pil_img: Image.Image,
    model: DINOv2Classifier,
    processor: Any,
    device: torch.device,
) -> np.ndarray:
    """Extract the [CLS] embedding for a single image.

    Used by the video temporal transformer to build frame sequences.
    Returns: (1024,) numpy array.
    """
    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    cls_emb = model.extract_features(pixel_values)  # (1, 1024)
    return cls_emb[0].cpu().numpy()
