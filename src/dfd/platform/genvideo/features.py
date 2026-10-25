"""ReStraV-style trajectory features on frozen DINOv2-small (adaptation, see docs/GENERATED_VIDEO_PLAN.md section 2.2 and E-020).

Reproduced from the description of ReStraV (Interno et al., arXiv 2507.00583, R-078, partly read): frozen DINOv2 ViT-S/14, each of 8 frames of a 2 s
window gives one embedding (the last-layer tokens flattened), 7 stepwise L2 distances and 6 curvature angles (degrees, arccos of the cosine
similarity of consecutive difference vectors), plus mean, variance, min and max of each signal: a 21-value vector.
Adaptations (NOT identical to the published pipeline): the 8 frames are picked from the 14 frames of the canonical clip (7 fps, 2.0 s) instead of
24 decoded frames; the embedding is HF `last_hidden_state` (257 tokens x 384, after the model's final layer norm) in fp32; the classifier and its
training data are ours (E-020). Which tokens the authors flatten is ambiguous in the text read (R-078) and was not checked in their code.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .canonical import N_FRAMES

PICK = np.linspace(0, N_FRAMES - 1, 8).round().astype(int)       # [0 2 4 6 7 9 11 13]
ENCODER_ID = "IMG-07"
REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHT_LOG = REPO_ROOT / "data" / "_logs" / "encoder_weights.json"
FEATURE_NAMES = ([f"dist{i}" for i in range(7)] + [f"curv{i}" for i in range(6)] +
                 [f"{s}_{k}" for s in ("dist", "curv") for k in ("mean", "var", "min", "max")])


def trajectory_features(emb: np.ndarray) -> np.ndarray:
    """emb (8, D) float -> 21 values."""
    e = np.asarray(emb, dtype=np.float64)
    v = np.diff(e, axis=0)                                           # 7 difference vectors
    d = np.linalg.norm(v, axis=1)
    cos = np.einsum("ij,ij->i", v[:-1], v[1:]) / np.maximum(d[:-1] * d[1:], 1e-12)
    th = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))             # 6 angles

    def summ(x):
        return [x.mean(), x.var(), x.min(), x.max()]
    return np.concatenate([d, th, summ(d), summ(th)]).astype(np.float32)


class DinoRunner:
    """Lazy, thread-safe DINOv2-small runner using the hash-verified local copy recorded in data/_logs/encoder_weights.json."""

    def __init__(self, device=None):
        self._device, self._model, self._lock = device, None, threading.Lock()
        self.revision = None

    def _load(self):
        import torch
        from transformers import AutoModel
        rec = next(r for r in json.loads(WEIGHT_LOG.read_text()) if r["id"] == ENCODER_ID)
        if not rec["match"]:
            raise RuntimeError("DINOv2-small weights not hash-verified (scripts/sweep/fetch_encoders.py)")
        d = rec["dir"]
        pp = json.loads((Path(d) / "preprocessor_config.json").read_text())
        self.mean = np.array(pp["image_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(pp["image_std"], dtype=np.float32).reshape(1, 3, 1, 1)
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(d, torch_dtype=torch.float32).to(dev).eval()
        self._dev, self.revision = dev, rec["revision"]

    def embed(self, frames: np.ndarray):
        """frames (n, 224, 224, 3) uint8 -> ((n, 257 * 384) float32 flattened last_hidden_state, (n, 384) CLS token)."""
        import torch
        with self._lock:
            if self._model is None:
                self._load()
            x = (frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - self.mean) / self.std
            with torch.no_grad():
                h = self._model(pixel_values=torch.from_numpy(x).to(self._dev)).last_hidden_state.float().cpu().numpy()
        return h.reshape(h.shape[0], -1), h[:, 0, :]

    def features(self, frames14: np.ndarray) -> dict:
        flat, cls = self.embed(frames14[PICK])
        return {"restrav21": trajectory_features(flat), "cls_mean": cls.mean(axis=0).astype(np.float32), "cls_all": cls.astype(np.float32)}


_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()


def shared_runner() -> DinoRunner:
    """One DINOv2-small per process, so a new module instance per request does not reload the model."""
    with _SHARED_LOCK:
        if "r" not in _SHARED:
            _SHARED["r"] = DinoRunner()
        return _SHARED["r"]
