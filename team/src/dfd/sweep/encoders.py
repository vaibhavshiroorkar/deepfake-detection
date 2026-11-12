"""Frozen image encoders of the O-007 sweep (docs/O007_SWEEP.md section 4).

``load(cid)`` returns an ``Encoder`` whose ``__call__`` maps a uint8 RGB batch [B, 3, H, W] on the GPU to
a float feature [B, D]. Preprocessing is the one on each model card (size, mean, std), applied on the GPU to the
224 px face crops of dfd.prep (D-034); the feature is the standard pooled embedding before any head:
global-average pool for the CNNs, the DINOv2 pooler output (class token after the final norm), the projected image
embedding for CLIP, the attention-pooling output of the vision tower for SigLIP.
Weights are loaded from the pinned, hash-verified files written by scripts/sweep/fetch_encoders.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

LOG = Path("data/_logs/encoder_weights.json")

TIMM = {"IMG-01": "efficientnet_b4.ra2_in1k", "IMG-04": "legacy_xception"}
HF_KIND = {"IMG-07": "dinov2", "IMG-08": "dinov2", "IMG-13": "clip", "IMG-15": "clip", "IMG-17": "siglip"}
IMAGE_IDS = ["IMG-01", "IMG-04", "IMG-07", "IMG-08", "IMG-13", "IMG-15", "IMG-17"]


class HFImageProcessor:
    """The model card's image rule (transformers image processor). A class, not a lambda, so that DataLoader workers can pickle it."""

    def __init__(self, folder: str):
        self.folder, self._ip = folder, None

    def __call__(self, img):
        if self._ip is None:
            from transformers import AutoImageProcessor
            self._ip = AutoImageProcessor.from_pretrained(self.folder)
        return self._ip(images=img, return_tensors="pt")["pixel_values"][0]

    def __getstate__(self):
        return {"folder": self.folder, "_ip": None}


@dataclass
class Encoder:
    cid: str
    name: str
    model: torch.nn.Module
    size: int
    mean: tuple
    std: tuple
    fn: Callable
    dim: int
    n_params: int
    revision: str
    processor: Optional[Callable] = None   # PIL image -> normalised float tensor [3, H, W] by the model card's own rule (whole-image task)

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Features of already preprocessed (normalised) pixels, for example from `processor`."""
        dev = pixels.device.type
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            return self.fn(pixels).float()

    @torch.no_grad()
    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.train_features(x_uint8)

    def train_features(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """Same preprocessing and feature as __call__ but with gradients (fine-tuning and step-time measurement)."""
        x = x_uint8.float().div_(255.0)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=x.shape[-1] > self.size)
        m = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        dev = x.device.type
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            return self.fn((x - m) / s).float()


def _record(cid: str) -> dict:
    recs = {r["id"]: r for r in json.loads(LOG.read_text())}
    r = recs.get(cid)
    if r is None or not r["match"]:
        raise RuntimeError(f"{cid}: weights not downloaded or hash not verified (scripts/sweep/fetch_encoders.py)")
    return r


def load(cid: str, device: str = "cuda", timing_only: bool = False) -> Encoder:
    """timing_only=True builds a timm model with random weights (no verified file needed); never use it for features."""
    if timing_only and cid in TIMM:
        import timm
        m = timm.create_model(TIMM[cid], pretrained=False, num_classes=0).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        return Encoder(cid, TIMM[cid], m, cfg["input_size"][-1], tuple(cfg["mean"]), tuple(cfg["std"]), m, m.num_features,
                       sum(p.numel() for p in m.parameters()), "random-init (timing only)")
    rec = _record(cid)
    wfile = Path(rec["dir"]) / rec["file"]
    if cid in TIMM:
        import timm
        m = timm.create_model(TIMM[cid], pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=str(wfile))).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        return Encoder(cid, TIMM[cid], m, cfg["input_size"][-1], tuple(cfg["mean"]), tuple(cfg["std"]), m, m.num_features,
                       sum(p.numel() for p in m.parameters()), rec["revision"], timm.data.create_transform(**cfg))
    kind = HF_KIND[cid]
    d = rec["dir"]
    pp = json.loads((Path(d) / "preprocessor_config.json").read_text())
    mean, std = tuple(pp["image_mean"]), tuple(pp["image_std"])
    size = pp["size"] if isinstance(pp["size"], int) else pp["size"].get("height") or pp["size"].get("shortest_edge")
    if kind == "dinov2":
        from transformers import AutoModel
        m = AutoModel.from_pretrained(d, torch_dtype=torch.float32).to(device).eval()
        size = pp.get("crop_size", {}).get("height", 224)  # the 224 crop of the model card; our crops are already 224
        fn, dim, n = (lambda x: m(pixel_values=x).pooler_output), m.config.hidden_size, sum(p.numel() for p in m.parameters())
    elif kind == "clip":
        from transformers import CLIPModel
        full = CLIPModel.from_pretrained(d, torch_dtype=torch.float32).to(device).eval()
        m = full

        def fn(x):
            o = full.get_image_features(pixel_values=x)
            return o if torch.is_tensor(o) else o.pooler_output
        size = pp.get("crop_size", {}).get("height", 224) if isinstance(pp.get("crop_size"), dict) else 224
        dim = full.config.projection_dim
        n = sum(p.numel() for p in full.vision_model.parameters()) + sum(p.numel() for p in full.visual_projection.parameters())
    else:
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(d, torch_dtype=torch.float32).to(device).eval()
        m = full
        fn = lambda x: full.vision_model(pixel_values=x).pooler_output  # noqa: E731
        dim = full.config.vision_config.hidden_size
        n = sum(p.numel() for p in full.vision_model.parameters())
    return Encoder(cid, rec["repo"], m, int(size), mean, std, fn, int(dim), int(n), rec["revision"], HFImageProcessor(d))


# ---------------------------------------------------------------- audio encoders (E-003)
AUDIO_IDS = ["AU-1", "AU-4", "AU-6", "AU-8"]
AUDIO_KIND = {"AU-1": "ssl", "AU-4": "ssl", "AU-8": "ssl", "AU-6": "whisper"}
WINDOW = 64_600            # first 4 seconds at 16 kHz (the AASIST window), repeat-padded
WHISPER_FRAMES = -(-WINDOW // 320)   # encoder frames of 20 ms that cover the window: 202 (the rest of the 30 s input is padding)


@dataclass
class AudioEncoder:
    cid: str
    name: str
    model: torch.nn.Module
    fe: object
    kind: str
    dim: int          # per-frame hidden size; the stored feature is mean and standard deviation over time, 2 * dim
    n_params: int
    revision: str
    device: str

    @torch.no_grad()
    def __call__(self, waves) -> torch.Tensor:
        """waves: float32 array [B, WINDOW] at 16 kHz. Returns [B, 2 * dim]: mean and std over the frames of the last hidden layer."""
        arr = [w for w in waves]
        with torch.autocast(self.device, dtype=torch.float16, enabled=(self.device == "cuda")):
            if self.kind == "whisper":
                feats = self.fe(arr, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                h = self.model(feats).last_hidden_state[:, :WHISPER_FRAMES]       # only the frames of the 4 seconds are pooled
            else:
                x = self.fe(arr, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
                h = self.model(x).last_hidden_state
        h = h.float()
        return torch.cat([h.mean(1), h.std(1, unbiased=False)], dim=1)


def load_audio(cid: str, device: str = "cuda") -> AudioEncoder:
    rec = _record(cid)
    d = rec["dir"]
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(d)
    m = AutoModel.from_pretrained(d, dtype=torch.float32)
    kind = AUDIO_KIND[cid]
    if kind == "whisper":
        m = m.encoder
    m = m.to(device).eval()
    dim = m.config.d_model if kind == "whisper" else m.config.hidden_size
    return AudioEncoder(cid, rec["repo"], m, fe, kind, int(dim), sum(p.numel() for p in m.parameters()), rec["revision"], device)
