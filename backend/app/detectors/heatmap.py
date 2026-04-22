"""Visual evidence to accompany the numeric verdict. Each heatmap is
generated from the same data the corresponding signal scored, so what
the user sees on the overlay matches what the score is reading."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageChops


def _to_data_url(rgba: np.ndarray) -> str:
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _colorize(intensity: np.ndarray) -> np.ndarray:
    """Map a single-channel 0..1 array to an RGBA overlay using a warm ramp.
    Low values are transparent so the original image shows through."""
    intensity = np.clip(intensity, 0.0, 1.0)
    u8 = (intensity * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (np.power(intensity, 0.7) * 220).astype(np.uint8)
    rgba = np.dstack([color, alpha])
    return rgba


def ela_heatmap(pil_img: Image.Image, max_side: int = 720) -> str:
    """Bright pixels = regions whose recompression residue exceeds the rest
    of the frame. Same operation that drives `_ela_signal`."""
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = np.asarray(ImageChops.difference(rgb, recompressed), dtype=np.float32)
    intensity = diff.mean(axis=2)
    p99 = float(np.percentile(intensity, 99)) or 1.0
    intensity = intensity / max(p99, 1.0)
    return _to_data_url(_colorize(intensity))


def noise_heatmap(cv_img: np.ndarray, max_side: int = 720) -> str:
    """Per-patch high-frequency std. Patches darker than their neighbors
    indicate suspiciously smooth (likely synthesized) regions; very bright
    patches are unusual sensor noise."""
    bgr = cv_img
    h, w = bgr.shape[:2]
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    # Local std via box filter on hp^2 minus mean^2
    k = max(8, min(h, w) // 32)
    mean = cv2.boxFilter(hp, ddepth=-1, ksize=(k, k))
    sq = cv2.boxFilter(hp * hp, ddepth=-1, ksize=(k, k))
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    # Highlight deviation from the median noise level — both unusually
    # smooth and unusually noisy regions are interesting.
    med = float(np.median(local_std)) or 1.0
    intensity = np.abs(local_std - med) / max(med, 1.0)
    intensity = np.clip(intensity, 0.0, 1.5) / 1.5
    return _to_data_url(_colorize(intensity))
