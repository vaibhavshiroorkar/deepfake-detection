"""
Image forensics: detecting AI-generated and manipulated images.

Two distinct problems require two distinct signal sets:
  1. AI synthesis (GAN / diffusion) — no compositing artifacts, but missing
     the physical imperfections every real camera introduces.
  2. Traditional manipulation (splicing, inpainting) — ELA and face seam
     signals catch these.

Signals are combined with weights tuned to catch modern generators while
keeping false-positive rates reasonable on unedited photographs.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags

from . import c2pa as _c2pa
from .heatmap import ela_heatmap, noise_heatmap


@dataclass
class Signal:
    name: str
    score: float  # 0..1, higher = more suspicious
    detail: str


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _pil_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _cv_from_pil(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Signal 1 — Error Level Analysis (catches splicing; AI images show flat ELA)
# ---------------------------------------------------------------------------

def _ela_signal(img: Image.Image) -> Signal:
    """ELA reveals JPEG re-compression history.
    Composited regions show high residue. AI-generated images show suspiciously
    flat, low residue because they were synthesised in one pass.
    Both extremes are penalised."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(rgb, recompressed)
    arr = np.asarray(diff, dtype=np.float32)
    mean = float(arr.mean())
    peak = float(np.percentile(arr, 99))
    std = float(arr.std())

    score = 0.0
    # High residue → composite / manipulation
    score += min(1.0, max(0.0, (mean - 6.0) / 12.0)) * 0.45
    score += min(1.0, max(0.0, (peak - 35.0) / 55.0)) * 0.30

    # Suspiciously flat ELA → AI synthesis. Real JPEGs have mean 4-8 and
    # meaningful std. Generated images are too uniform.
    if mean < 3.5 and std < 3.0:
        score += 0.35
    elif mean < 5.0 and std < 4.5:
        score += 0.15

    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.35:
        detail = (f"Residual mean {mean:.2f}, 99th-pct {peak:.1f}. "
                  "Recompression residue is normal for a camera photograph.")
    elif mean < 5.0:
        detail = (f"Residual mean {mean:.2f}, std {std:.2f}. "
                  "ELA is suspiciously flat and clean — consistent with AI synthesis.")
    else:
        detail = (f"Residual mean {mean:.2f}, 99th-pct {peak:.1f}. "
                  "Residue clusters suggest re-compressed or composited regions.")
    return Signal("Error-level analysis", score, detail)


# ---------------------------------------------------------------------------
# Signal 2 — Focus / depth-of-field uniformity
# ---------------------------------------------------------------------------

def _focus_uniformity_signal(cv_img: np.ndarray) -> Signal:
    """Real lenses produce depth-of-field blur: sharpness varies across the
    frame. AI-generated images are uniformly sharp everywhere — no part is
    defocused, even when perspective demands it.

    Measures coefficient of variation of patch-level Laplacian variance.
    Low CV → suspiciously uniform focus → AI."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    step = max(32, min(h, w) // 10)
    sharpness = []
    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = gray[y : y + step, x : x + step]
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            v = float(lap.var())
            if v > 0:
                sharpness.append(v)

    if len(sharpness) < 6:
        return Signal("Focus variation", 0.2,
                      "Image too small to profile depth of field.")

    sharpness = np.array(sharpness)
    cv = float(sharpness.std() / (sharpness.mean() + 1e-6))

    # Natural photos: CV typically 1.0–3.0 (some areas sharp, some blurry).
    # AI images: CV often < 0.5 (uniform sharpness throughout).
    if cv < 0.35:
        score = 0.85
    elif cv < 0.6:
        score = 0.65
    elif cv < 1.0:
        score = 0.35
    else:
        score = 0.1

    return Signal(
        "Focus variation",
        score,
        f"Sharpness CV across {len(sharpness)} patches: {cv:.2f}. "
        + ("Unnaturally uniform sharpness — no depth-of-field blur. "
           "Real lenses can't keep every plane in focus simultaneously."
           if score > 0.5
           else "Sharpness varies naturally across the frame."),
    )


# ---------------------------------------------------------------------------
# Signal 3 — Chromatic aberration
# ---------------------------------------------------------------------------

def _chromatic_aberration_signal(cv_img: np.ndarray) -> Signal:
    """Every real lens produces chromatic aberration: colour channels shift
    slightly at high-contrast edges. AI models generate pixel-perfect channel
    alignment — a physical impossibility with glass optics.

    Measures the spatial correlation between channel-specific edge maps.
    Perfect correlation → no CA → suspicious."""
    b, g, r = [c.astype(np.uint8) for c in cv2.split(cv_img)]
    # Use Sobel magnitude instead of Canny for smoother gradient comparison
    def sobel_mag(ch: np.ndarray) -> np.ndarray:
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(sx * sx + sy * sy)

    mag_r = sobel_mag(r).ravel()
    mag_g = sobel_mag(g).ravel()
    mag_b = sobel_mag(b).ravel()

    # Only compare at strong-edge positions (top 20 % by green gradient)
    thresh = float(np.percentile(mag_g, 80))
    mask = mag_g > thresh
    if mask.sum() < 500:
        return Signal("Chromatic aberration", 0.3,
                      "Insufficient edge pixels to measure CA.")

    rg_corr = float(np.corrcoef(mag_r[mask], mag_g[mask])[0, 1])
    bg_corr = float(np.corrcoef(mag_b[mask], mag_g[mask])[0, 1])
    alignment = float(np.clip((rg_corr + bg_corr) / 2, 0.0, 1.0))

    # Real lenses: alignment ~0.85–0.95 (slight channel shift).
    # AI images: alignment ~0.98–1.0 (channels move in perfect lockstep).
    if alignment > 0.985:
        score = 0.85
    elif alignment > 0.97:
        score = 0.60
    elif alignment > 0.95:
        score = 0.35
    else:
        score = 0.1

    return Signal(
        "Chromatic aberration",
        score,
        f"Channel gradient alignment {alignment:.4f}. "
        + ("Channels are in perfect alignment — glass lenses can't do this. "
           "Consistent with AI synthesis."
           if score > 0.5
           else "Natural colour-channel misalignment from lens optics."),
    )


# ---------------------------------------------------------------------------
# Signal 4 — Frequency-domain grid artifacts
# ---------------------------------------------------------------------------

def _grid_artifact_signal(cv_img: np.ndarray) -> Signal:
    """Diffusion models process images in latent patches (commonly 8 × 8 or
    16 × 16 pixels in pixel space after upsampling). This leaves periodic
    energy peaks in the 2-D FFT at those spatial frequencies.

    Real photographs have smooth, monotonically decaying radial spectra."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = 512
    gray = cv2.resize(gray, (size, size))
    gray -= gray.mean()

    fft = np.fft.fft2(gray)
    fshift = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fshift))

    cy, cx = size // 2, size // 2
    # Blank the DC component
    mag[cy - 4 : cy + 5, cx - 4 : cx + 5] = 0.0

    # Build a smooth radial baseline
    y_idx, x_idx = np.indices(mag.shape)
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    radial_counts = np.bincount(r.ravel().astype(int), minlength=size)
    radial_sums = np.bincount(r.ravel().astype(int), mag.ravel(), minlength=size)
    with np.errstate(invalid="ignore"):
        radial_mean = np.where(radial_counts > 0,
                               radial_sums / radial_counts, 0.0)

    # Expected value at each pixel from radial average
    expected = radial_mean[np.clip(r.astype(int), 0, size - 1)]
    residual = mag - expected  # positive = unexplained peak

    peak_score = 0.0
    for period in [8, 12, 16, 32, 64]:
        freq = size // period
        if freq < 3:
            continue
        for dy, dx in [(freq, 0), (0, freq), (freq, freq),
                       (-freq, freq), (freq, -freq)]:
            py, px = cy + dy, cx + dx
            if 0 < py < size and 0 < px < size:
                local_peak = residual[py - 2 : py + 3, px - 2 : px + 3].max()
                if local_peak > 0.15:
                    peak_score += local_peak * 0.4

    score = float(np.clip(peak_score, 0.0, 1.0))
    return Signal(
        "Frequency grid artifacts",
        score,
        f"Peak residual at patch-boundary frequencies: {peak_score:.3f}. "
        + ("Periodic energy peaks detected — characteristic of diffusion-model "
           "patch processing or GAN upsampling."
           if score > 0.3
           else "No significant periodic artifacts in the frequency spectrum."),
    )


# ---------------------------------------------------------------------------
# Signal 5 — Sensor noise profile
# ---------------------------------------------------------------------------

def _sensor_noise_signal(cv_img: np.ndarray) -> Signal:
    """Real camera sensors inject thermal and shot noise uniformly, even in
    flat regions. AI-generated images either omit noise entirely or add a
    synthetic pattern that doesn't match real sensor statistics.

    Measures noise level in smooth image regions; very low or perfectly
    uniform noise is suspicious."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    step = max(16, min(h, w) // 20)
    smooth_patch_noise = []

    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = gray[y : y + step, x : x + step]
            # High-pass: remove structure, keep noise
            hp = patch - cv2.GaussianBlur(patch, (5, 5), 0)
            local_grad = cv2.Laplacian(patch, cv2.CV_32F).var()
            # Only use smooth patches (little scene texture)
            if local_grad < 200:
                smooth_patch_noise.append(float(hp.std()))

    if len(smooth_patch_noise) < 5:
        return Signal("Sensor noise", 0.25,
                      "Too few smooth regions to profile sensor noise.")

    noise_arr = np.array(smooth_patch_noise)
    noise_level = float(noise_arr.mean())
    noise_consistency = float(noise_arr.std())

    score = 0.0
    # Too clean: AI images often have near-zero noise in flat areas
    if noise_level < 0.4:
        score += 0.7
    elif noise_level < 0.9:
        score += 0.4
    elif noise_level < 1.5:
        score += 0.15

    # Suspicious uniformity of the noise itself
    if noise_consistency < 0.1 and noise_level < 2.0:
        score += 0.2

    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Sensor noise",
        score,
        f"Smooth-region noise level {noise_level:.3f}, consistency {noise_consistency:.3f}. "
        + ("Suspiciously clean or uniform noise — real sensors always add "
           "thermal and shot noise."
           if score > 0.4
           else "Natural sensor noise present in flat image regions."),
    )


# ---------------------------------------------------------------------------
# Signal 6 — Face-region boundary (composite / face-swap heuristic)
# ---------------------------------------------------------------------------

def _face_signal(cv_img: np.ndarray) -> Signal:
    """Detects seam artifacts at face boundaries, the hallmark of face-swap
    and face-replacement composites. Pure AI generation usually passes this;
    it is most useful for GAN-based deepfakes and Photoshop splicing."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 80:
        return Signal("Face boundary", 0.0,
                      "Image too small for face analysis.")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    try:
        faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
    except cv2.error:
        faces = []
    if len(faces) == 0:
        return Signal("Face boundary", 0.0,
                      "No faces detected — this check does not apply.")
    scores = []
    for (x, y, fw, fh) in faces:
        pad = max(8, fw // 10)
        y0 = max(0, y - pad); y1 = min(gray.shape[0], y + fh + pad)
        x0 = max(0, x - pad); x1 = min(gray.shape[1], x + fw + pad)
        region = gray[y0:y1, x0:x1]
        inner = cv2.Canny(gray[y : y + fh, x : x + fw], 60, 160).mean()
        outer = cv2.Canny(region, 60, 160).mean()
        scores.append(min(1.0, abs(outer - inner) / 18.0))
    avg = float(np.mean(scores))
    return Signal(
        "Face boundary",
        avg,
        f"{len(faces)} face region(s). Boundary edge delta: {avg:.2f}. "
        + ("Halo or seam at face boundary — consistent with face-swap."
           if avg > 0.4
           else "No seam artifacts at face boundaries."),
    )


# ---------------------------------------------------------------------------
# Signal 7 — EXIF / capture metadata
# ---------------------------------------------------------------------------

def _metadata_signal(img: Image.Image) -> Signal:
    try:
        exif_obj = img.getexif()
        exif = dict(exif_obj) if exif_obj else None
    except Exception:
        exif = None
    if not exif:
        return Signal(
            "Capture metadata",
            0.45,
            "No EXIF metadata. AI-generated images never carry camera EXIF; "
            "real photographs almost always do.",
        )
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    make = str(tags.get("Make", "")).strip()
    model_name = str(tags.get("Model", "")).strip()
    software = str(tags.get("Software", "")).lower()
    dt = str(tags.get("DateTimeOriginal", tags.get("DateTime", "")))

    if any(s in software for s in
           ["stable diffusion", "midjourney", "dall", "firefly",
            "generated", "invoke", "comfy", "automatic1111"]):
        return Signal("Capture metadata", 0.95,
                      f"Software tag identifies a generative tool: '{software}'.")
    if make and model_name:
        return Signal(
            "Capture metadata", 0.05,
            f"Camera: {make} {model_name}{' @ ' + dt if dt else ''}.",
        )
    return Signal("Capture metadata", 0.30,
                  "Partial EXIF — consistent with re-saved or screen-captured images.")


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------

def _verdict(signals: list[Signal]) -> tuple[float, str]:
    weights = {
        "Focus variation":          1.4,
        "Chromatic aberration":     1.3,
        "Sensor noise":             1.2,
        "Frequency grid artifacts": 1.1,
        "Error-level analysis":     1.0,
        "Capture metadata":         0.8,
        "Face boundary":            0.7,
    }
    num = denom = 0.0
    for s in signals:
        if "does not apply" in s.detail:
            continue
        w = weights.get(s.name, 0.5)
        num += s.score * w
        denom += w
    avg = num / denom if denom else 0.0
    peak = max((s.score for s in signals), default=0.0)
    score = float(np.clip(0.65 * avg + 0.35 * peak, 0.0, 1.0))

    if score < 0.30:
        label = "likely authentic"
    elif score < 0.50:
        label = "inconclusive"
    elif score < 0.72:
        label = "likely AI-generated or manipulated"
    else:
        label = "highly likely AI-generated or manipulated"
    return score, label


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_image(
    data: bytes,
    filename: str = "image",
    *,
    with_heatmaps: bool = True,
) -> dict[str, Any]:
    img = _pil_from_bytes(data)
    cv_img = _cv_from_pil(img)

    signals = [
        _focus_uniformity_signal(cv_img),
        _chromatic_aberration_signal(cv_img),
        _sensor_noise_signal(cv_img),
        _grid_artifact_signal(cv_img),
        _ela_signal(img),
        _metadata_signal(img),
        _face_signal(cv_img),
    ]
    score, label = _verdict(signals)
    width, height = img.size

    out: dict[str, Any] = {
        "kind": "image",
        "filename": filename,
        "dimensions": {"width": width, "height": height},
        "suspicion": score,
        "verdict": label,
        "confidence": round(abs(score - 0.5) * 2, 3),
        "signals": [
            {"name": s.name, "score": round(s.score, 3), "detail": s.detail}
            for s in signals
        ],
    }

    if with_heatmaps:
        try:
            out["heatmaps"] = {
                "ela": ela_heatmap(img),
                "noise": noise_heatmap(cv_img),
            }
        except Exception:
            out["heatmaps"] = None

    manifest = _c2pa.read_manifest(data) if _c2pa.is_available() else None
    if manifest is not None:
        out["c2pa"] = manifest

    return out
