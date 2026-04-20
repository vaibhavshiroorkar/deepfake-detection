"""
Image forensics. Heuristic signals combined into a verdict.

The analysis is explicitly multi-signal so the frontend can show its work:
ELA, noise consistency, frequency-domain artifacts, face region anomalies,
edge continuity. No single signal is decisive; the aggregate is.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags


@dataclass
class Signal:
    name: str
    score: float  # 0..1, higher = more suspicious
    detail: str


def _pil_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _cv_from_pil(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _ela_signal(img: Image.Image) -> Signal:
    """Error Level Analysis — recompress at quality 90 and measure per-pixel
    deviation. Authentic photographs degrade uniformly; composites and
    synthetic regions often leave bright ELA residue."""
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

    # Calibrated on sample set: mean ~4-8 is typical for real JPEGs,
    # >14 is suspicious. Composites push peaks aggressively.
    score = 0.0
    score += min(1.0, max(0.0, (mean - 5.0) / 14.0)) * 0.5
    score += min(1.0, max(0.0, (peak - 30.0) / 60.0)) * 0.35
    score += min(1.0, max(0.0, (std - 8.0) / 18.0)) * 0.15
    score = float(np.clip(score, 0.0, 1.0))

    return Signal(
        name="Error-level analysis",
        score=score,
        detail=f"Residual mean {mean:.2f}, 99th-percentile {peak:.1f}. "
        + (
            "Recompression residue is uniform — consistent with a single-origin photograph."
            if score < 0.35
            else "Residue clusters suggest regions compressed more than once or synthesized."
        ),
    )


def _noise_signal(cv_img: np.ndarray) -> Signal:
    """Local noise variance. Real sensor noise is spatially stationary;
    synthetic imagery often has patches of unnaturally smooth or
    unnaturally uniform noise."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    step = max(16, min(h, w) // 24)
    patches = []
    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = gray[y : y + step, x : x + step]
            hp = patch - cv2.GaussianBlur(patch, (5, 5), 0)
            patches.append(hp.std())
    patches = np.array(patches) if patches else np.array([0.0])
    variance_of_variance = float(patches.std())
    median_noise = float(np.median(patches))

    score = 0.0
    # Very low median noise (over-smoothed) OR very high variance of
    # variance (uneven noise floor) both suggest manipulation.
    if median_noise < 1.2:
        score += 0.55
    elif median_noise < 2.0:
        score += 0.2
    score += min(1.0, variance_of_variance / 6.0) * 0.45
    score = float(np.clip(score, 0.0, 1.0))

    return Signal(
        name="Noise stationarity",
        score=score,
        detail=f"Median high-frequency noise {median_noise:.2f}, cross-patch variance {variance_of_variance:.2f}. "
        + (
            "Noise floor is consistent across the frame."
            if score < 0.4
            else "Uneven noise distribution — a common fingerprint of inpainted or generated regions."
        ),
    )


def _frequency_signal(cv_img: np.ndarray) -> Signal:
    """High-frequency spectral signature. Many GAN and diffusion outputs
    have measurably different radial spectra than camera captures."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (256, 256))
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = 128, 128
    y, x = np.indices(mag.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.bincount(r.ravel())
    radial = radial[: min(128, len(radial))]

    # Slope of the spectrum in mid-to-high band. Flatter than expected
    # can indicate upsampled or synthesized content.
    lo, hi = 20, 100
    if hi >= len(radial):
        hi = len(radial) - 1
    band = radial[lo:hi]
    if len(band) < 4:
        return Signal("Spectral falloff", 0.2, "Image too small for spectral analysis.")
    slope = float(np.polyfit(np.arange(len(band)), band, 1)[0])

    # Typical cameras: slope around -0.02 to -0.05. Flat or positive is suspicious.
    score = float(np.clip((slope + 0.01) / 0.04 + 0.4, 0.0, 1.0))

    return Signal(
        name="Spectral falloff",
        score=score,
        detail=f"Radial spectrum slope {slope:.4f}. "
        + (
            "Spectrum falls off as expected for a lensed capture."
            if score < 0.45
            else "Mid-band energy is elevated — often a signature of upsampled or diffused output."
        ),
    )


def _face_signal(cv_img: np.ndarray) -> Signal:
    """Face region inspection: detect faces and measure edge
    discontinuity around them (swap/morph boundary heuristic)."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))

    if len(faces) == 0:
        return Signal(
            name="Face boundary",
            score=0.0,
            detail="No faces detected — this check does not apply.",
        )

    scores = []
    for (x, y, w, h) in faces:
        pad = max(8, w // 10)
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        region = gray[y0:y1, x0:x1]
        edges_inner = cv2.Canny(gray[y:y + h, x:x + w], 60, 160).mean()
        edges_outer = cv2.Canny(region, 60, 160).mean()
        # A face swapped onto a body often has a halo of edges at the boundary.
        delta = abs(edges_outer - edges_inner)
        scores.append(min(1.0, delta / 18.0))
    avg = float(np.mean(scores))
    return Signal(
        name="Face boundary",
        score=avg,
        detail=f"{len(faces)} face region(s) examined. Boundary edge delta averaged {avg:.2f}. "
        + (
            "Transitions into skin and hair look continuous."
            if avg < 0.4
            else "Halo or seam artifacts near face boundaries — consistent with face-swap techniques."
        ),
    )


def _metadata_signal(img: Image.Image) -> Signal:
    """EXIF check. Absent camera metadata isn't proof of manipulation,
    but its presence is a mild signal in the other direction."""
    exif = getattr(img, "_getexif", lambda: None)()
    if not exif:
        return Signal(
            name="Capture metadata",
            score=0.35,
            detail="No EXIF metadata present. Common for screenshots, web-exported images, and — sometimes — generated output.",
        )
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    make = str(tags.get("Make", "")).strip()
    model = str(tags.get("Model", "")).strip()
    software = str(tags.get("Software", "")).lower()
    datetime = str(tags.get("DateTimeOriginal", tags.get("DateTime", "")))

    suspicious_software = any(
        s in software for s in ["stable diffusion", "midjourney", "dall", "firefly", "generated"]
    )
    if suspicious_software:
        return Signal(
            name="Capture metadata",
            score=0.9,
            detail=f"Software tag names a generative tool: '{software}'.",
        )
    if make and model:
        return Signal(
            name="Capture metadata",
            score=0.1,
            detail=f"Camera tags present: {make} {model}{(' @ ' + datetime) if datetime else ''}.",
        )
    return Signal(
        name="Capture metadata",
        score=0.3,
        detail="Partial EXIF data — consistent with re-saved or edited images.",
    )


def _verdict(signals: list[Signal]) -> tuple[float, str]:
    """Weighted aggregation. Signals at the extremes dominate."""
    weights = {
        "Error-level analysis": 1.0,
        "Noise stationarity": 1.0,
        "Spectral falloff": 0.9,
        "Face boundary": 0.8,
        "Capture metadata": 0.5,
    }
    num = 0.0
    denom = 0.0
    for s in signals:
        if s.name == "Face boundary" and s.score == 0.0 and "does not apply" in s.detail:
            continue
        w = weights.get(s.name, 0.5)
        num += s.score * w
        denom += w
    score = num / denom if denom else 0.0
    # Nudge: images with one very high signal should not be washed out.
    peak = max((s.score for s in signals), default=0.0)
    score = 0.7 * score + 0.3 * peak

    if score < 0.3:
        label = "likely authentic"
    elif score < 0.55:
        label = "inconclusive"
    elif score < 0.75:
        label = "likely manipulated"
    else:
        label = "highly likely manipulated"
    return float(np.clip(score, 0.0, 1.0)), label


def analyze_image(data: bytes, filename: str = "image") -> dict[str, Any]:
    img = _pil_from_bytes(data)
    cv_img = _cv_from_pil(img)

    signals = [
        _ela_signal(img),
        _noise_signal(cv_img),
        _frequency_signal(cv_img),
        _face_signal(cv_img),
        _metadata_signal(img),
    ]
    score, label = _verdict(signals)
    width, height = img.size

    return {
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
