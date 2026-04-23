"""
Image forensics — Phase 2.

Primary signal: **DINOv2-large** with a trained classification head,
fine-tuned on FaceForensics++ and Celeb-DF datasets.  DINOv2 produces
exceptionally rich visual features from its self-supervised pretraining,
and the classification head maps them to a calibrated P(AI-generated).

Fallback: if DINOv2 head weights are not trained yet, the system
falls back to the existing Swin-v2 classifier (Organika/sdxl-detector).

MTCNN crops faces and the classifier is run over each crop separately
to surface face-swap artefacts.

Classical signals (focus uniformity, chromatic aberration, sensor noise,
ELA, EXIF) are kept as supporting evidence with gated weights.
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


def _exif_tags(img: Image.Image) -> dict:
    try:
        raw = dict(img.getexif() or {})
    except Exception:
        return {}
    return {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}


def _is_camera_origin(tags: dict) -> bool:
    """True if EXIF identifies a physical camera (Make + Model present)
    and the Software field is not a known generator."""
    make = str(tags.get("Make", "")).strip()
    model_name = str(tags.get("Model", "")).strip()
    software = str(tags.get("Software", "")).lower()
    if any(s in software for s in
           ("stable diffusion", "midjourney", "dall", "firefly",
            "generated", "invoke", "comfy", "automatic1111")):
        return False
    return bool(make and model_name)


# ---------------------------------------------------------------------------
# Primary signal — DINOv2 classifier (Phase 2), Swin-v2 fallback
# ---------------------------------------------------------------------------

def _classify_ai_dinov2(pil_img: Image.Image) -> tuple[float, dict] | None:
    """Try DINOv2 classifier.  Returns None if unavailable."""
    try:
        from ..models import get_dinov2_classifier, get_device
        from .dinov2_head import predict_image

        model, processor = get_dinov2_classifier()
        device = get_device()
        p_fake, p_real = predict_image(pil_img, model, processor, device)
        return p_fake, {
            "p_ai": p_fake,
            "p_real": p_real,
            "model": "DINOv2-large + trained head",
        }
    except Exception:
        return None


def _classify_ai_swinv2(pil_img: Image.Image) -> tuple[float, dict]:
    """Swin-v2 fallback classifier."""
    import torch
    from ..models import get_ai_image_classifier, get_device

    model, processor, lmap = get_ai_image_classifier()
    device = get_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    p_ai = float(probs[lmap["ai_index"]])
    p_real = float(probs[lmap["real_index"]])
    return p_ai, {"p_ai": p_ai, "p_real": p_real, "model": lmap["name"]}


def _classify_ai(pil_img: Image.Image) -> tuple[float, dict]:
    """Returns (P(AI), info_dict).  Tries DINOv2 first, falls back to Swin-v2."""
    result = _classify_ai_dinov2(pil_img)
    if result is not None:
        return result
    return _classify_ai_swinv2(pil_img)


def _ai_classifier_signal(pil_img: Image.Image) -> Signal:
    try:
        p_ai, info = _classify_ai(pil_img)
    except Exception as exc:  # noqa: BLE001
        return Signal(
            "AI-image classifier",
            0.0,
            f"Pretrained classifier unavailable ({type(exc).__name__}); skipping.",
        )
    model_name = info.get("model", "unknown")
    detail = (
        f"{model_name}: P(AI) = {p_ai:.3f}, P(real) = {info['p_real']:.3f}. "
        + ("Classifier is confident the image is AI-generated."
           if p_ai > 0.75 else
           "Classifier leans toward AI-generated."
           if p_ai > 0.55 else
           "Classifier leans toward authentic."
           if p_ai < 0.45 else
           "Classifier is uncertain.")
    )
    return Signal("AI-image classifier", p_ai, detail)


# ---------------------------------------------------------------------------
# Face detection (MTCNN) + per-face classifier rescoring
# ---------------------------------------------------------------------------

def _detect_faces_mtcnn(pil_img: Image.Image) -> list[tuple[int, int, int, int, float]]:
    try:
        from ..models import get_face_detector
        detector = get_face_detector()
        boxes, probs = detector.detect(pil_img)
    except Exception:
        return []
    if boxes is None:
        return []
    out: list[tuple[int, int, int, int, float]] = []
    w, h = pil_img.size
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.85:
            continue
        x1, y1, x2, y2 = box
        x1 = int(max(0, x1)); y1 = int(max(0, y1))
        x2 = int(min(w, x2)); y2 = int(min(h, y2))
        if x2 - x1 < 30 or y2 - y1 < 30:
            continue
        out.append((x1, y1, x2, y2, float(prob)))
    return out


def _face_classifier_signal(
    pil_img: Image.Image,
    faces: list[tuple[int, int, int, int, float]],
) -> Signal:
    if not faces:
        return Signal("Face classifier", 0.0,
                      "No faces detected — this check does not apply.")
    per_face = []
    for (x1, y1, x2, y2, _p) in faces:
        crop = pil_img.crop((x1, y1, x2, y2))
        try:
            p_ai, _ = _classify_ai(crop)
        except Exception as exc:  # noqa: BLE001
            return Signal(
                "Face classifier",
                0.0,
                f"Pretrained classifier unavailable for face crops ({type(exc).__name__}).",
            )
        per_face.append(p_ai)
    avg = float(np.mean(per_face))
    peak = float(np.max(per_face))
    score = float(np.clip(0.5 * avg + 0.5 * peak, 0.0, 1.0))
    return Signal(
        "Face classifier",
        score,
        f"{len(faces)} face crop(s) classified. avg P(AI)={avg:.2f}, peak={peak:.2f}. "
        + ("At least one face crop scores as synthetic."
           if score > 0.55
           else "Face crops look authentic."),
    )


def _face_boundary_signal(
    cv_img: np.ndarray,
    faces: list[tuple[int, int, int, int, float]],
) -> Signal:
    if not faces:
        return Signal("Face boundary", 0.0,
                      "No faces detected — this check does not apply.")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scores = []
    for (x1, y1, x2, y2, _p) in faces:
        fw = x2 - x1
        pad = max(8, fw // 10)
        ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
        ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
        inner = cv2.Canny(gray[y1:y2, x1:x2], 60, 160).mean()
        outer = cv2.Canny(gray[oy1:oy2, ox1:ox2], 60, 160).mean()
        scores.append(min(1.0, abs(outer - inner) / 18.0))
    avg = float(np.mean(scores))
    return Signal(
        "Face boundary",
        avg,
        f"{len(faces)} face region(s) via MTCNN. Boundary edge delta: {avg:.2f}. "
        + ("Halo or seam at face boundary — consistent with face-swap."
           if avg > 0.4
           else "No seam artifacts at face boundaries."),
    )


# ---------------------------------------------------------------------------
# Classical signals (kept as supporting evidence, gated by EXIF)
# ---------------------------------------------------------------------------

def _ela_signal(img: Image.Image) -> Signal:
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
    score += min(1.0, max(0.0, (mean - 6.0) / 12.0)) * 0.45
    score += min(1.0, max(0.0, (peak - 35.0) / 55.0)) * 0.30
    if mean < 3.5 and std < 3.0:
        score += 0.20
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.35:
        detail = (f"Residual mean {mean:.2f}, 99th-pct {peak:.1f}. "
                  "Recompression residue is normal.")
    elif mean < 5.0:
        detail = (f"Residual mean {mean:.2f}, std {std:.2f}. "
                  "ELA is suspiciously flat and clean.")
    else:
        detail = (f"Residual mean {mean:.2f}, 99th-pct {peak:.1f}. "
                  "Residue clusters suggest re-compressed or composited regions.")
    return Signal("Error-level analysis", score, detail)


def _focus_uniformity_signal(cv_img: np.ndarray) -> Signal:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    step = max(32, min(h, w) // 10)
    sharpness = []
    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = gray[y : y + step, x : x + step]
            v = float(cv2.Laplacian(patch, cv2.CV_32F).var())
            if v > 0:
                sharpness.append(v)

    if len(sharpness) < 6:
        return Signal("Focus variation", 0.2,
                      "Image too small to profile depth of field.")

    sharpness = np.array(sharpness)
    cv = float(sharpness.std() / (sharpness.mean() + 1e-6))

    if cv < 0.35:
        score = 0.75
    elif cv < 0.6:
        score = 0.50
    elif cv < 1.0:
        score = 0.25
    else:
        score = 0.05

    return Signal(
        "Focus variation",
        score,
        f"Sharpness CV across {len(sharpness)} patches: {cv:.2f}. "
        + ("Unnaturally uniform sharpness — no depth-of-field blur."
           if score > 0.5
           else "Sharpness varies naturally across the frame."),
    )


def _chromatic_aberration_signal(cv_img: np.ndarray) -> Signal:
    b, g, r = [c.astype(np.uint8) for c in cv2.split(cv_img)]

    def sobel_mag(ch: np.ndarray) -> np.ndarray:
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(sx * sx + sy * sy)

    mag_r = sobel_mag(r).ravel()
    mag_g = sobel_mag(g).ravel()
    mag_b = sobel_mag(b).ravel()

    thresh = float(np.percentile(mag_g, 80))
    mask = mag_g > thresh
    if mask.sum() < 500:
        return Signal("Chromatic aberration", 0.2,
                      "Insufficient edge pixels to measure CA.")

    rg_corr = float(np.corrcoef(mag_r[mask], mag_g[mask])[0, 1])
    bg_corr = float(np.corrcoef(mag_b[mask], mag_g[mask])[0, 1])
    alignment = float(np.clip((rg_corr + bg_corr) / 2, 0.0, 1.0))

    if alignment > 0.985:
        score = 0.70
    elif alignment > 0.97:
        score = 0.45
    elif alignment > 0.95:
        score = 0.25
    else:
        score = 0.05

    return Signal(
        "Chromatic aberration",
        score,
        f"Channel gradient alignment {alignment:.4f}. "
        + ("Channels in near-perfect alignment — atypical for glass optics."
           if score > 0.5
           else "Natural colour-channel misalignment from lens optics."),
    )


def _sensor_noise_signal(cv_img: np.ndarray) -> Signal:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    step = max(16, min(h, w) // 20)
    smooth_patch_noise = []

    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = gray[y : y + step, x : x + step]
            hp = patch - cv2.GaussianBlur(patch, (5, 5), 0)
            local_grad = cv2.Laplacian(patch, cv2.CV_32F).var()
            if local_grad < 200:
                smooth_patch_noise.append(float(hp.std()))

    if len(smooth_patch_noise) < 5:
        return Signal("Sensor noise", 0.2,
                      "Too few smooth regions to profile sensor noise.")

    noise_arr = np.array(smooth_patch_noise)
    noise_level = float(noise_arr.mean())
    noise_consistency = float(noise_arr.std())

    score = 0.0
    if noise_level < 0.4:
        score += 0.55
    elif noise_level < 0.9:
        score += 0.30
    elif noise_level < 1.5:
        score += 0.10
    if noise_consistency < 0.1 and noise_level < 2.0:
        score += 0.15

    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Sensor noise",
        score,
        f"Smooth-region noise level {noise_level:.3f}, consistency {noise_consistency:.3f}. "
        + ("Suspiciously clean or uniform noise."
           if score > 0.4
           else "Natural sensor noise present in flat regions."),
    )


def _metadata_signal(tags: dict) -> Signal:
    if not tags:
        return Signal(
            "Capture metadata",
            0.20,
            "No EXIF metadata. Common for social-media or screen-captured images; "
            "weak prior toward synthetic origin.",
        )
    make = str(tags.get("Make", "")).strip()
    model_name = str(tags.get("Model", "")).strip()
    software = str(tags.get("Software", "")).lower()
    dt = str(tags.get("DateTimeOriginal", tags.get("DateTime", "")))

    if any(s in software for s in
           ("stable diffusion", "midjourney", "dall", "firefly",
            "generated", "invoke", "comfy", "automatic1111")):
        return Signal("Capture metadata", 0.95,
                      f"Software tag identifies a generative tool: '{software}'.")
    if make and model_name:
        return Signal(
            "Capture metadata", 0.05,
            f"Camera: {make} {model_name}{' @ ' + dt if dt else ''}.",
        )
    return Signal("Capture metadata", 0.20,
                  "Partial EXIF — consistent with re-saved or screen-captured images.")


# ---------------------------------------------------------------------------
# Verdict aggregation — Phase 2 weights
# ---------------------------------------------------------------------------

def _verdict(signals: list[Signal], camera_origin: bool) -> tuple[float, str]:
    physics_weight = 0.5 if not camera_origin else 1.1
    weights = {
        "AI-image classifier":  3.0,   # Phase 2: DINOv2 gets highest weight
        "Face classifier":      1.8,
        "Error-level analysis": 0.8,
        "Capture metadata":     0.6,
        "Face boundary":        0.5,
        "Focus variation":      physics_weight,
        "Chromatic aberration": physics_weight,
        "Sensor noise":         physics_weight,
    }
    num = denom = 0.0
    contributing = []
    for s in signals:
        if "does not apply" in s.detail or "unavailable" in s.detail:
            continue
        w = weights.get(s.name, 0.4)
        num += s.score * w
        denom += w
        contributing.append(s.score)
    avg = num / denom if denom else 0.0
    peak = max(contributing, default=0.0)
    score = float(np.clip(0.75 * avg + 0.25 * peak, 0.0, 1.0))

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
    tags = _exif_tags(img)
    camera_origin = _is_camera_origin(tags)

    faces = _detect_faces_mtcnn(img)

    signals = [
        _ai_classifier_signal(img),
        _face_classifier_signal(img, faces),
        _focus_uniformity_signal(cv_img),
        _chromatic_aberration_signal(cv_img),
        _sensor_noise_signal(cv_img),
        _ela_signal(img),
        _metadata_signal(tags),
        _face_boundary_signal(cv_img, faces),
    ]
    score, label = _verdict(signals, camera_origin)
    width, height = img.size

    out: dict[str, Any] = {
        "kind": "image",
        "filename": filename,
        "dimensions": {"width": width, "height": height},
        "suspicion": score,
        "verdict": label,
        "confidence": round(abs(score - 0.5) * 2, 3),
        "camera_origin": camera_origin,
        "signals": [
            {"name": s.name, "score": round(s.score, 3), "detail": s.detail}
            for s in signals
        ],
    }

    if faces:
        out["faces"] = [
            {"box": [x1, y1, x2, y2], "confidence": round(p, 3)}
            for (x1, y1, x2, y2, p) in faces
        ]

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
