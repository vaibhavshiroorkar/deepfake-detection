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
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags

from . import c2pa as _c2pa
from .heatmap import ela_heatmap, noise_heatmap


# ---------------------------------------------------------------------------
# Probability calibration — temper over-confident pretrained classifiers
# ---------------------------------------------------------------------------

def _calibrate(p: float, center: float = 0.72, slope: float = 6.0) -> float:
    """Sigmoid calibration: pulls low/mid scores toward 0 and only amplifies
    scores well above `center`. Counters the habit of some pretrained
    detectors of returning 0.9+ on ordinary modern photography.

    p=0.3 →0.07, p=0.5 →0.23, p=0.7 →0.47, p=0.8 →0.62,
    p=0.9 →0.74, p=0.95 →0.80, p=0.98 →0.83, p=1.0 →0.85
    """
    return 1.0 / (1.0 + math.exp(-(p - center) * slope))


def _agreement_adjust(anchor_p: float, secondary_p: float) -> float:
    """If a secondary classifier disagrees with the anchor, pull its
    contribution toward 0.5 (uncertain) rather than letting either model
    dominate through an uncorroborated strong call.

    When both agree (gap < 0.2), the secondary is kept as-is.
    Large disagreements (gap > 0.6) pull secondary fully to 0.5.
    """
    gap = abs(anchor_p - secondary_p)
    if gap < 0.2:
        return secondary_p
    pull = min(1.0, (gap - 0.2) / 0.4)
    return secondary_p * (1 - pull) + 0.5 * pull


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
    """Try DINOv2 classifier. Returns None if weights aren't trained or load fails.

    Without trained head weights the classifier emits ~50/50 noise, which is
    worse than the Swin-v2 fallback — so we skip it entirely unless
    VERITAS_DINOV2_WEIGHTS points at a real file.
    """
    try:
        from ..models import dinov2_weights_available, get_dinov2_classifier, get_device
        if not dinov2_weights_available():
            return None
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


def _classify_ai_ensemble_v2(pil_img: Image.Image) -> tuple[float, dict] | None:
    """Second pretrained classifier with different training bias.

    Running two image classifiers trained on different datasets and
    averaging reduces per-model blind spots — especially for diffusion
    content where the primary Swin-v2 (SDXL-focused) tends to under-call.
    """
    try:
        import torch
        from ..models import get_device, get_ensemble_ai_image_classifier

        model, processor, lmap = get_ensemble_ai_image_classifier()
        device = get_device()
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        p_ai = float(probs[lmap["ai_index"]])
        p_real = float(probs[lmap["real_index"]])
        return p_ai, {"p_ai": p_ai, "p_real": p_real, "model": lmap["name"]}
    except Exception:
        return None


def _ai_classifier_ensemble_signal(
    pil_img: Image.Image,
    anchor_p_ai: float | None = None,
) -> Signal:
    """Second pretrained classifier, with two corrections:
       1. Sigmoid calibration — the default `umm-maybe/AI-image-detector`
          is known to return 0.9+ on ordinary real photos, so we shift
          the distribution to require genuine confidence.
       2. Agreement adjustment — if the anchor (primary) classifier
          strongly disagrees, the ensemble's contribution is pulled toward
          0.5 instead of letting an uncorroborated model dominate.
    """
    result = _classify_ai_ensemble_v2(pil_img)
    if result is None:
        return Signal(
            "AI-image ensemble",
            0.0,
            "Second pretrained classifier unavailable; skipping.",
        )
    raw_p, info = result
    calibrated = _calibrate(raw_p)
    if anchor_p_ai is not None:
        adjusted = _agreement_adjust(anchor_p_ai, calibrated)
        disagreement = abs(anchor_p_ai - raw_p)
    else:
        adjusted = calibrated
        disagreement = 0.0

    if adjusted > 0.6:
        tail = "Second classifier also flags this as synthetic."
    elif adjusted < 0.4:
        tail = "Second classifier agrees this looks authentic."
    else:
        tail = "Second classifier is undecided."
    if disagreement > 0.35:
        tail += " Note: disagrees with primary classifier, so the score is held closer to uncertain."
    detail = (
        f"{info['model']}: raw P(AI) = {raw_p:.3f}, calibrated = {adjusted:.3f}. "
        + tail
    )
    return Signal("AI-image ensemble", adjusted, detail)


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
    """The underlying classifier is trained on whole images, not tight face
    crops — its scores on crops are noisier and biased high. We calibrate
    each crop's raw P(AI) through the same sigmoid used for the ensemble
    so a crop scoring 0.93 raw contributes ~0.77 to the verdict, not 0.93.
    """
    if not faces:
        return Signal("Face classifier", 0.0,
                      "No faces detected, so this check does not apply.")
    raw_per_face: list[float] = []
    cal_per_face: list[float] = []
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
        raw_per_face.append(p_ai)
        cal_per_face.append(_calibrate(p_ai))

    raw_avg = float(np.mean(raw_per_face))
    cal_avg = float(np.mean(cal_per_face))
    cal_peak = float(np.max(cal_per_face))
    score = float(np.clip(0.5 * cal_avg + 0.5 * cal_peak, 0.0, 1.0))
    return Signal(
        "Face classifier",
        score,
        f"{len(faces)} face crop(s). raw avg P(AI)={raw_avg:.2f}, "
        f"calibrated={cal_avg:.2f}, peak={cal_peak:.2f}. "
        + ("At least one face crop scores as synthetic after calibration."
           if score > 0.55
           else "Face crops look authentic after calibration."),
    )


def _face_boundary_signal(
    cv_img: np.ndarray,
    faces: list[tuple[int, int, int, int, float]],
) -> Signal:
    if not faces:
        return Signal("Face boundary", 0.0,
                      "No faces detected, so this check does not apply.")
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
        + ("Halo or seam at face boundary, consistent with face-swap."
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
        + ("Unnaturally uniform sharpness, with no depth-of-field blur."
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
        + ("Channels in near-perfect alignment, which is atypical for glass optics."
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
                  "Partial EXIF, consistent with re-saved or screen-captured images.")


# ---------------------------------------------------------------------------
# Verdict aggregation — Phase 2 weights
# ---------------------------------------------------------------------------

def _verdict(signals: list[Signal], camera_origin: bool) -> tuple[float, str]:
    physics_weight = 0.5 if not camera_origin else 1.1
    weights = {
        "AI-image classifier":  2.4,   # Primary Swin-v2 / DINOv2 classifier (anchor)
        "AI-image ensemble":    1.4,   # Second pretrained — calibrated, lower weight
        "Face classifier":      1.2,   # Whole-image classifier on face crops — noisy
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
    # Weighted-mean-dominant blend; a lone spike shouldn't push the verdict.
    score = float(np.clip(0.82 * avg + 0.18 * peak, 0.0, 1.0))

    # Widened "inconclusive" band — real images that tickle one or two
    # over-confident signals stay inconclusive rather than being called out.
    if score < 0.35:
        label = "likely authentic"
    elif score < 0.58:
        label = "inconclusive"
    elif score < 0.78:
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

    # Primary classifier runs first; its score anchors the ensemble so the
    # ensemble can dampen its own contribution when the two disagree.
    primary_signal = _ai_classifier_signal(img)
    anchor_p = primary_signal.score if "unavailable" not in primary_signal.detail else None

    signals = [
        primary_signal,
        _ai_classifier_ensemble_signal(img, anchor_p_ai=anchor_p),
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
