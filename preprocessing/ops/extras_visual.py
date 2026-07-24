"""EXTRAS — visual enhancement/degradation ops (RGB uint8 in, RGB uint8 out).

These are NOT part of the stored preprocessing contract. They are robustness /
augmentation probes toggled independently in the dashboard (baseline = all off =
the real pipeline). Enhancement: sharpen, denoise, clahe. Degradation: blur,
jpeg_recompress, downscale_upscale. See docs/preprocessing.md.
"""
import cv2
import numpy as np


def sharpen(img, amount: float):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def denoise(img, strength: int):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def clahe(img, clip_limit: float):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def gaussian_blur(img, kernel: int):
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg_recompress(img, quality: int):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def downscale_upscale(img, factor: float):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
