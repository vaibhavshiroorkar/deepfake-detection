"""Visual enhancement and degradation ops (RGB uint8 in, RGB uint8 out).

These are not part of the stored preprocessing contract. They are robustness and
augmentation probes toggled independently in the dashboard, so the baseline is
all of them off, which is the real pipeline. Enhancement: sharpen, denoise,
clahe. Degradation: gaussian_blur, jpeg_recompress, downscale_upscale.
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def denoise(image: np.ndarray, strength: int) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def clahe(image: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lightness = equalizer.apply(lightness)
    merged = cv2.merge((lightness, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def gaussian_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    size = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image, (size, size), 0)


def jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def downscale_upscale(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, int(width * factor)), max(1, int(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
