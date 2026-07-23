"""Pure per-step visual preprocessing ops (RGB uint8 in, RGB uint8 out).

No Streamlit, no I/O — unit-testable and reusable. Enhancement steps (sharpen,
denoise, clahe) and degradation/robustness steps (blur, jpeg, downscale) share
this signature so pages can toggle them independently.
"""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def mouth_region(face_224, size: int = 96):
    h, w = face_224.shape[:2]
    crop = face_224[int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def imagenet_normalize(img_uint8) -> np.ndarray:
    arr = img_uint8.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def normalized_range(arr) -> tuple[float, float]:
    return float(arr.min()), float(arr.max())
