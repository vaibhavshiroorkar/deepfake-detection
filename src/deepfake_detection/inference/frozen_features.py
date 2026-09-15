"""One clip to one frozen-encoder vector, used by training and by serving.

Both the offline probe and the served abstention gate need the same embedding,
and a second copy of this would eventually drift. A density model fitted on
features computed one way and queried with features computed another reports
distances that mean nothing, and nothing about that fails loudly.

Contiguous windows rather than frames spread across the clip: spread sampling is
what taught this project's visual stream that motion means authenticity,
correlation -0.316 among manipulated clips. Full frames rather than face crops,
because generated video without a person in it has no face to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ENCODERS = {
    "dinov3": ("vit_small_patch16_dinov3.lvd1689m", 224),
    "dinov3-b": ("vit_base_patch16_dinov3.lvd1689m", 224),
}

WINDOWS = 3
WINDOW_LENGTH = 32


def window_starts(total: int, count: int, length: int) -> list[int]:
    """Start indices for `count` contiguous windows spread across the clip."""
    if total <= length:
        return [0]
    if count == 1:
        return [max(0, (total - length) // 2)]
    span = total - length
    return [round(span * i / (count - 1)) for i in range(count)]


def read_windows(
    path: Path, count: int = WINDOWS, length: int = WINDOW_LENGTH, size: int = 224
) -> np.ndarray | None:
    """Decoded windows as uint8, [window, frame, size, size, 3], or None."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = length * count
    starts = window_starts(total, count, length)
    wanted = {index for start in starts for index in range(start, start + length)}

    frames: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames[index] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        index += 1
        if index > max(wanted, default=0):
            break
    capture.release()
    if not frames:
        return None

    ordered = sorted(frames)
    stacked = []
    for start in starts:
        picked = [frames[i] for i in range(start, start + length) if i in frames]
        if not picked:
            picked = [frames[ordered[0]]]
        while len(picked) < length:
            picked.append(picked[-1])
        stacked.append(np.stack(picked))
    return np.stack(stacked)


def embed_windows(model, windows: np.ndarray, device: str, batch_size: int = 32):
    """Mean and spread across windows, concatenated.

    The spread uses `unbiased=False` so a clip short enough to produce a single
    window gets zero rather than NaN. Zero is the honest value: nothing varied
    because there was nothing to vary across.
    """
    import torch

    from deepfake_detection.preprocessing.ops.constants import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    size = windows.shape[-2]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    flat = torch.from_numpy(windows.reshape(-1, size, size, 3))
    flat = flat.permute(0, 3, 1, 2).float().div_(255.0)

    pieces = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start : start + batch_size].to(device)
            batch = (batch - mean) / std
            pieces.append(model(batch).float().cpu())
    embedded = torch.cat(pieces).reshape(windows.shape[0], windows.shape[1], -1)
    per_window = embedded.mean(dim=1)
    vector = torch.cat(
        [per_window.mean(dim=0), per_window.std(dim=0, unbiased=False)]
    )
    return vector.numpy()


def load_encoder(name: str = "dinov3", device: str = "cuda"):
    """The frozen encoder and the input size it expects."""
    import timm

    model_name, size = ENCODERS[name]
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model.to(device).eval(), size


def embed_clip(
    path: Path, model, size: int, device: str = "cuda"
) -> np.ndarray | None:
    """One video file to one feature vector, or None when it cannot be read."""
    windows = read_windows(Path(path), size=size)
    if windows is None:
        return None
    return embed_windows(model, windows, device)
