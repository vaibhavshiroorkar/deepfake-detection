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
    # These open_clip encoders need the `probe` extra. Their historical probe
    # ranking used wrong normalization and requires corrected extraction.
    "pe": ("PE-Core-B-16", 224),
    # `-quickgelu`, not plain `ViT-B-16`. OpenAI trained these weights with
    # QuickGELU, and open_clip will load them into the standard-GELU
    # architecture with only a warning. That is the same defect as loading a
    # WavLM checkpoint into a Wav2Vec2 body: it runs, it is subtly the wrong
    # model, and nothing in the output says so.
    "clip": ("ViT-B-16-quickgelu", 224),
}

# Which ENCODERS entries come from open_clip, and the pretrained tag each needs.
OPEN_CLIP_ENCODERS = {
    "pe": "meta",
    "clip": "openai",
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

    `model` takes RGB pixels in [0, 1] and applies its own normalisation, which
    is what `load_encoder` returns. The spread uses `unbiased=False` so a clip
    short enough to produce a single window gets zero rather than NaN. Zero is
    the honest value: nothing varied because there was nothing to vary across.
    """
    import torch

    size = windows.shape[-2]
    flat = torch.from_numpy(windows.reshape(-1, size, size, 3))
    flat = flat.permute(0, 3, 1, 2).float().div_(255.0)

    pieces = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start : start + batch_size].to(device)
            pieces.append(model(batch).float().cpu())
    embedded = torch.cat(pieces).reshape(windows.shape[0], windows.shape[1], -1)
    per_window = embedded.mean(dim=1)
    vector = torch.cat([per_window.mean(dim=0), per_window.std(dim=0, unbiased=False)])
    return vector.numpy()


def _normalised(encode, mean, std):
    """Wrap an encoder so it takes [0, 1] pixels and normalises them itself.

    Each encoder was trained with its own statistics: ImageNet for the timm
    models, 0.5 for Perception Encoder, the OpenAI values for CLIP. Applying
    ImageNet statistics to all of them fed PE inputs scaled about 2.2 times off,
    which runs without error and quietly measures a different model.

    Built here rather than at module scope because this module imports torch
    lazily to keep CLI startup cheap.
    """
    import torch

    class NormalisedEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encode = encode
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, frames):
            return self.encode((frames - self.mean) / self.std)

    return NormalisedEncoder()


def load_encoder(name: str = "dinov3", device: str = "cuda"):
    """The frozen encoder and the input size it expects.

    Two families. The timm names are loaded directly. The open_clip names, which
    are Perception Encoder and CLIP, need the `probe` extra. Both come back
    wrapped to take [0, 1] pixels, so `embed_windows` treats them the same.

    The default stays `dinov3` because the served authenticity gate was fitted
    on DINOv3 features, and a gate queried with another encoder's features
    reports meaningless distances. The leave-one-generator-out probe uses `pe`.
    """
    model_name, size = ENCODERS[name]
    if name in OPEN_CLIP_ENCODERS:
        try:
            import open_clip
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                f"Encoder {name!r} needs open_clip_torch. Install the probe "
                "extra: uv sync --extra probe"
            ) from error
        tag = OPEN_CLIP_ENCODERS[name]
        model = open_clip.create_model(model_name, pretrained=tag)
        config = open_clip.get_pretrained_cfg(model_name, tag)
        wrapped = _normalised(model.encode_image, config["mean"], config["std"])
        wrapped.add_module("model", model)
        return wrapped.to(device).eval(), size

    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    config = model.pretrained_cfg
    wrapped = _normalised(model, config["mean"], config["std"])
    return wrapped.to(device).eval(), size


def normalisation(name: str) -> dict[str, tuple[float, ...]]:
    """The mean and std `load_encoder` applies for `name`, for feature metadata."""
    model_name, _ = ENCODERS[name]
    if name in OPEN_CLIP_ENCODERS:
        import open_clip

        config = open_clip.get_pretrained_cfg(model_name, OPEN_CLIP_ENCODERS[name])
    else:
        import timm

        config = timm.create_model(model_name, pretrained=False).pretrained_cfg
    return {"mean": tuple(config["mean"]), "std": tuple(config["std"])}


def embed_clip(path: Path, model, size: int, device: str = "cuda") -> np.ndarray | None:
    """One video file to one feature vector, or None when it cannot be read."""
    windows = read_windows(Path(path), size=size)
    if windows is None:
        return None
    return embed_windows(model, windows, device)
