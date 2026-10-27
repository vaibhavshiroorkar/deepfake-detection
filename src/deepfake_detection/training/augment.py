"""Training-time augmentation for the visual branch, applied to cached crops.

Two measured failures motivate it.

**The codec shortcut.** FakeAVCeleb encodes its genuine video as H.264 and its
face-swap plus wav2lip fakes as MPEG-4 Part 2. Re-encoding genuine test clips as
MPEG-4 made the served ensemble call 93 percent of them fake, and re-encoding
the fakes as H.264 dropped detection from 100 to 31 percent
(`scripts/codec_shortcut_check.py`). A model that reads the encoder cannot
transfer to footage encoded any other way. Compression here is applied to both
classes at random, so the encoder stops predicting the label.

**Few genuine faces and no blending signal.** FakeAVCeleb has 349 genuine video
tracks, and an unseen face-swap family (FaceSwap) sits at chance (F34). A
self-blended image (Shiohara and Yamasaki, CVPR 2022) turns a genuine crop into
a pseudo-fake: the face region is blended with a slightly altered copy of
itself, which leaves a blending boundary and nothing else to learn. Every
genuine clip becomes a source of fakes, and the model learns the artefact that
face swaps share rather than the identity or generator of one corpus.

Everything works on the cached view: 16 crops, ImageNet-normalised CHW floats.
Parameters are drawn once per clip and applied to every frame, so a clip stays
temporally consistent the way a real recording or a real forgery is.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np

from deepfake_detection.preprocessing.ops.constants import IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

CODECS = ("jpeg", "h264", "mpeg4")


@dataclass(frozen=True, slots=True)
class VisualAugmentConfig:
    # Probability that a clip goes through a compression round trip. High on
    # purpose: the original codec has to stop being the cheapest cue, which a
    # light touch would leave in place.
    compression: float = 0.8
    # Probability that a genuine clip becomes a self-blended fake.
    self_blend: float = 0.0
    downscale: float = 0.3
    blur: float = 0.2
    color: float = 0.5
    flip: float = 0.5

    def __post_init__(self) -> None:
        for name in ("compression", "self_blend", "downscale", "blur", "color", "flip"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability, got {value}")


def to_pixels(view: np.ndarray) -> np.ndarray:
    """ImageNet-normalised [T, 3, H, W] floats to uint8 [T, H, W, 3]."""
    values = np.asarray(view, dtype=np.float32) * _STD + _MEAN
    return np.clip(values * 255.0 + 0.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


def to_view(pixels: np.ndarray) -> np.ndarray:
    """uint8 [T, H, W, 3] back to ImageNet-normalised [T, 3, H, W] floats."""
    values = pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return ((values - _MEAN) / _STD).astype(np.float32)


def codec_roundtrip(
    pixels: np.ndarray, codec: str, quality: float, rng: np.random.Generator
) -> np.ndarray:
    """Encode and decode a frame sequence. `quality` is 0 (worst) to 1 (best)."""
    if codec == "jpeg":
        level = int(30 + 65 * quality)
        out = []
        for frame in pixels:
            ok, encoded = cv2.imencode(
                ".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, level]
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            out.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1])
        return np.stack(out)
    import av

    height, width = pixels.shape[1:3]
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="matroska") as container:
        stream = container.add_stream(
            "libx264" if codec == "h264" else "mpeg4", rate=25
        )
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        if codec == "h264":
            stream.options = {"crf": str(int(40 - 17 * quality)), "preset": "veryfast"}
        else:
            stream.bit_rate = int(120_000 + 480_000 * quality)
        for frame in pixels:
            image = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), "rgb24")
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    with av.open(buffer, mode="r") as container:
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if len(decoded) != len(pixels):
        # An encoder that drops or duplicates a frame would change the clip's
        # timing, so the untouched frames are kept instead.
        return pixels
    return np.stack(decoded)


def face_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """A soft elliptical mask over the face, [H, W] floats in [0, 1].

    The cached crop is the face box widened by 1.4, so the face fills roughly
    the central 70 percent. The ellipse is jittered in size and position, and
    its edge is blurred so the boundary is a gradient rather than a step.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    centre = (
        int(width * (0.5 + rng.uniform(-0.04, 0.04))),
        int(height * (0.53 + rng.uniform(-0.04, 0.04))),
    )
    axes = (
        int(width * rng.uniform(0.24, 0.34)),
        int(height * rng.uniform(0.30, 0.40)),
    )
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    sigma = rng.uniform(3, 12)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return mask / max(float(mask.max()), 1e-6)


def self_blend(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blend each frame's face region with an altered copy of itself."""
    count, height, width = pixels.shape[:3]
    mask = face_mask(height, width, rng)[..., None] * rng.choice((0.5, 0.75, 1.0))
    shift = rng.uniform(-18, 18, size=3).astype(np.float32)
    contrast = rng.uniform(0.85, 1.15)
    scale = rng.uniform(0.95, 1.05)
    dx, dy = rng.uniform(-3, 3, size=2)
    sharpen = rng.random() < 0.5
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, scale)
    matrix[:, 2] += (dx, dy)
    out = np.empty_like(pixels)
    for index in range(count):
        frame = pixels[index].astype(np.float32)
        source = (frame - frame.mean()) * contrast + frame.mean() + shift
        if sharpen:
            source = cv2.addWeighted(
                source, 1.5, cv2.GaussianBlur(source, (0, 0), 2), -0.5, 0
            )
        else:
            source = cv2.GaussianBlur(source, (0, 0), 1.0)
        source = cv2.warpAffine(
            source, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
        )
        blended = mask * source + (1 - mask) * frame
        out[index] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _downscale(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = pixels.shape[1:3]
    factor = rng.uniform(0.4, 0.9)
    small = (max(8, int(width * factor)), max(8, int(height * factor)))
    return np.stack(
        [
            cv2.resize(
                cv2.resize(f, small, interpolation=cv2.INTER_AREA),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            for f in pixels
        ]
    )


def _color(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(0.8, 1.2)
    bias = rng.uniform(-20, 20)
    return np.clip(pixels.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


class VisualAugmenter:
    """Callable used by `CachedBranchDataset`: (view, label) -> (view, label)."""

    def __init__(self, config: VisualAugmentConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def _generator(self) -> np.random.Generator:
        # Created lazily, inside each DataLoader worker, mixed with the worker's
        # torch seed, so workers do not replay one another's draws.
        if self._rng is None:
            import torch

            self._rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**32])
        return self._rng

    def __call__(self, view: np.ndarray, label: float) -> tuple[np.ndarray, float]:
        rng = self._generator()
        config = self.config
        pixels = to_pixels(view)
        if label < 0.5 and rng.random() < config.self_blend:
            pixels = self_blend(pixels, rng)
            label = 1.0
        if rng.random() < config.flip:
            pixels = pixels[:, :, ::-1]
        if rng.random() < config.color:
            pixels = _color(pixels, rng)
        if rng.random() < config.blur:
            sigma = rng.uniform(0.5, 1.5)
            pixels = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in pixels])
        if rng.random() < config.downscale:
            pixels = _downscale(pixels, rng)
        if rng.random() < config.compression:
            codec = CODECS[int(rng.integers(len(CODECS)))]
            pixels = codec_roundtrip(
                np.ascontiguousarray(pixels), codec, rng.random(), rng
            )
        return to_view(np.ascontiguousarray(pixels)), label


class CodecSwap:
    """Validation transform: each class re-encoded with the other's codec.

    Genuine clips go through MPEG-4 and fakes through H.264, the swap that
    flipped the served ensemble's verdicts. Plain FakeAVCeleb validation sits
    at ROC-AUC 1.0 on every epoch, so it cannot tell a checkpoint that reads
    faces from one that reads the encoder. This can, and it is built from
    validation clips alone, so no test corpus informs selection.

    Deterministic: fixed settings per class, no random draw.
    """

    def __call__(self, view: np.ndarray, label: float) -> tuple[np.ndarray, float]:
        pixels = np.ascontiguousarray(to_pixels(view))
        codec, quality = ("h264", 0.5) if label >= 0.5 else ("mpeg4", 0.4)
        swapped = codec_roundtrip(pixels, codec, quality, np.random.default_rng(0))
        return to_view(np.ascontiguousarray(swapped)), label
