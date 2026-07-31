"""Turning a StreamTrace into pictures.

models/streams/common/introspect.py captures what a model computed and returns
numpy; this module colours it. The split is so the capture can be tested without
a display and the drawing can be reused by any page.

Two rules the whole module follows:

  * Activation maps are upsampled with nearest-neighbour, never interpolated. A
    7x7 stage map drawn as smooth 224x224 gradient looks like the network
    localised something to the pixel. It did not. Blocks are the honest picture.
  * Every map is min-max normalised for display, so colour shows *relative*
    response within one map and nothing across maps. Absolute magnitudes are
    printed as numbers instead.

Colour follows the Preprocessing page: magma for anything spectrogram-like,
#3b82f6 for line plots.
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.ops.constants import IMAGENET_MEAN, IMAGENET_STD

ACTIVATION_CMAP = "magma"
LINE_COLOR = "#3b82f6"
DISPLAY_SIZE = 224
FRAMES_PER_ROW = 8


def show_frames(container, frames, caption: str, per_row: int = FRAMES_PER_ROW):
    """Render a sequence as a compact grid, `per_row` to a line.

    Shared with the Preprocessing page so both walks through a clip lay their
    frames out identically.
    """
    container.caption(caption)
    for start in range(0, len(frames), per_row):
        cols = container.columns(per_row)
        for j, col in enumerate(cols):
            if start + j < len(frames):
                col.image(frames[start + j], width="stretch")


def denormalize(frames: np.ndarray) -> list[np.ndarray]:
    """[T, 3, H, W] ImageNet-normalised floats -> displayable uint8 RGB crops."""
    out = []
    for frame in frames:
        hwc = np.transpose(frame, (1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN
        out.append(np.clip(hwc * 255, 0, 255).astype(np.uint8))
    return out


def heatmap(map2d: np.ndarray, size: int = DISPLAY_SIZE,
            cmap: str = ACTIVATION_CMAP) -> np.ndarray:
    """A [h, w] activation map as an RGB image, blocky on purpose (see module docstring)."""
    # Deferred: introspect pulls torch, and the Preprocessing page imports this
    # module only for show_frames.
    from models.streams.common import introspect

    scaled = introspect.normalize01(map2d)
    big = cv2.resize(scaled, (size, size), interpolation=cv2.INTER_NEAREST)
    colours = plt.get_cmap(cmap)(big)[:, :, :3]
    return (colours * 255).astype(np.uint8)


def overlay(frame: np.ndarray, map2d: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """An activation map blended over the face crop it was computed from."""
    base = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(heatmap(map2d), alpha, base, 1 - alpha, 0)


def channel_grid(detail: np.ndarray, indices: list[int]) -> list[np.ndarray]:
    """One small heatmap per requested channel of a [C, H, W] activation."""
    return [heatmap(detail[i], size=112) for i in indices]


def matrix_fig(matrix: np.ndarray, title: str, xlabel: str, ylabel: str,
               figsize=(10, 2.4)):
    """A [rows, cols] matrix as a heatmap: per-frame features, or an RNN's output."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=ACTIVATION_CMAP,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    return fig


def vector_fig(vector: np.ndarray, title: str, figsize=(10, 1.9)):
    """A 1-D embedding as a stem plot, with zero marked so sign is readable."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(vector.size), vector, width=1.0, color=LINE_COLOR)
    ax.axhline(0, color="#9ca3af", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dimension", fontsize=8)
    ax.margins(x=0)
    return fig


def line_fig(values: np.ndarray, title: str, xlabel: str = "frame", figsize=(10, 1.9)):
    """A per-frame quantity against frame index, with the markers kept visible."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(len(values)), values, marker="o", markersize=3,
            linewidth=1.2, color=LINE_COLOR)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.margins(x=0.01)
    return fig


def attention_fig(weights: np.ndarray, title: str, xlabel: str, ylabel: str,
                  figsize=(5.2, 4.4)):
    """A square attention matrix, drawn square so the diagonal reads as a diagonal."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap=ACTIVATION_CMAP, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    return fig
