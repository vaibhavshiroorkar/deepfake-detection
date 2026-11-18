"""Where the face model found its evidence: Grad-CAM heatmaps on the face crops it scored.

Grad-CAM weights each channel of the backbone's last convolution by the mean gradient of the "fake" logit
and keeps the positive part, so the map shows what pushed the score toward fake. The same maps rank the
16 frames, so the frames the site cites are the ones the model leaned on. A heatmap shows where the model
looked; it does not prove that area was edited, and the page says so.
"""

from __future__ import annotations

import base64
from collections import Counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MEAN = np.asarray(
    (0.485, 0.456, 0.406), dtype=np.float32
)  # the ImageNet normalization of the face crops
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
SHOWN_SIZE = 224


def _last_conv(model: nn.Module) -> nn.Conv2d:
    convs = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("the face model's backbone has no convolution to explain")
    return convs[-1]


def grad_cam(
    models: list[nn.Module], frames: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Heatmaps (T, H, W) scaled to 0..1 over the whole clip, and each frame's share of the evidence (T,).

    `frames` is the batch the face model scores, shaped (1, T, 3, H, W); the seeds' maps are averaged.
    """
    height, width = frames.shape[-2:]
    total = None
    for model in models:
        captured = {}
        hook = _last_conv(model).register_forward_hook(
            lambda _m, _i, out, store=captured: store.update(maps=out)
        )
        try:
            # cuDNN cannot backpropagate through the GRU in eval mode; eval mode itself must stay.
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                logit = model(
                    frames.detach().clone().requires_grad_(True)
                ).logits.reshape(-1)[0]
                maps = captured["maps"]
                (grads,) = torch.autograd.grad(logit, maps)
        finally:
            hook.remove()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
        total = cam.detach() if total is None else total + cam.detach()
    cams = (total / len(models)).float().cpu().numpy()
    share = cams.sum(axis=(1, 2))
    peak = cams.max()
    return (cams / peak if peak > 0 else cams), share


def region(cam: np.ndarray) -> str:
    """The part of a face crop where most of the heat is. The bands are approximate: the crop is a box
    around the face with a margin, not aligned to landmarks."""
    hot = np.where(cam >= 0.5 * cam.max(), cam, 0.0)
    if hot.sum() <= 0:
        return "whole face"
    ys, xs = np.indices(cam.shape)
    y = float((ys * hot).sum() / hot.sum()) / cam.shape[0]
    x = float((xs * hot).sum() / hot.sum()) / cam.shape[1]
    if x < 0.15 or x > 0.85 or y < 0.1 or y > 0.92:
        return "edge of the face"
    if y < 0.45:
        return "eyes and brow"
    return "nose and cheeks" if y < 0.62 else "mouth and jaw"


def overlay(frame: np.ndarray, cam: np.ndarray) -> str:
    """The face crop with its heatmap laid over it, as a JPEG data URL. Cool areas keep the face visible."""
    rgb = np.clip((frame.transpose(1, 2, 0) * STD + MEAN) * 255.0, 0, 255).astype(
        np.uint8
    )
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (SHOWN_SIZE, SHOWN_SIZE))
    heat_map = cv2.resize(cam.astype(np.float32), (SHOWN_SIZE, SHOWN_SIZE))
    heat = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = 0.65 * heat_map[..., None]
    blended = (bgr * (1 - alpha) + heat * alpha).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the heatmap picture")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def explain_face(
    models: list[nn.Module], frames: torch.Tensor, times, offset: float, top: int = 3
) -> dict:
    """The `top` frames with the most evidence, each with its time in the upload, heatmap and region."""
    cams, share = grad_cam(models, frames)
    strongest = share.max() if share.max() > 0 else 1.0
    order = np.argsort(-share, kind="stable")[:top]
    cited = [
        {
            "time": round(offset + float(times[i]), 2),
            "image": overlay(frames[0, i].detach().float().cpu().numpy(), cams[i]),
            "region": region(cams[i]),
            "strength": round(float(share[i] / strongest), 2),
        }
        for i in order
    ]
    where = Counter(c["region"] for c in cited).most_common(1)[0][0]
    summary = (
        f"The face model's evidence came mostly from the {where}, strongest at {_clock(cited[0]['time'])}. "
        "The heatmap shows where the model looked; it is not proof that this area was edited."
    )
    return {"frames": cited, "summary": summary}
