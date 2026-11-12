"""Offset search and confidence for SyncNet v2, following syncnet_python SyncNetInstance.py (MIT, Copyright (c) 2016-present Joon Son Chung).

Reference lines (SI = SyncNetInstance.py): calc_pdist lines 129-141; feature loop lines 220-238; offset, confidence lines 246-252.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .model import S
from .preprocess import MFCC_PER_FRAME, MFCC_WINDOW, SAMPLES_PER_FRAME, VIDEO_WINDOW

VSHIFT = 15        # run_syncnet.py --vshift default '15' (searched offsets: -15..+15 frames, +-0.6 s at 25 fps)
BATCH_SIZE = 20    # run_syncnet.py --batch_size default '20'


def calc_pdist(feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = VSHIFT) -> list:
    """SI lines 129-141, same logic: for each video feature the distances to the audio features at 2*vshift+1 shifts."""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    return [torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :])
            for i in range(0, len(feat1))]


@dataclass
class SyncResult:
    offset_frames: int       # SI line 251: vshift - argmin of the mean distance
    min_dist: float          # SI line 249
    confidence: float        # SI line 252: median of the mean distance minus its minimum
    n_windows: int           # number of 5-frame windows scored
    vshift: int
    mean_dist: np.ndarray    # (2*vshift+1,) mean distance per candidate shift


def scorable_windows(n_frames: int, n_samples: int) -> int:
    """SI lines 214 and 220: min_length = min(len(images), floor(len(audio)/640)); windows scored = min_length - 5."""
    return min(n_frames, math.floor(n_samples / SAMPLES_PER_FRAME)) - VIDEO_WINDOW


@torch.inference_mode()
def score_track(model: S, crops: np.ndarray, mfcc_ct: np.ndarray, n_samples: int, device: str, vshift: int = VSHIFT,
                batch_size: int = BATCH_SIZE) -> SyncResult:
    """``crops``: (T, 224, 224, 3) uint8 BGR face crops at 25 fps; ``mfcc_ct``: (13, M) float32 from preprocess.audio_mfcc_tensor;
    ``n_samples``: audio length in samples. Crops and audio must start at the same instant.
    Raises ValueError if no more than ``vshift`` windows are available."""
    model.eval()
    lastframe = scorable_windows(len(crops), n_samples)
    if lastframe <= vshift:
        raise ValueError(f"only {lastframe} scorable windows, need more than {vshift}")
    vid = torch.from_numpy(np.ascontiguousarray(crops[:lastframe + VIDEO_WINDOW - 1])).permute(3, 0, 1, 2).float()   # (3, T, H, W), BGR 0-255 as cv2.imread
    cct = torch.from_numpy(mfcc_ct).float()[None, None]                                                              # (1, 1, 13, M)
    im_feat, cc_feat = [], []
    for i in range(0, lastframe, batch_size):
        idx = range(i, min(lastframe, i + batch_size))
        im_in = torch.stack([vid[:, v:v + VIDEO_WINDOW] for v in idx], 0).to(device)                                 # (B, 3, 5, H, W)
        im_feat.append(model.forward_lip(im_in).cpu())
        cc_in = torch.cat([cct[:, :, :, v * MFCC_PER_FRAME:v * MFCC_PER_FRAME + MFCC_WINDOW] for v in idx], 0).to(device)
        cc_feat.append(model.forward_aud(cc_in).cpu())
    im_feat, cc_feat = torch.cat(im_feat, 0), torch.cat(cc_feat, 0)
    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    return SyncResult(offset_frames=int(vshift - minidx.item()), min_dist=float(minval.item()),
                      confidence=float((torch.median(mdist) - minval).item()), n_windows=int(lastframe), vshift=vshift,
                      mean_dist=mdist.numpy())
