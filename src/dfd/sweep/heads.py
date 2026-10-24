"""Linear head on frozen features (docs/O007_SWEEP.md section 5, D-036).

Features are standardised with the training mean and standard deviation; one linear layer; binary cross-entropy with class
weights w_y = n / (2 n_y) (a balanced prior); AdamW, learning rate 1e-3, weight decay 1e-4; batches of 1,024; at most
200 epochs; early stopping on the validation AUC (patience 10). The seed changes the initialisation and the batch order.
Clip scores are the mean of the frame probabilities (D-024); items without frames have no score (NaN) and are excluded by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dfd.eval.metrics import roc_auc

LR, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 1024, 200, 10


@dataclass
class Split:
    """Frames or items of one split. `clip_of[i]` is the clip of row i of `X`; `y` and `groups` are per clip."""
    X: np.ndarray            # [M, D] float32
    clip_of: np.ndarray      # [M] int
    y: np.ndarray            # [N] int
    groups: np.ndarray       # [N] str
    sample_id: np.ndarray    # [N] str


def clip_scores(p_rows: np.ndarray, clip_of: np.ndarray, n: int) -> np.ndarray:
    """Mean of the row probabilities per clip; NaN for clips without rows."""
    s = np.bincount(clip_of, weights=p_rows, minlength=n)
    c = np.bincount(clip_of, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = ~np.isnan(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return roc_auc(y[ok], s[ok])


@dataclass
class HeadResult:
    seed: int
    epochs: int
    best_epoch: int
    val_auc: float
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float

    def prob_rows(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.std) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def train_head(train: Split, val: Split, seed: int) -> HeadResult:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    X = train.X.astype(np.float32)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mean) / std)
    y_rows = torch.from_numpy(train.y[train.clip_of].astype(np.float32))
    n1, n0 = float(y_rows.sum()), float((1 - y_rows).sum())
    n = n1 + n0
    wts = torch.where(y_rows == 1, torch.tensor(n / (2 * n1)), torch.tensor(n / (2 * n0)))
    lin = torch.nn.Linear(Xt.shape[1], 1)
    opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=WD)
    Xv = torch.from_numpy(((val.X - mean) / std).astype(np.float32))
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    ep = 0
    for ep in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(Xt.shape[0], generator=gen)
        for k in range(0, perm.numel(), BATCH):
            idx = perm[k:k + BATCH]
            logit = lin(Xt[idx]).squeeze(1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y_rows[idx], reduction="none") * wts[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(lin(Xv).squeeze(1)).numpy()
        auc = _auc(val.y, clip_scores(pv, val.clip_of, val.y.size))
        if auc > best + 1e-9:
            best, best_ep, bad = auc, ep, 0
            best_state = (lin.weight.detach().clone().squeeze(0).numpy(), float(lin.bias.detach().item()))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:  # validation AUC was undefined in every epoch: keep the last weights, the caller sees val_auc nan
        best_state = (lin.weight.detach().clone().squeeze(0).numpy(), float(lin.bias.detach().item()))
        best = float('nan')
    w, b = best_state
    return HeadResult(seed, ep, best_ep, best, mean, std, w, b)
