"""From-scratch references of the sweep (IMG-20 for images, AU-REF for speech): reported, never part of the selection.

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.train_reference --task T-V      (also T-I, T-A)

Fixed budget of docs/O007_SWEEP.md section 8: the network is trained end to end on the task's training split for exactly 10 epochs,
AdamW with learning rate 1e-3, no augmentation, class-balanced binary cross-entropy (the same weights as the frozen heads), batch 64
(video: 8 clips of up to 16 frames), no early stopping and no model selection: the weights after epoch 10 are used. Seeds 0, 1 and 2.
Protocol differences to the frozen-feature candidates, stated in every table: end-to-end training, no pretrained weights, fixed epochs.
Output: data/_results/<task>/reference_scores.npz (sample_id of dev_x, scores [seed, clip]) and reference.json (per-seed validation and
dev_x AUC, parameters, seconds). dfd.sweep.rank reads the scores and reports them without using them in the ranking.

IMG-20: four conv blocks (3 x 3 convolution, batch norm, ReLU, 2 x 2 max pool; widths 32, 64, 128, 256), global average pool, linear.
  T-V: the stored 224 px face crops. T-I: the centre 224 x 224 crop of the whole image (no resize, so the pixel statistics stay).
AU-REF: the first 64,600 samples (repeat-padded, the cached windows of E-003) -> log-mel (n_fft 512, hop 160, 64 mel bands, 16 kHz, fixed
  filterbank) -> the same four-block CNN on one channel -> global average pool -> linear.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dfd.data import loaders as L
from dfd.eval.metrics import roc_auc
from dfd.sweep import extract_audio, extract_images
from dfd.prep.pipeline import folder_name
from dfd.sweep.tasks import RESULTS, TASKS

EPOCHS, LR, BATCH, SEEDS = 10, 1e-3, 64, (0, 1, 2)
SR, N_FFT, HOP, N_MELS = 16000, 512, 160, 64


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None) -> np.ndarray:
    """HTK mel triangular filterbank [n_mels, n_fft // 2 + 1] (written here because librosa and torchaudio are not installed)."""
    fmax = fmax or sr / 2
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)      # noqa: E731
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)    # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, bins.size), np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
    return fb


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))
        self.register_buffer("win", torch.hann_window(N_FFT))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:          # [B, L] -> [B, 1, n_mels, frames]
        spec = torch.stft(wav, N_FFT, HOP, window=self.win, return_complex=True).abs() ** 2
        return torch.log(torch.einsum("mf,bft->bmt", self.fb, spec) + 1e-6).unsqueeze(1)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, front: nn.Module | None = None, widths=(32, 64, 128, 256)):
        super().__init__()
        self.front = front
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(c, 1)

    def forward(self, x):
        if self.front is not None:
            x = self.front(x)
        return self.fc(self.body(x).mean(dim=(2, 3))).squeeze(1)


MEAN, STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


class WholeImage(Dataset):
    """Centre 224 x 224 crop of the whole image, uint8 [3, 224, 224]."""

    def __init__(self, records, root="data"):
        self.records, self.root = records, Path(root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        with Image.open(self.root / self.records[i].path) as im:
            a = np.asarray(im.convert("RGB"))
        h, w = a.shape[:2]
        y0, x0 = max(0, (h - 224) // 2), max(0, (w - 224) // 2)
        a = a[y0:y0 + 224, x0:x0 + 224]
        if a.shape[0] < 224 or a.shape[1] < 224:                      # smaller than the window: pad with zeros
            a = np.pad(a, ((0, 224 - a.shape[0]), (0, 224 - a.shape[1]), (0, 0)))
        return torch.from_numpy(a.transpose(2, 0, 1).copy()), self.records[i].label, i


def make_data(task_id: str, split_name: str, which: str):
    """Dataset for a split of a task: returns (dataset, labels, groups, sample_ids)."""
    t = TASKS[task_id]
    if task_id == "T-V":
        dx = which == "devx"
        recs = L.select_records("data/_manifests/celebdf_v2.manifest.csv" if dx else "data/_manifests/ffpp.manifest.csv",
                                "data/_splits/celebdf_v2-82a3a70594e9.csv" if dx else "data/_splits/ffpp-037e2976fade.csv", "dev_x" if dx else which, modality="video")
        crops = "data/_crops/celebdf_v2_yunet_n16" if dx else "data/_crops/ffpp_c23_yunet_n16"
        recs = [r for r in recs if (Path(crops) / folder_name(r.sample_id) / "_done.json").is_file()]
        return L.FaceClipDataset(recs, crops, data_root="data"), recs
    if task_id == "T-I":
        j = {"train": extract_images.JOBS[0], "val": extract_images.JOBS[1], "devx": extract_images.JOBS[2]}[which]
        recs = L.select_records(j[1], j[2], j[3], modality="image")
        return WholeImage(recs), recs
    j = {"train": extract_audio.JOBS[0], "val": extract_audio.JOBS[1], "devx": extract_audio.JOBS[2]}[which]
    recs = L.select_records(j[1], j[2], j[3], modality="audio")
    return extract_audio.AudioSet(recs, cache=extract_audio.CACHE), recs


def batches(task_id, ds, shuffle, seed, workers):
    gen = torch.Generator().manual_seed(seed)
    bs = 8 if task_id == "T-V" else BATCH
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, generator=gen if shuffle else None, drop_last=False,
                      persistent_workers=workers > 0)


def prep(task_id, batch, dev):
    """(inputs on the device, row labels, row -> clip index, row validity) from a batch of the dataset."""
    if task_id == "T-V":
        x, m, y = batch["x"], batch["mask"], batch["label"]
        rows = x[m]
        clip = torch.arange(x.shape[0]).unsqueeze(1).expand_as(m)[m]
        return ((rows.float() / 255 - MEAN.to(rows.device)) / STD.to(rows.device)).to(dev), y[clip].float().to(dev), clip, m.any(1)
    if task_id == "T-I":
        x, y, _ = batch
        return ((x.float() / 255 - MEAN) / STD).to(dev), y.float().to(dev), torch.arange(x.shape[0]), torch.ones(x.shape[0], dtype=torch.bool)
    x, idx = batch
    return x.to(dev), None, torch.arange(x.shape[0]), torch.ones(x.shape[0], dtype=torch.bool)


def build_model(task_id: str) -> nn.Module:
    return SmallCNN(1, LogMel()) if task_id == "T-A" else SmallCNN(3)


@torch.no_grad()
def predict(model, task_id, ds, recs, dev, workers):
    """Clip score = mean of the row probabilities (video: frames), NaN for clips without rows."""
    model.eval()
    scores = np.full(len(recs), np.nan, np.float32)
    pos = 0
    for batch in batches(task_id, ds, False, 0, workers):
        x, _, clip, valid = prep(task_id, batch, dev)
        p = torch.sigmoid(model(x)).float().cpu().numpy() if x.shape[0] else np.zeros(0)
        n = len(valid)
        sums = np.bincount(clip.numpy(), weights=p, minlength=n)
        cnt = np.bincount(clip.numpy(), minlength=n)
        scores[pos:pos + n] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        pos += n
    return scores


def strided(recs, limit):
    """A spread-out subset for a smoke test (the manifests are sorted by class, so the first N would be one class)."""
    step = max(1, len(recs) // limit)
    return recs[::step][:limit]


def labels_of(recs):
    return np.array([r.label for r in recs])


def train_one(task_id: str, seed: int, dev: str, workers: int, limit: int | None):
    torch.manual_seed(seed)
    tr, tr_recs = make_data(task_id, "train", "train")
    if limit:
        tr.records = tr_recs = strided(tr_recs, limit)
    model = build_model(task_id).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    y_all = labels_of(tr_recs)
    n1, n0 = float((y_all == 1).sum()), float((y_all == 0).sum())
    w1, w0 = (n0 + n1) / (2 * n1), (n0 + n1) / (2 * n0)
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        for batch in batches(task_id, tr, True, seed * 1000 + ep, workers):
            if task_id == "T-A":
                x, idx = batch
                y = torch.tensor(y_all[idx.numpy()], dtype=torch.float32, device=dev)
                x = x.to(dev)
                row_y = y
            else:
                x, row_y, _, _ = prep(task_id, batch, dev)
            if x.shape[0] == 0:
                continue
            logit = model(x)
            w = torch.where(row_y == 1, torch.tensor(w1, device=dev), torch.tensor(w0, device=dev))
            loss = (F.binary_cross_entropy_with_logits(logit, row_y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model, time.perf_counter() - t0, sum(p.numel() for p in model.parameters())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training items (a smoke test, not a result)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir or RESULTS) / a.task
    out.mkdir(parents=True, exist_ok=True)
    va_ds, va_recs = make_data(a.task, "val", "val")
    dv_ds, dv_recs = make_data(a.task, "devx", "devx")
    if a.limit:
        va_ds.records = va_recs = strided(va_recs, a.limit)
        dv_ds.records = dv_recs = strided(dv_recs, a.limit)
    per, dev_scores = [], []
    for s in a.seeds:
        model, secs, n_params = train_one(a.task, s, dev, a.workers, a.limit)
        pv = predict(model, a.task, va_ds, va_recs, dev, a.workers)
        pd = predict(model, a.task, dv_ds, dv_recs, dev, a.workers)
        ok_v, ok_d = ~np.isnan(pv), ~np.isnan(pd)
        auc = lambda y, p, ok: float(roc_auc(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else float("nan")  # noqa: E731
        per.append(dict(seed=s, val_auc=auc(labels_of(va_recs), pv, ok_v), devx_auc=auc(labels_of(dv_recs), pd, ok_d), train_seconds=round(secs, 1), params=n_params))
        dev_scores.append(pd)
        print(json.dumps(per[-1]), flush=True)
    np.savez(out / "reference_scores.npz", sample_id=np.array([r.sample_id for r in dv_recs]), scores=np.array(dev_scores, np.float32))
    (out / "reference.json").write_text(json.dumps(dict(
        task=a.task, name="IMG-20" if a.task != "T-A" else "AU-REF", epochs=EPOCHS, lr=LR, batch=BATCH, seeds=a.seeds, per_seed=per, limit=a.limit,
        protocol="end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, class-balanced BCE, last-epoch weights, no model selection",
        date=time.strftime("%Y-%m-%d %H:%M"), torch=torch.__version__), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
