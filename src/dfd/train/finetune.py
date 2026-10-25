"""I-1 full fine-tuning of an image encoder on the T-V frames (E-005, D-042, D-051).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.train.finetune --cand IMG-07 --lr 3e-5 --seed 0

One run = the encoder (all vision-tower parameters) plus a linear head, trained for `steps` optimiser steps of `batch` frames on the
FF++ c23 training frames (D-034 crops), then scored on the validation split, the development slice (dev_x) and a fixed strided subset of
the training clips (every 4th, for the train AUC). Settings of D-042 and O007_SWEEP.md section 9: AdamW (betas 0.9, 0.999), weight decay 0.05,
cosine schedule with 5 percent linear warm-up, fp16 autocast with gradient scaling, effective batch 32 (no accumulation), no augmentation,
class-balanced binary cross-entropy with the weights of the I-0 head (w_y = n / (2 n_y) over the training frames), head with the default
PyTorch initialisation. 1,800 steps are one pass over the 57,600 training frames. A run whose training exceeds the cap (600 s), or whose
whole wall time exceeds 900 s, is stopped and recorded as exceeded (no score is made from a stopped run).
Frame order and initialisation depend on the seed only. Clip score = mean of the frame probabilities (D-024). Nothing from test_in or
test_x is read. Output: <out>/<cand>_lr<lr>_s<seed>/{result.json, scores.npz}; weights in checkpoints/I-1/<same name>/weights.pt (not in git).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dfd.data import loaders as L
from dfd.eval.metrics import roc_auc
from dfd.prep.pipeline import folder_name
from dfd.sweep import encoders
from dfd.sweep.step_time import vision_params

STEPS, BATCH, WD, WARM_FRAC, CAP_TRAIN_S, CAP_WALL_S = 1800, 32, 0.05, 0.05, 600.0, 900.0
MAX_WORKERS = 16
I0_SCORES = "data/_results/I-0/T-V/scores.npz"
SPLITS = {   # name -> manifest, split csv, crops dir, split name, key prefix in the I-0 score file
    "train": ("data/_manifests/ffpp.manifest.csv", "data/_splits/ffpp-037e2976fade.csv", "data/_crops/ffpp_c23_yunet_n16", "train", "train"),
    "val": ("data/_manifests/ffpp.manifest.csv", "data/_splits/ffpp-037e2976fade.csv", "data/_crops/ffpp_c23_yunet_n16", "val", "val"),
    "dev": ("data/_manifests/celebdf_v2.manifest.csv", "data/_splits/celebdf_v2-82a3a70594e9.csv", "data/_crops/celebdf_v2_yunet_n16", "dev_x", "dev"),
}
TRAIN_SUBSET_STEP = 4


def clip_table(split: str, limit: int | None = None):
    """Clips of a split in the order of the I-0 score file: sample ids, labels, groups and, per detected frame, (path, clip index)."""
    manifest, split_csv, crops, sname, key = SPLITS[split]
    z = np.load(I0_SCORES, allow_pickle=True)
    sids, y, g = z[f"{key}_sample_id"], z[f"{key}_y"], z[f"{key}_group"]
    sel = np.arange(len(sids))
    recs = {r.sample_id: r for r in L.select_records(manifest, split_csv, sname, modality="video")}
    paths, clip_of = [], []
    for i in sel:
        r = recs[str(sids[i])]
        f = Path(crops) / folder_name(r.sample_id) / "index.csv"
        with open(f, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["detected"] == "1" and row["crop_path"] and int(row["frame_index"]) < 16:
                    paths.append(str(Path(crops) / row["crop_path"]))
                    clip_of.append(int(i))
    t = dict(sample_id=sids[sel], y=y[sel], group=g[sel], paths=np.array(paths), clip_of=np.array(clip_of))
    if limit:
        keep = np.isin(t["clip_of"], np.arange(0, len(sids), max(1, len(sids) // limit))[:limit])
        t["paths"], t["clip_of"] = t["paths"][keep], t["clip_of"][keep]
    return t


class Frames(Dataset):
    def __init__(self, paths, clip_of, y_clip):
        self.paths, self.clip_of, self.y = paths, clip_of, y_clip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable crop {self.paths[i]}")
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return x, float(self.y[self.clip_of[i]]), int(i)


def lr_at(step: int, total: int, lr: float) -> float:
    warm = max(1, int(round(WARM_FRAC * total)))
    if step < warm:
        return lr * (step + 1) / warm
    return 0.5 * lr * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def score_frames(enc, head, ds, workers, dev="cuda"):
    enc.model.eval()
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=workers, persistent_workers=False)
    out = np.zeros(len(ds), np.float32)
    for x, _, idx in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            z = head(enc.train_features(x.to(dev, non_blocking=True))).squeeze(1)
        out[idx.numpy()] = torch.sigmoid(z.float()).cpu().numpy()
    return out


def clip_mean(p, clip_of, n):
    s = np.bincount(clip_of, weights=p, minlength=n)
    c = np.bincount(clip_of, minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def auc_nan(y, s):
    ok = ~np.isnan(s)
    return float(roc_auc(np.asarray(y)[ok], s[ok])) if len(np.unique(np.asarray(y)[ok])) == 2 else float("nan")


def run(cand, lr, seed, out_root, steps=STEPS, workers=10, limit=None, tag=None):
    name = f"{cand}_lr{lr:g}_s{seed}"
    out = Path(out_root) / name
    if (out / "result.json").is_file():
        print("skip", name, flush=True)
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    workers = min(workers, MAX_WORKERS)
    t_wall = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tabs = {s: clip_table(s, limit) for s in SPLITS}
    tr = tabs["train"]
    n_tr = len(tr["paths"])
    n1 = float(tr["y"][tr["clip_of"]].sum())
    n0 = n_tr - n1
    w1, w0 = n_tr / (2 * n1), n_tr / (2 * n0)
    enc = encoders.load(cand)
    ps = vision_params(enc)
    for p in ps:
        p.requires_grad_(True)
    head = torch.nn.Linear(enc.dim, 1).cuda()
    opt = torch.optim.AdamW(ps + list(head.parameters()), lr=lr, weight_decay=WD)
    scaler = torch.amp.GradScaler()
    gen = torch.Generator().manual_seed(seed)
    n_perm = steps * BATCH
    order = torch.cat([torch.randperm(n_tr, generator=gen) for _ in range(-(-n_perm // n_tr))])[:n_perm]
    ds_tr = Frames(tr["paths"], tr["clip_of"], tr["y"])
    sampler = [int(i) for i in order]
    dl = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, num_workers=workers, persistent_workers=False, drop_last=True, prefetch_factor=4)
    torch.cuda.reset_peak_memory_stats()
    enc.model.train()
    losses, curve, nonfinite, events, run_min = [], [], 0, 0, float("inf")
    status, t0 = "ok", time.perf_counter()
    for step, (x, yb, _) in enumerate(dl):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, steps, lr)
        x, yb = x.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            z = head(enc.train_features(x)).squeeze(1)
        w = torch.where(yb == 1, torch.tensor(w1, device="cuda"), torch.tensor(w0, device="cuda"))
        loss = (F.binary_cross_entropy_with_logits(z.float(), yb, reduction="none") * w).mean()
        if not torch.isfinite(loss):
            nonfinite += 1
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            m = float(np.mean(losses[-50:]))
            curve.append((step + 1, round(m, 5)))
            if step + 1 > int(round(WARM_FRAC * steps)):
                run_min = min(run_min, m)
                events += int(m > 5 * run_min)
            print(f"{name} step {step + 1} loss50 {m:.4f} lr {lr_at(step, steps, lr):.2e} {time.perf_counter() - t0:.0f}s", flush=True)
        if time.perf_counter() - t0 > CAP_TRAIN_S:
            status = "exceeded_training_cap"
            break
    train_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    res = dict(name=name, cand=cand, lr=lr, seed=seed, steps_done=len(losses), steps=steps, batch=BATCH, status=status, train_seconds=round(train_s, 1),
               peak_vram_gb=round(peak, 2), n_train_frames=n_tr, class_weights=[w0, w1], nonfinite_losses=nonfinite, instability_events=events,
               loss_curve=curve, trainable_params=sum(p.numel() for p in ps), workers=workers, torch=torch.__version__, smoke=bool(limit or tag),
               date=time.strftime("%Y-%m-%d %H:%M"))
    if status != "ok":
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return res
    # evaluation: validation, dev_x and the strided training subset in one pass
    t1 = time.perf_counter()
    sub_clips = np.arange(0, len(tr["y"]), TRAIN_SUBSET_STEP)
    keep = np.isin(tr["clip_of"], sub_clips)
    sets = {"val": (tabs["val"]["paths"], tabs["val"]["clip_of"], tabs["val"]["y"]), "dev": (tabs["dev"]["paths"], tabs["dev"]["clip_of"], tabs["dev"]["y"]),
            "trsub": (tr["paths"][keep], tr["clip_of"][keep], tr["y"])}
    save, aucs = {}, {}
    for k, (pp, co, yy) in sets.items():
        p = score_frames(enc, head, Frames(pp, co, yy), workers)
        n = len(yy)
        cs = clip_mean(p, co, n)
        if k == "trsub":
            cs, yy_, gg = cs[sub_clips], yy[sub_clips], tr["group"][sub_clips]
            save.update({"trsub_scores": cs, "trsub_y": yy_, "trsub_sample_id": tr["sample_id"][sub_clips]})
            aucs[k] = auc_nan(yy_, cs)
        else:
            save.update({f"{k}_scores": cs, f"{k}_y": yy, f"{k}_group": tabs[k]["group"], f"{k}_sample_id": tabs[k]["sample_id"],
                         f"{k}_frame_p": p.astype(np.float16), f"{k}_frame_clip": co})
            aucs[k] = auc_nan(yy, cs)
    res.update(val_auc=aucs["val"], devx_auc=aucs["dev"], train_subset_auc=aucs["trsub"], eval_seconds=round(time.perf_counter() - t1, 1),
               wall_seconds=round(time.perf_counter() - t_wall, 1))
    if res["wall_seconds"] > CAP_WALL_S:
        res["status"] = "exceeded_wall_cap"
    np.savez(out / "scores.npz", **save)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    if not (limit or tag):
        wdir = Path("checkpoints/I-1") / name
        wdir.mkdir(parents=True, exist_ok=True)
        sd = {n: p.detach().cpu() for n, p in enc.model.named_parameters() if p.requires_grad and any(p is q for q in ps)}
        bufs = {n: b.detach().cpu() for n, b in enc.model.named_buffers()}   # BatchNorm running statistics of the CNN baseline (E-013); empty effect for the ViTs
        torch.save(dict(params=sd, buffers=bufs, head={k: v.cpu() for k, v in head.state_dict().items()}, cand=cand, lr=lr, seed=seed), wdir / "weights.pt")
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}), flush=True)
    del enc, head, opt
    torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="use about N clips per split (a smoke test, not a result)")
    ap.add_argument("--out", default="data/_results/I-1")
    a = ap.parse_args(argv)
    run(a.cand, a.lr, a.seed, a.out, a.steps, a.workers, a.limit, tag="smoke" if a.limit else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
