"""E-011, visual part: fine-tune IMG-17 on LAV-DF head-training frames with the E-005 recipe (label modify_video), then score the fusion-fit subset
(`train` of lavdf_subset) and `val` (and `test_in` only with --final).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/train_visual_lavdf.py --seeds 0 1 2

Recipe unchanged from E-005 (D-058): AdamW, weight decay 0.05, learning rate 1e-5 with 5 percent warm-up and cosine decay, fp16 autocast, batch 32,
1,800 steps, class-balanced BCE, no augmentation, no selection on any split. Only the training frames and the label change. Resumable per seed.
Output: data/_results/I-6/visual_lavdf_s<seed>.npz (raw clip probabilities), weights in checkpoints/I-6/ (not in git).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dfd.data import loaders as L
from dfd.prep.pipeline import folder_name
from dfd.sweep import encoders
from dfd.sweep.step_time import vision_params
from dfd.train import finetune as FT

TRAIN = ("data/_manifests/lavdf_headtrain.manifest.csv", "data/_splits/lavdf_headtrain-b5544b99f6bf.csv", "data/_crops/lavdf_headtrain_yunet_n16", "train")
SUB = ("data/_manifests/lavdf_subset.manifest.csv", "data/_splits/lavdf_subset-e9d12decaf1f.csv", "data/_crops/lavdf_subset_yunet_n16")
OUT = Path("data/_results/I-6")
LR = 1e-5


def table(manifest, split_csv, crops, split, final=False):
    recs = L.select_records(manifest, split_csv, split, modality="video", final=final)
    paths, clip_of = [], []
    for i, r in enumerate(recs):
        with open(Path(crops) / folder_name(r.sample_id) / "index.csv", "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["detected"] == "1" and row["crop_path"] and int(row["frame_index"]) < 16:
                    paths.append(str(Path(crops) / row["crop_path"]))
                    clip_of.append(i)
    y = np.array([int(dict(kv.split("=") for kv in r.notes.split(";"))["modify_video"]) for r in recs])
    return recs, np.array(paths), np.array(clip_of), y


def run(seed, workers, final):
    out = OUT / f"visual_lavdf_s{seed}.npz"
    if out.is_file() and (not final or "test_in_p" in np.load(out, allow_pickle=True).files):
        print("skip", out, flush=True)
        return
    wdir = Path("checkpoints/I-6") / f"IMG-17_lavdf_s{seed}"
    enc = encoders.load("IMG-17")
    head = torch.nn.Linear(enc.dim, 1).cuda()
    if (wdir / "weights.pt").is_file():
        w = torch.load(wdir / "weights.pt", map_location="cpu")
        names = dict(enc.model.named_parameters())
        with torch.no_grad():
            for n, v in w["params"].items():
                names[n].copy_(v.to(names[n].device, names[n].dtype))
        head.load_state_dict(w["head"])
        prev = dict(np.load(out, allow_pickle=True)) if out.is_file() else {}
        log = prev.get("train_log", np.array([]))
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, paths, clip_of, y = table(*TRAIN)
        n_tr = len(paths)
        n1 = float(y[clip_of].sum())
        n0 = n_tr - n1
        w1, w0 = n_tr / (2 * n1), n_tr / (2 * n0)
        ps = vision_params(enc)
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(ps + list(head.parameters()), lr=LR, weight_decay=FT.WD)
        scaler = torch.amp.GradScaler()
        gen = torch.Generator().manual_seed(seed)
        n_perm = FT.STEPS * FT.BATCH
        order = torch.cat([torch.randperm(n_tr, generator=gen) for _ in range(-(-n_perm // n_tr))])[:n_perm]
        dl = DataLoader(FT.Frames(paths, clip_of, y), batch_size=FT.BATCH, sampler=[int(i) for i in order], num_workers=workers, persistent_workers=False, drop_last=True, prefetch_factor=4)
        enc.model.train()
        losses, curve, t0 = [], [], time.perf_counter()
        for step, (x, yb, _) in enumerate(dl):
            for gp in opt.param_groups:
                gp["lr"] = FT.lr_at(step, FT.STEPS, LR)
            x, yb = x.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = head(enc.train_features(x)).squeeze(1)
            wt = torch.where(yb == 1, torch.tensor(w1, device="cuda"), torch.tensor(w0, device="cuda"))
            loss = (F.binary_cross_entropy_with_logits(z.float(), yb, reduction="none") * wt).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if (step + 1) % 100 == 0:
                curve.append((step + 1, round(float(np.mean(losses[-100:])), 4)))
                print(f"seed {seed} step {step + 1} loss100 {curve[-1][1]} {time.perf_counter() - t0:.0f}s", flush=True)
        log = np.array(curve)
        wdir.mkdir(parents=True, exist_ok=True)
        sd = {n: p.detach().cpu() for n, p in enc.model.named_parameters() if p.requires_grad and any(p is q for q in ps)}
        torch.save(dict(params=sd, head={k: v.cpu() for k, v in head.state_dict().items()}, seed=seed), wdir / "weights.pt")
        np.savez(out, train_log=log, train_seconds=time.perf_counter() - t0, n_train_frames=n_tr)
        prev = dict(np.load(out, allow_pickle=True))
    res = dict(prev)
    for split in (["train", "val"] + (["test_in"] if final else [])):
        if f"{split}_p" in res:
            continue
        recs, paths, clip_of, y = table(SUB[0], SUB[1], SUB[2], split, final)
        p = FT.score_frames(enc, head, FT.Frames(paths, clip_of, y), workers)
        res[f"{split}_p"] = FT.clip_mean(p, clip_of, len(recs))
        res[f"{split}_sample_id"] = np.array([r.sample_id for r in recs])
        print("seed", seed, split, "scored; clips without a face frame:", int(np.isnan(res[f"{split}_p"]).sum()), flush=True)
    np.savez(out, **res)
    del enc, head
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        run(s, a.workers, a.final)


if __name__ == "__main__":
    main()
