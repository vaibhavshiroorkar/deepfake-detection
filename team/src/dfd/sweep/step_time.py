"""Measure full fine-tuning step time and peak VRAM of every image encoder (input to the I-1 budget, D-042).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.step_time [--tag uncontended]

One step = forward, backward and AdamW update of encoder plus a linear head, fp16 autocast with gradient scaling, on random
uint8 224 px frames (the content does not change the cost). Median of 10 timed steps after 3 warm-up steps. A candidate that runs
out of memory is recorded as OOM at that batch size. Results: data/_logs/step_time_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from dfd.sweep import encoders

BATCHES = (16, 32)


def vision_params(enc):
    m = enc.model
    if hasattr(m, "vision_model"):
        ps = list(m.vision_model.parameters())
        if hasattr(m, "visual_projection"):
            ps += list(m.visual_projection.parameters())
        return ps
    return list(m.parameters())


def measure(cid, b):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    enc = encoders.load(cid, timing_only=True)
    enc.model.train()
    ps = vision_params(enc)
    for p in ps:
        p.requires_grad_(True)
    head = torch.nn.Linear(enc.dim, 1).cuda()
    opt = torch.optim.AdamW(ps + list(head.parameters()), lr=1e-5)
    scaler = torch.amp.GradScaler()
    x = torch.randint(0, 256, (b, 3, 224, 224), dtype=torch.uint8, device="cuda")
    y = torch.randint(0, 2, (b, 1), device="cuda").float()
    times = []
    for i in range(13):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(enc.train_features(x)), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t0)
    r = dict(cid=cid, batch=b, step_ms=round(1000 * statistics.median(times), 1), frames_per_s=round(b / statistics.median(times), 1),
             peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), trainable_params=sum(p.numel() for p in ps))
    del enc, opt, head, x, y, ps
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="uncontended")
    ap.add_argument("--cands", default=",".join(encoders.IMAGE_IDS))
    a = ap.parse_args(argv)
    out = []
    for cid in a.cands.split(","):
        for b in BATCHES:
            try:
                r = measure(cid, b)
            except torch.cuda.OutOfMemoryError:
                r = dict(cid=cid, batch=b, oom=True)
            print(json.dumps(r), flush=True)
            out.append(r)
            torch.cuda.empty_cache()
    meta = dict(tag=a.tag, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), date=time.strftime("%Y-%m-%d %H:%M"), results=out)
    with open(f"data/_logs/step_time_{a.tag}.json", "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
