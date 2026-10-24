"""Cost record of every candidate encoder (docs/O007_SWEEP.md section 7), measured on one machine.

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.cost_record --task T-V      (also T-I, T-A)

Per candidate: parameters; FLOPs of one forward pass at the native input (torch FlopCounterMode, a count of floating point operations,
2 per multiply-add); peak GPU memory of an inference batch (64 images or 32 four-second clips) in half precision, whole process
including the weights; model-only time per 1,000 inputs on a preloaded batch (median of 3 repeats of 5 batches after 3 warm-up
batches); end-to-end time per 1,000 inputs including decoding and the encoder's own preprocessing on real files (median of 3 repeats
over a fixed set of real inputs, workers as in the extraction). The GPU must otherwise be idle. Written to
data/_results/cost_record.json as {task: {candidate: {...}}}; dfd.sweep.rank uses `end_to_end_s_per_1000` for the tie rule.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from dfd.data import loaders as L
from dfd.sweep import encoders, extract_audio, extract_images
from dfd.sweep.tasks import RESULTS, TASKS

REC = RESULTS / "cost_record.json"
N_E2E = {"T-V": 256, "T-I": 256, "T-A": 128}


def _sync():
    torch.cuda.synchronize()


def timed(fn, repeats=3, calls=5, warm=3) -> float:
    for _ in range(warm):
        fn()
    out = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        _sync()
        out.append((time.perf_counter() - t0) / calls)
    return statistics.median(out)


def flops_of(fn) -> int:
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def measure_image(cid: str, task_id: str) -> dict:
    enc = encoders.load(cid)
    S = enc.size
    batch = 64
    x1 = torch.randn(1, 3, S, S, device="cuda")
    xb = torch.randn(batch, 3, S, S, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    enc.encode_pixels(xb)
    peak = torch.cuda.max_memory_allocated() / 2**30
    model_s = timed(lambda: enc.encode_pixels(xb)) / batch * 1000
    flops = flops_of(lambda: enc.encode_pixels(x1))
    if task_id == "T-V":      # frames: stored 224 px face crops, resized by the encoder wrapper (what the extraction does)
        recs = L.select_records("data/_manifests/ffpp.manifest.csv", "data/_splits/ffpp-037e2976fade.csv", "val", modality="video")[:16]
        ds = L.FaceClipDataset(recs, "data/_crops/ffpp_c23_yunet_n16", data_root="data")
        dl = DataLoader(ds, batch_size=16, num_workers=8, persistent_workers=True)

        def one_pass():
            n = 0
            for b in dl:
                f = b["x"][b["mask"]]
                enc(f.cuda())
                n += f.shape[0]
            return n
    else:
        recs = L.select_records(*extract_images.JOBS[2][1:3], extract_images.JOBS[2][3], modality="image")[:N_E2E[task_id]]
        dl = DataLoader(extract_images.ImgSet(recs, enc.processor), batch_size=64, num_workers=8, persistent_workers=True)

        def one_pass():
            n = 0
            for x, _ in dl:
                enc.encode_pixels(x.cuda())
                n += x.shape[0]
            return n
    one_pass()
    ts, n = [], 0
    for _ in range(3):
        _sync()
        t0 = time.perf_counter()
        n = one_pass()
        _sync()
        ts.append(time.perf_counter() - t0)
    return dict(params=enc.n_params, dim=enc.dim, input_size=S, flops_per_forward=flops, peak_vram_gb=round(peak, 2), batch=batch,
                model_only_s_per_1000=round(model_s, 3), end_to_end_s_per_1000=round(statistics.median(ts) / n * 1000, 3), end_to_end_items=n,
                note="fp16 inference; end-to-end on real files with the extraction's workers")


def measure_audio(cid: str) -> dict:
    enc = encoders.load_audio(cid)
    batch = 32
    w = np.random.default_rng(0).uniform(-0.3, 0.3, (batch, encoders.WINDOW)).astype(np.float32)
    torch.cuda.reset_peak_memory_stats()
    enc(w)
    peak = torch.cuda.max_memory_allocated() / 2**30
    model_s = timed(lambda: enc(w)) / batch * 1000
    flops = flops_of(lambda: enc(w[:1]))
    recs = L.select_records(*extract_audio.JOBS[2][1:3], extract_audio.JOBS[2][3], modality="audio")[:N_E2E["T-A"]]
    dl = DataLoader(extract_audio.AudioSet(recs, cache=None), batch_size=32, num_workers=8, persistent_workers=True)   # decoding included, no sample cache

    def one_pass():
        n = 0
        for x, _ in dl:
            enc(x.numpy())
            n += x.shape[0]
        return n
    one_pass()
    ts, n = [], 0
    for _ in range(3):
        _sync()
        t0 = time.perf_counter()
        n = one_pass()
        _sync()
        ts.append(time.perf_counter() - t0)
    return dict(params=enc.n_params, dim=2 * enc.dim, input_size=encoders.WINDOW, flops_per_forward=flops, peak_vram_gb=round(peak, 2), batch=batch,
                model_only_s_per_1000=round(model_s, 3), end_to_end_s_per_1000=round(statistics.median(ts) / n * 1000, 3), end_to_end_items=n,
                note="fp16 inference; end-to-end includes decoding of In-the-Wild files (no sample cache) and the feature extractor")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    a = ap.parse_args(argv)
    rec = json.loads(REC.read_text()) if REC.is_file() else {}
    task = TASKS[a.task]
    out = {}
    for cid in task.candidates:
        out[cid] = measure_audio(cid) if a.task == "T-A" else measure_image(cid, a.task)
        out[cid].update(date=time.strftime("%Y-%m-%d %H:%M"), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        print(cid, json.dumps(out[cid]), flush=True)
        torch.cuda.empty_cache()
    rec[a.task] = out
    REC.parent.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
