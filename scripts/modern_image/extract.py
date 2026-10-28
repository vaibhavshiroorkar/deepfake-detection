"""Extract features and frozen-model scores once per shard (W1-I; E-028 and E-029), cached per shard.

Usage (repo root, PYTHONPATH=src): python scripts/modern_image/extract.py eval|small [workers]
eval : IMG-13 features, the frozen T-I score (calibrated), YuNet face count, the frozen T-V face-crop score when a face is found, IMG-15
       features (stored for E-029 under data/_results/modern/e029/cache_eval15/), and the JPEG-90 re-encoding control for a fixed sample.
small: IMG-13 and IMG-15 features only, to data/_results/modern/e029/cache_small/.
Only shards whose SHA-256 matched (download_log.jsonl) are read; finished shards are skipped, so the script can be rerun as shards arrive.
NSFW-flagged rows are skipped and counted. Every row's image_name, label and model_name are checked against the metadata index.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import _env  # noqa: E402


def reencode_keys(rows):
    """Fixed sample for control C2: PNG fakes, up to 20 per generator, numpy default_rng(20261027) over the index order."""
    rng = np.random.default_rng(lib.SEED)
    by = {}
    for r in rows:
        if r["label"] == "1" and r["format"] == "PNG":
            by.setdefault(r["model_name"], []).append((r["shard"], int(r["row"])))
    keep = set()
    for g in sorted(by):
        lst = by[g]
        for i in rng.permutation(len(lst))[:20]:
            keep.add(lst[i])
    return keep


def main():
    which = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    import torch
    from dfd.infer.real_adapter import RealAdapter, _calibrate
    from dfd.sweep import encoders
    rows = lib.subset_rows(which)
    idx = {(r["shard"], int(r["row"])): r for r in rows}
    reenc = reencode_keys(rows) if which == "eval" else set()
    out_dir = _env.RES / ("e028/cache" if which == "eval" else "e029/cache_small")
    out15 = _env.RES / "e029" / "cache_eval15" if which == "eval" else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out15.mkdir(parents=True, exist_ok=True)
    if which == "eval":
        json.dump(sorted([list(k) for k in reenc]), open(_env.RES / "e028" / "reencode_sample.json", "w"))
    adapter = RealAdapter(lib.CKPT, device="cuda")          # checks the checkpoint and the encoder hashes
    enc13 = adapter._encoder("T-I")
    head = adapter.heads["T-I"]
    temp = float(adapter.meta["tasks"]["T-I"]["calibration"]["temperature"])
    enc15 = encoders.load("IMG-15", "cuda")
    m13, s13 = torch.tensor(lib.IMG13_MEAN).view(1, 3, 1, 1).cuda(), torch.tensor(lib.IMG13_STD).view(1, 3, 1, 1).cuda()
    m15, s15 = torch.tensor(lib.IMG15_MEAN).view(1, 3, 1, 1).cuda(), torch.tensor(lib.IMG15_STD).view(1, 3, 1, 1).cuda()

    def feats(px_list):
        px = torch.from_numpy(np.stack(px_list)).cuda()
        f13 = enc13.encode_pixels(px).cpu().numpy()
        f15 = enc15.encode_pixels(((px * s13 + m13) - m15) / s15).cpu().numpy()
        return f13, f15

    shards = lib.verified_shards(which)
    todo = [s for s in sorted(shards) if not (out_dir / (Path(s).stem + ".npz")).exists()]
    print(which, "verified", len(shards), "to do", len(todo), flush=True)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(nw, initializer=lib._init_worker, initargs=(enc13.processor.folder, which == "eval"))
    t0 = time.time()
    n_done = 0
    for s in todo:
        ts = time.time()
        skipped_nsfw, mismatch = 0, 0

        def items():
            nonlocal skipped_nsfw, mismatch
            for row, d in lib.iter_shard(shards[s]):
                r = idx.get((s, row))
                if r is None or r["image_name"] != d["image_name"] or str(d["label"]) != r["label"] or d["model_name"] != r["model_name"]:
                    mismatch += 1
                    continue
                if str(d["nsfw_flag"]).lower() == "true":
                    skipped_nsfw += 1
                    continue
                yield (row, d["image_data"], (s, row) in reenc)
        R = {k: [] for k in ["row", "ok", "h", "w", "n_faces", "ti", "tv"]}
        F13, F15, err = [], [], []
        RE = {"row": [], "ti": []}
        F13r, F15r = [], []
        buf, buf_rows, bre, bre_rows = [], [], [], []

        def flush():
            if buf:
                a, b = feats(buf)
                F13.append(a)
                F15.append(b)
                buf.clear()
            if bre:
                a, b = feats(bre)
                F13r.append(a)
                F15r.append(b)
                RE["row"].extend(bre_rows)
                bre.clear()
                bre_rows.clear()
        for o in pool.imap(lib.work, items(), chunksize=4):
            if not o["ok"]:
                err.append({"row": o["key"], "err": o["err"]})
                continue
            R["row"].append(o["key"])
            R["ok"].append(True)
            R["h"].append(o["h"])
            R["w"].append(o["w"])
            R["n_faces"].append(o["n_faces"])
            R["tv"].append(adapter._score_rgb("T-V", [o["crop"]]) if o["crop"] is not None else np.nan)
            buf.append(o["px"])
            if o["px_re"] is not None:
                bre.append(o["px_re"])
                bre_rows.append(o["key"])
            if len(buf) >= 64 or len(bre) >= 64:
                flush()
        flush()
        f13 = np.concatenate(F13) if F13 else np.zeros((0, 768), np.float32)
        f15 = np.concatenate(F15) if F15 else np.zeros((0, 768), np.float32)
        ti = np.array([_calibrate(float(p), temp) for p in head.prob_rows(f13)]) if len(f13) else np.zeros(0)
        save = dict(row=np.array(R["row"], np.int32), h=np.array(R["h"]), w=np.array(R["w"]), n_faces=np.array(R["n_faces"]),
                    f13=f13.astype(np.float32), ti=ti, tv=np.array(R["tv"], float), err=json.dumps(err), skipped_nsfw=skipped_nsfw,
                    mismatch=mismatch, seconds=time.time() - ts)
        if F13r:
            f13r = np.concatenate(F13r)
            save.update(re_row=np.array(RE["row"], np.int32), re_f13=f13r,
                        re_ti=np.array([_calibrate(float(p), temp) for p in head.prob_rows(f13r)]))
            np.save(out15 / (Path(s).stem + "_re.npy"), np.concatenate(F15r).astype(np.float32))
        if which == "eval":
            np.save(out15 / (Path(s).stem + ".npy"), f15.astype(np.float32))
        else:
            save["f15"] = f15.astype(np.float32)
        np.savez(out_dir / (Path(s).stem + ".npz"), **save)
        n_done += len(R["row"])
        print(f"{s} rows {len(R['row'])} err {len(err)} nsfw {skipped_nsfw} mismatch {mismatch} {time.time() - ts:.1f}s total {n_done} "
              f"{time.time() - t0:.0f}s", flush=True)
    pool.close()
    print("done", n_done, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
