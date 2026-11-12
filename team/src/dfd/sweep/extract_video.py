"""E-001 feature extraction: frozen image encoders on face-crop frames (docs/O007_SWEEP.md section 4).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_video --cand IMG-07

For every (dataset, split) job the features of all 16 frames of every clip are stored as float16 in
data/_features/<cand>/T-V/<dataset>_<split>.npz (feats [N, T, D], mask [N, T], label, group_id, sample_id) with a
.json sidecar (preprocess_id, encoder revision, parameters, seconds, peak VRAM, clips missing crops). Masked frames
(no face detected) are zero and must be ignored by the head. Finished jobs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfd.data import loaders as L
from dfd.prep.pipeline import folder_name
from dfd.sweep import encoders

JOBS = [  # name, manifest, split file, crops dir, split
    ("ffpp", "data/_manifests/ffpp.manifest.csv", "data/_splits/ffpp-037e2976fade.csv", "data/_crops/ffpp_c23_yunet_n16", "train"),
    ("ffpp", "data/_manifests/ffpp.manifest.csv", "data/_splits/ffpp-037e2976fade.csv", "data/_crops/ffpp_c23_yunet_n16", "val"),
    ("celebdf", "data/_manifests/celebdf_v2.manifest.csv", "data/_splits/celebdf_v2-82a3a70594e9.csv", "data/_crops/celebdf_v2_yunet_n16", "dev_x"),
]
OUT = Path("data/_features")


def run_job(enc, name, manifest, split_csv, crops, split, workers, batch_clips):
    out = OUT / enc.cid / "T-V" / f"{name}_{split}.npz"
    if out.is_file() and out.with_suffix(".json").is_file():
        print("skip", out, flush=True)
        return
    recs = L.select_records(manifest, split_csv, split, modality="video")
    have = [r for r in recs if (Path(crops) / folder_name(r.sample_id) / "_done.json").is_file()]
    ds = L.FaceClipDataset(have, crops, data_root="data")
    dl = DataLoader(ds, batch_size=batch_clips, num_workers=workers, shuffle=False)
    N, T = len(ds), ds.n_frames
    feats = np.zeros((N, T, enc.dim), np.float16)
    mask = np.zeros((N, T), bool)
    labels, groups, sids = [], [], []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    i = 0
    for b in dl:
        x, m = b["x"], b["mask"]
        B = x.shape[0]
        flat = x[m]
        if flat.shape[0]:
            f = torch.cat([enc(flat[k:k + 256].cuda(non_blocking=True)) for k in range(0, flat.shape[0], 256)]).cpu()
            tmp = torch.zeros(B, T, enc.dim)
            tmp[m] = f
            feats[i:i + B] = tmp.numpy().astype(np.float16)
        mask[i:i + B] = m.numpy()
        labels += [int(v) for v in b["label"]]
        groups += list(b["group_id"])
        sids += list(b["sample_id"])
        i += B
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=feats, mask=mask, label=np.array(labels), group_id=np.array(groups), sample_id=np.array(sids))
    meta = dict(cand=enc.cid, encoder=enc.name, revision=enc.revision, dim=enc.dim, params=enc.n_params, input_size=enc.size,
                dataset=name, split=split, n_clips=N, n_clips_listed=len(recs), n_missing_crops=len(recs) - len(have),
                n_frames=int(mask.sum()), preprocess_id=ds.preprocess_id, seconds=round(dt, 1),
                frames_per_s=round(float(mask.sum()) / dt, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0), note="end-to-end incl. crop decoding, one run, not the section 7 cost record")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-clips", type=int, default=16)
    ap.add_argument("--limit-jobs", type=int, default=None)
    a = ap.parse_args(argv)
    enc = encoders.load(a.cand)
    for job in JOBS[:a.limit_jobs]:
        run_job(enc, *job, a.workers, a.batch_clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
