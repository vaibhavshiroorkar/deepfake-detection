"""E-002 feature extraction: frozen image encoders on whole generated images (docs/O007_SWEEP.md, D-036).

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_images --cand IMG-07

Whole images (no face detection), preprocessed by each encoder's own rule (resize, crop, normalisation of its model card, applied by
`Encoder.processor`). Jobs: ProGAN validation set, classes of the `train` split (14) and of the `val` split (6), and the `dev_x` slice of
the CNN-synthesis set (2 held-out generator units, D-035 Amendment 1). Features are float16, one row per image, in
data/_features/<cand>/T-I/<dataset>_<split>.npz with a .json sidecar written last (resumable: finished jobs are skipped).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dfd.data import loaders as L
from dfd.sweep import encoders

JOBS = [  # dataset name, manifest, split file, split
    ("progan_train", "data/_manifests/forensynths_progan_val.manifest.csv", "data/_splits/progan_val-b42dc9619141.csv", "train"),
    ("progan_val", "data/_manifests/forensynths_progan_val.manifest.csv", "data/_splits/progan_val-b42dc9619141.csv", "val"),
    ("cnn_dev_x", "data/_manifests/forensynths_cnn_subset.manifest.csv", "data/_splits/cnn_subset-4e973bc4da0f.csv", "dev_x"),
]
OUT = Path("data/_features")


class ImgSet(Dataset):
    def __init__(self, records, proc, root="data"):
        self.records, self.proc, self.root = records, proc, Path(root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(self.root / r.path) as im:
            return self.proc(im.convert("RGB")), i


def run_job(enc, name, manifest, split_csv, split, workers, batch, out_root=OUT):
    out = out_root / enc.cid / "T-I" / f"{name}.npz"
    if out.is_file() and out.with_suffix(".json").is_file():
        print("skip", out, flush=True)
        return
    recs = L.select_records(manifest, split_csv, split, modality="image")
    dl = DataLoader(ImgSet(recs, enc.processor), batch_size=batch, num_workers=workers, shuffle=False)
    feats = np.zeros((len(recs), enc.dim), np.float16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for x, idx in dl:
        feats[idx.numpy()] = enc.encode_pixels(x.cuda(non_blocking=True)).cpu().numpy().astype(np.float16)
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=feats, label=np.array([r.label for r in recs]), group_id=np.array([r.group_id for r in recs]),
             sample_id=np.array([r.sample_id for r in recs]))
    meta = dict(cand=enc.cid, encoder=enc.name, revision=enc.revision, dim=enc.dim, params=enc.n_params, input_size=enc.size, dataset=name,
                split=split, n_items=len(recs), seconds=round(dt, 1), items_per_s=round(len(recs) / dt, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                preprocessing="the encoder's own rule (model card processor)", note="end-to-end incl. image decoding, one run, not the section 7 cost record")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-jobs", type=int, default=None)
    ap.add_argument("--out-root", default=str(OUT), help="use another folder for a test run")
    a = ap.parse_args(argv)
    enc = encoders.load(a.cand)
    for job in JOBS[:a.limit_jobs]:
        run_job(enc, *job, a.workers, a.batch, Path(a.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
